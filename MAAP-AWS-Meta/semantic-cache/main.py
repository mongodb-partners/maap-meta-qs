import json
import os

import boto3
import pymongo
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from logger import AsyncRemoteLogger
from datetime import datetime, timezone
from typing import Optional

# Create an instance of the logger
logger = AsyncRemoteLogger(
    service_url="http://logger:8181", app_name="MAAP-AWS-Meta-Semantic-Cache"
)
# Load environment variables from .env file
load_dotenv()

# MongoDB Atlas Connection
MONGODB_URI = os.getenv("MONGODB_URI")

client = pymongo.MongoClient(MONGODB_URI)
db = client["semantic_cache"]
cache_collection = db["cache"]
VECTOR_SEARCH_INDEX_NAME = "cache_vector_index"

# AWS Bedrock Configuration (for Embeddings)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


# Initialize FastAPI app with title and version
app = FastAPI(title="Semantic Cache Service", version="1.0")


# Ensure collections exist; create if not already created
if cache_collection.name not in cache_collection.database.list_collection_names():
    cache_collection.database.create_collection(cache_collection.name)

    # Define index definitions for vector
    index_definitions = [
        {
            "name": VECTOR_SEARCH_INDEX_NAME,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1536,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "user_id"},
                ]
            },
        }
    ]

    # Create the defined search indexes on the cache_collection
    for index_definition in index_definitions:
        try:
            cache_collection.create_search_index(model=index_definition)
        except pymongo.errors.PyMongoError:
            # If an error occurs during index creation, simply pass
            pass
    try:
        # Create a TTL index on the timestamp field
        cache_collection.create_index(
            [("timestamp", 1)],
            expireAfterSeconds=3600,  # TTL of 1 hour
            name="timestamp_ttl_idx",
        )
    except pymongo.errors.PyMongoError:
        # If an error occurs during index creation, simply pass
        pass


class QueryRequest(BaseModel):
    user_id: str
    query: str


class CacheEntry(BaseModel):
    user_id: str
    query: str
    embedding: list | None
    response: str
    timestamp: datetime | None

    def parse_timestamp(self, timestamp: Optional[str]) -> datetime:
        if timestamp:
            try:
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "query": self.query,
            "embedding": self.embedding,
            "response": self.response,
            "timestamp": self.parse_timestamp(self.timestamp),
        }


# Generate embeddings using AWS Bedrock
@staticmethod
async def generate_embedding(text: str) -> list:
    """
    Generates an embedding vector for the given text using Bedrock's embedding model.

    Args:
        text (str): The text to generate an embedding for.

    Returns:
        list: The embedding vector as a list of floats.
              Returns an empty list if embedding generation fails.

    Raises:
        Exception: Logs any errors that occur during embedding generation but does not propagate them.
    """
    try:
        payload = {"inputText": text}
        response = bedrock_client.invoke_model(
            modelId=EMBEDDING_MODEL_ID, body=json.dumps(payload)
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return []


# Function to perform vector search in MongoDB
async def retrieve_from_cache(user_id: str, embedding: list, threshold: float = 0.95):
    """
    Retrieve a cached response from MongoDB based on vector similarity search.

    This function searches the cache for responses associated with the given user_id
    that have embeddings similar to the provided embedding vector. It returns a
    response only if the similarity score exceeds the specified threshold.

    Args:
        user_id (str): The identifier of the user making the request
        embedding (list): The vector representation of the query
        threshold (float, optional): The minimum similarity score required to return a cached result.
                                     Defaults to 0.95.

    Returns:
        str or None: The cached response if a match with sufficient similarity is found,
                     otherwise None.

    Raises:
        No exceptions are raised as they're caught internally and logged.
    """
    try:
        # Use MongoDB Atlas vector search with similarity threshold
        results = cache_collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": VECTOR_SEARCH_INDEX_NAME,
                        "path": "embedding",
                        "queryVector": embedding,
                        "numCandidates": 200,
                        "limit": 1,
                        "filter": {"user_id": user_id},
                    }
                },
                {
                    "$project": {
                        "score": {"$meta": "vectorSearchScore"},
                        "query": 1,
                        "response": 1,
                        "_id": 0,
                    }
                },
            ]
        )
        # Check if results contain a sufficiently similar response
        result = next(results, None)
        if result and result["score"] > threshold:  # Adjust threshold as needed
            return result["response"]
        return None
    except Exception as e:
        logger.error(f"Error retrieving from cache: {e}")
        return None


# Save query-response to cache
@app.post("/save_to_cache")
async def save_to_cache(entry: CacheEntry):
    """
    Save a query-response entry to the semantic cache.

    This function processes a CacheEntry by generating an embedding for the query
    and saving the entry to the cache collection in the database.

    Args:
        entry (CacheEntry): The cache entry containing the query, response, and user information

    Returns:
        dict: A status message indicating success or failure
              - On success: {"message": "Successfully saved to cache"}
              - On failure: {"message": "Failed to save to cache", "error": error_message}

    Raises:
        Exception: Exceptions are caught internally and returned as part of the error message
    """
    try:
        # Generate embedding for the query
        entry.embedding = await generate_embedding(entry.query)
        cache_collection.insert_one(entry.to_dict())
        logger.info(f"Saved query-response to cache for user {entry.user_id}")
        return {"message": "Successfully saved to cache"}
    except Exception as e:
        logger.error(f"Failed to save to cache: {e}")
        return {"message": "Failed to save to cache", "error": str(e)}


@app.post("/read_cache")
async def process_query(request: QueryRequest):
    """
    Process a query by generating an embedding and checking the cache.

    This endpoint handles POST requests to retrieve cached responses for user queries.
    It first generates an embedding for the query, then attempts to retrieve a matching
    response from the cache based on the user ID and query embedding.

    Args:
        request (QueryRequest): The request containing the user ID and query text.

    Returns:
        dict: A dictionary containing:
            - "response": The cached response if found, otherwise an empty string.
            - "error": Error message string if an exception occurred (only included on error).

    Raises:
        Exception: Various exceptions can be raised during embedding generation or cache retrieval,
                   but they are caught and returned as error responses.
    """
    try:
        embedding = await generate_embedding(request.query)
        cached_response = await retrieve_from_cache(request.user_id, embedding)
        if cached_response:
            logger.info(f"Cache hit for user {request.user_id}")
            return {"response": cached_response}
        logger.info(f"Cache miss for user {request.user_id}")
        return {"response": ""}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"response": "", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8183)
