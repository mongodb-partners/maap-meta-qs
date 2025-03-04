import os
import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
import traceback
import boto3
from botocore.exceptions import ClientError
import pymongo.errors
import pymongo.mongo_client
import uvicorn
import pymongo
import asyncio
import json
from pydantic import BaseModel
from bson.objectid import ObjectId
from logger import AsyncRemoteLogger
from dotenv import load_dotenv
from pydantic import Field
from bson import json_util

# Create an instance of the logger
logger = AsyncRemoteLogger(
    service_url="http://logger:8181", app_name="MAAP-AWS-Meta-AI-Memory"
)
# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app with title and version
app = FastAPI(title="AI Knowledge and Memory Service", version="1.0")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Initialize a shared boto3 client for Bedrock service
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")


# MongoDB Configuration variables
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = "ai_memory"
MONGODB_COLLECTION_CONVERSATIONS = "conversations"
VECTOR_SEARCH_INDEX_NAME = "conversations_vector_index"
FULLTEXT_SEARCH_INDEX_NAME = "conversations_text_index"
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "us.meta.llama3-3-70b-instruct-v1:0")

# Create a synchronous MongoDB client
client = pymongo.MongoClient(MONGODB_URI)
# Access the specified database and collections
db = client[MONGODB_DB_NAME]
memory_collection = db[MONGODB_COLLECTION_CONVERSATIONS]

# Ensure collections exist; create if not already created
if memory_collection.name not in memory_collection.database.list_collection_names():
    memory_collection.database.create_collection(memory_collection.name)

    # Define index definitions for vector and full-text search
    index_definitions = [
        {
            "name": VECTOR_SEARCH_INDEX_NAME,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embeddings",
                        "numDimensions": 1536,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "user_id"},
                ]
            },
        },
        {
            "name": FULLTEXT_SEARCH_INDEX_NAME,
            "type": "search",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {"text": {"type": "string"}},
                }
            },
        },
    ]

    # Create the defined search indexes on the memory_collection
    for index_definition in index_definitions:
        try:
            memory_collection.create_search_index(model=index_definition)
        except pymongo.errors.PyMongoError:
            # If an error occurs during index creation, simply pass
            pass

    try:
        # Create a TTL index on the timestamp field
        memory_collection.create_index(
            [("timestamp", 1)],
            expireAfterSeconds=86400,  # TTL of 1 day
            name="timestamp_ttl_idx",
        )
    except pymongo.errors.PyMongoError:
        # If an error occurs during index creation, simply pass
        pass


@staticmethod
def generate_embedding(text: str) -> list:
    if not text.strip():
        logger.error("Text is empty. Cannot generate embeddings.")
        raise ValueError("Input text cannot be empty.")

    try:
        max_tokens = 8000  # Embedding model input token limit
        tokens = text.split()  # Simple tokenization by spaces
        text = " ".join(tokens[:max_tokens])  # Keep only allowed tokens

        payload = {"inputText": text}
        response = bedrock_client.invoke_model(
            modelId=EMBEDDING_MODEL_ID, body=json.dumps(payload)
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise  # Re-raise the exception to propagate the error


# Pydantic model for input validation
class MessageInput(BaseModel):
    user_id: str = Field(..., min_length=1, description="User ID cannot be empty")
    conversation_id: str = Field(
        ..., min_length=1, description="Conversation ID cannot be empty"
    )
    type: str = Field(
        ..., pattern="^(human|ai)$", description="Must be 'human' or 'ai'"
    )
    text: str = Field(..., min_length=1, description="Message text cannot be empty.")
    timestamp: str | None = Field(None, description="UTC timestamp (optional)")


# Message class
class Message:
    def __init__(self, message_data: MessageInput):
        self.user_id = message_data.user_id.strip()
        self.conversation_id = message_data.conversation_id.strip()
        self.type = message_data.type
        self.text = message_data.text.strip()
        self.timestamp = self.parse_timestamp(message_data.timestamp)
        self.embeddings = generate_embedding(self.text)

    def parse_timestamp(self, timestamp: Optional[str]) -> datetime.datetime:
        if timestamp:
            try:
                return datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid timestamp format")
        return datetime.datetime.now(datetime.timezone.utc)

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "type": self.type,
            "text": self.text,
            "timestamp": self.timestamp,
            "embeddings": self.embeddings,
        }


@app.post("/conversation/")
async def add_message(message: MessageInput):
    try:
        new_message = Message(message)
        memory_collection.insert_one(new_message.to_dict())
        log_data = {k: v for k, v in new_message.to_dict().items() if k != "embeddings"}
        logger.info(f"Message added: {log_data}")
        return {"message": "Message added successfully."}

    except Exception as error:
        logger.error(
            f"DB Error: {str(traceback.TracebackException.from_exception(error).stack.format())}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error\n"
            + str(traceback.TracebackException.from_exception(error).stack.format()),
        )


# Function to perform a hybrid search using both full-text and vector search
def hybrid_search(query, vector_query, user_id, weight=0.5, top_n=10):
    """
    Perform a hybrid search operation on MongoDB by combining full-text and vector (semantic) search results using an aggregation pipeline.

    This function executes a two-pronged search:
    1. A full-text search on the collection using a text index.
    2. A vector search on the collection using a precomputed embedding (vector) index.

    Both search results are normalized, weighted, and merged to compute a final hybrid score, which is used to sort and select the top documents.

    Args:
        query (str): The full-text search query string to be executed against the text index.
        vector_query (list): The vector representation of the query used for semantic search.
        user_id (str): The identifier of the user, used to filter search results to only include documents belonging to the specified user.
        weight (float, optional): A weight factor between 0 and 1 that determines the influence of the vector (semantic) score relative to the text search score.
                                  A value of 0 emphasizes full-text search exclusively, while 1 emphasizes vector search exclusively.
                                  Defaults to 0.5.
        top_n (int, optional): The maximum number of search results to return. Defaults to 10.

    Returns:
        list: A list of documents where each document is a dictionary that includes:
              - "_id": The unique identifier of the document.
              - "fts_score": The normalized score from the full-text search.
              - "vs_score": The normalized score from the vector search.
              - "score": The computed hybrid score based on the weighted contribution of both search methods.
              - "text": The text content of the document.
              - "type": The type/category of the document.
              - "timestamp": The timestamp of the document.
              - "conversation_id": The identifier grouping related conversation or session entries.

    Raises:
        AssertionError: If the provided weight is not within the range [0, 1].

    Notes:
        The aggregation pipeline consists of these key stages:
          - Executing a text search to fetch matching documents and compute a text search score ("fts_score").
          - Filtering documents by user_id.
          - Normalizing the full-text search score by dividing by the maximum score.
          - Performing an embedded vector search on another collection using the provided vector query.
          - Normalizing the vector search score ("vs_score") similarly.
          - Merging results from both searches using a union operation.
          - Grouping the merged results by document ID and selecting the maximum of each score.
          - Calculating the final hybrid score, which is a weighted sum of normalized vector and text scores.
          - Sorting documents by the hybrid score in descending order and limiting the result set to 'top_n' entries.

    Example:
        >>> results = hybrid_search("example query", [0.1, 0.2, ...], "user123", weight=0.7, top_n=5)
        >>> for document in results:
        ...     print(document)
    """
    # Validate weight input value
    assert 0 <= weight <= 1, "Weight must be between 0 and 1"
    pipeline = [
        {
            "$search": {
                "index": FULLTEXT_SEARCH_INDEX_NAME,
                "text": {"query": query, "path": "text"},
            }
        },
        {"$match": {"user_id": user_id}},
        {"$addFields": {"fts_score": {"$meta": "searchScore"}}},
        {"$setWindowFields": {"output": {"maxScore": {"$max": "$fts_score"}}}},
        {
            "$addFields": {
                "normalized_fts_score": {"$divide": ["$fts_score", "$maxScore"]}
            }
        },
        {
            "$project": {
                "text": 1,
                "type": 1,
                "timestamp": 1,
                "conversation_id": 1,
                "normalized_fts_score": 1,
            }
        },
        {
            "$unionWith": {
                "coll": memory_collection.name,
                "pipeline": [
                    {
                        "$vectorSearch": {
                            "index": VECTOR_SEARCH_INDEX_NAME,
                            "queryVector": vector_query,  # Use precomputed vector query
                            "path": "embeddings",
                            "numCandidates": 200,
                            "limit": top_n,  # Limit vector search results
                            "filter": {"user_id": user_id},
                        }
                    },
                    {"$addFields": {"vs_score": {"$meta": "vectorSearchScore"}}},
                    {
                        "$setWindowFields": {
                            "output": {"maxScore": {"$max": "$vs_score"}}
                        }
                    },
                    {
                        "$addFields": {
                            "normalized_vs_score": {
                                "$divide": ["$vs_score", "$maxScore"]
                            }
                        }
                    },
                    {
                        "$project": {
                            "text": 1,
                            "type": 1,
                            "timestamp": 1,
                            "conversation_id": 1,
                            "normalized_vs_score": 1,
                        }
                    },
                ],
            }
        },
        {
            "$group": {
                "_id": "$_id",  # Group by document ID
                "fts_score": {"$max": "$normalized_fts_score"},
                "vs_score": {"$max": "$normalized_vs_score"},
                "text_field": {"$first": "$text"},
                "type_field": {"$first": "$type"},
                "timestamp_field": {"$first": "$timestamp"},
                "conversation_id_field": {"$first": "$conversation_id"},
            }
        },
        {
            "$addFields": {
                "hybrid_score": {
                    "$add": [
                        {"$multiply": [weight, {"$ifNull": ["$vs_score", 0]}]},
                        {"$multiply": [1 - weight, {"$ifNull": ["$fts_score", 0]}]},
                    ]
                }
            }
        },
        {"$sort": {"hybrid_score": -1}},  # Sort by combined hybrid score descending
        {"$limit": top_n},  # Limit final output
        {
            "$project": {
                "_id": 1,  # MongoDB document ID
                "fts_score": 1,
                "vs_score": 1,
                "score": "$hybrid_score",
                "text": "$text_field",
                "type": "$type_field",
                "timestamp": "$timestamp_field",
                "conversation_id": "$conversation_id_field",
            }
        },
    ]

    # Execute the aggregation pipeline and return the results
    results = list(memory_collection.aggregate(pipeline))
    return results


def serialize_document(doc):
    """Helper function to serialize MongoDB documents."""
    doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
    return doc


# Endpoint to search memory items using hybrid search
@app.get("/search/")
async def search_memory(user_id: str, query: str):
    """
    Searches memory items by user_id and a textual query using hybrid search across full-text and vector search.
    """
    try:
        # Generate embedding for the query text
        vector_query = generate_embedding(query)
        # Perform hybrid search over the stored messages
        documents = hybrid_search(query, vector_query, user_id, weight=0.5, top_n=5)
        print(documents)
        # Filter results by minimum hybrid score threshold
        relevant_results = [doc for doc in documents if doc["score"] >= 0.75]
        if not relevant_results:
            return {"documents": "No documents found"}
        else:
            return {"documents": [serialize_document(doc) for doc in relevant_results]}
    except Exception as error:
        logger.error(
            f"{str(traceback.TracebackException.from_exception(error).stack.format())}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error\n"
            + str(traceback.TracebackException.from_exception(error).stack.format()),
        )


# Asynchronously send prompt to Bedrock model and receive its response
async def send_to_bedrock(prompt):
    """
    Send a prompt to the Bedrock model asynchronously.
    """
    payload = [{"role": "user", "content": [{"text": prompt}]}]
    model_id = LLM_MODEL_ID

    try:
        # Use asyncio.to_thread to call the blocking boto3 client method
        response = await asyncio.to_thread(
            bedrock_client.converse,
            modelId=model_id,
            messages=payload,
        )
        model_response = response["output"]["message"]
        # Concatenate text parts from the model response
        response_text = " ".join(i["text"] for i in model_response["content"])
        return response_text
    except ClientError as err:
        logger.error("A client error occurred: %s", err.response["Error"]["Message"])
        return None


# Endpoint to generate a summary for a conversation using Bedrock model
@app.get("/generate_conversation_summary/")
async def generate_conversation_summary(documents: str):
    """
    Generates a detailed and structured summary for a conversation provided in JSON format.

    This function builds a comprehensive prompt that includes detailed instructions and the JSON-encoded conversation data.
    It then sends this prompt to an AI service (Bedrock) to produce a summary that captures all critical discussion points, key topics, decisions made, and any unresolved questions.
    This helps ensure that the conversation's context and essential details are accurately preserved.

    Parameters:
        documents (str): A JSON-formatted string representing the conversation history.

    Returns:
        dict: A dictionary containing the key "summary" with the generated summary text.

    Raises:
        HTTPException: If an error occurs during the generation process.
    """
    try:
        # Construct a prompt with detailed instructions and conversation history
        prompt = (
            f"You are an advanced AI assistant skilled in analyzing and summarizing conversation histories while preserving all essential details.\n"
            f"Given the following conversation data in JSON format, generate a detailed and structured summary that captures all key points, topics discussed, decisions made, and relevant insights.\n\n"
            f"Ensure your summary follows these guidelines:\n"
            f"- **Maintain Clarity & Accuracy:** Include all significant details, technical discussions, and conclusions.\n"
            f"- **Preserve Context & Meaning:** Avoid omitting important points that could alter the conversation's intent.\n"
            f"- **Organized Structure:** Present the summary in a logical flow or chronological order.\n"
            f"- **Key Highlights:** Explicitly state major questions asked, AI responses, decisions made, and follow-up discussions.\n"
            f"- **Avoid Redundancy:** Summarize effectively without unnecessary repetition.\n\n"
            f"### Output Format:\n"
            f"- **Topic:** Briefly describe the conversation's purpose.\n"
            f"- **Key Discussion Points:** Outline the main topics covered.\n"
            f"- **Decisions & Takeaways:** Highlight key conclusions or next steps.\n"
            f"- **Unresolved Questions (if any):** Mention pending queries or areas needing further clarification.\n\n"
            f"Provide a **clear, structured, and comprehensive** summary ensuring no critical detail is overlooked.\n\n"
            f"Input JSON: {json.dumps(documents, default=json_util.default)}"
        )

        # Send prompt to Bedrock and wait for summary response
        summary = await send_to_bedrock(prompt)
        return {"summary": summary}
    except Exception as error:
        logger.error(
            f"{str(traceback.TracebackException.from_exception(error).stack.format())}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error\n"
            + str(traceback.TracebackException.from_exception(error).stack.format()),
        )


# Endpoint to retrieve a conversation with additional context around matching messages
@app.get("/get_conversation_context/")
async def get_conversation_context(_id: str):
    """
    Fetches conversation records with context surrounding a specific message, identified by its unique _id.

    This endpoint retrieves a conversation record by its object ID from a MongoDB collection. It then extracts the
    user ID, conversation ID, timestamp, and type from the fetched record. Depending on the message type:

    - If the message type is "ai", it retrieves up to four preceding messages (messages with a timestamp less than or equal to the target,
        sorted in descending order) and two subsequent messages (messages with a timestamp greater than the target, sorted in ascending order).
    - For any other message type, it retrieves up to three preceding messages (sorted in descending order) and three subsequent messages (sorted in ascending order).

    The function then combines these messages, sorts them in ascending order based on the timestamp, and returns the complete conversation context.

    Parameters:
            _id (str): The unique identifier corresponding to the conversation record in MongoDB.

    Returns:
            dict: A dictionary with a key "documents" containing the list of conversation records that include the context messages.
                  If no record is found for the provided _id, it returns {"documents": "No documents found"}.

    Raises:
            HTTPException: If any error occurs during the process, an HTTP 500 error is raised with the error details.
    """
    try:
        # Fetch the conversation record for the given object ID
        conversation_record = memory_collection.find_one(
            {"_id": ObjectId(_id)},
            projection={
                "_id": 0,
                "embeddings": 0,
            },
        )
        if not conversation_record:
            return {"documents": "No documents found"}

        # Extract the user ID and conversation ID from the conversation record
        user_id = conversation_record["user_id"]
        conversation_id = conversation_record["conversation_id"]
        timestamp = conversation_record["timestamp"]
        type = conversation_record["type"]

        if type == "ai":
            # get 3 previous messages and 2 next messages
            context = list(
                memory_collection.find(
                    {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "timestamp": {"$lte": timestamp},
                    },
                    projection={
                        "_id": 0,
                        "embeddings": 0,
                    },
                )
                .sort("timestamp", pymongo.DESCENDING)
                .limit(4)
            )

            context_after = list(
                memory_collection.find(
                    {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "timestamp": {"$gt": timestamp},
                    },
                    projection={
                        "_id": 0,
                        "embeddings": 0,
                    },
                )
                .sort("timestamp", pymongo.ASCENDING)
                .limit(2)
            )

        else:
            # get 2 previous messages and 3 next messages
            context = list(
                memory_collection.find(
                    {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "timestamp": {"$lte": timestamp},
                    },
                    projection={
                        "_id": 0,
                        "embeddings": 0,
                    },
                )
                .sort("timestamp", pymongo.DESCENDING)
                .limit(3)
            )

            context_after = list(
                memory_collection.find(
                    {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "timestamp": {"$gt": timestamp},
                    },
                    projection={
                        "_id": 0,
                        "embeddings": 0,
                    },
                )
                .sort("timestamp", pymongo.ASCENDING)
                .limit(3)
            )

        # Combine the context and context_after, then return sorted on timestamp.
        conversation_with_context = sorted(
            context + context_after,
            key=lambda x: x["timestamp"],
        )
        print(conversation_with_context)
        return {"documents": conversation_with_context}
    except Exception as error:
        logger.error(
            f"{str(traceback.TracebackException.from_exception(error).stack.format())}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error\n"
            + str(traceback.TracebackException.from_exception(error).stack.format()),
        )


@app.get("/retrieve_memory/")
async def retrieve_memory(user_id: str, text: str):
    """
    Retrieve memory items, context, and summary for a specified user based on input text.

    This endpoint performs the following steps:
      1. Searches memory items using the search_memory function.
      2. Extracts the conversation ID from the first matching memory item.
      3. Retrieves the surrounding conversation context via get_conversation_context.
      4. Generates a detailed summary of the conversation using the generate_conversation_summary function.

    Returns:
      A JSON object containing:
        - memory_items: The search results from the memory database.
        - context: The contextual conversation messages.
        - summary: A comprehensive summary of the conversation.
    """
    try:
        # Retrieve memory items based on text search
        memory_items = await search_memory(user_id, text)
        if memory_items["documents"] == "No documents found":
            return {"documents": "No documents found"}

        # Extract conversation ID from the first memory item
        object_id = memory_items["documents"][0]["_id"]

        # Retrieve conversation context around the matching memory item
        context = await get_conversation_context(object_id)

        # Generate a detailed summary for the conversation
        summary = await generate_conversation_summary(context["documents"])
        result = {
            "memory_items": memory_items["documents"],
            "context": context["documents"],
            "conversation_summary": summary["summary"],
        }
        logger.print(result)

        return result
    except Exception as error:
        logger.error(
            f"{str(traceback.TracebackException.from_exception(error).stack.format())}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error\n"
            + str(traceback.TracebackException.from_exception(error).stack.format()),
        )


# Main section to run the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8182)
