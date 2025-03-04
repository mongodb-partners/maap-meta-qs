from typing import Dict, List

import boto3
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_aws import BedrockEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
from pymongo import MongoClient
from tavily import TavilyClient
from cache import MongoDBCache

class Tools:

    def __init__(
        self,
        mongodb_uri: str | None = None,
        mongodb_dbs: Dict[str, List[str]] | None = None,
        aws_region: str | None = None,
        tavily_api_key: str | None = None,
        index_names: Dict[str, str] | None = None,
        text_keys: Dict[str, str] | None = None,
        embedding_keys: Dict[str, str] | None = None,
    ):
        self.mongodb_uri = mongodb_uri
        self.mongodb_dbs = (
            mongodb_dbs  # {"database_name": ["collection1", "collection2"]}
        )
        self.aws_region = aws_region
        self.tavily_api_key = tavily_api_key
        self.index_names = index_names or {}
        self.text_keys = text_keys or {}
        self.embedding_keys = embedding_keys or {}
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=aws_region)
        self.bedrock_embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1", client=self.bedrock_client
        )

        self.mongo_client = MongoClient(mongodb_uri) if mongodb_uri else None
        self.tavily_client = (
            TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
        )

    def search_mongodb(self, query: str, user_id: str) -> List[str]:
        """
        Performs a hybrid search across multiple MongoDB databases and collections.

        Args:
            query (str): Search query.
            user_id (str): ID of the user for filtering results.
            data_sources (List[str]): List of data sources (databases/collections) to search.

        Returns:
            List[str]: Retrieved document contents.
        """
        if not self.mongo_client or not self.mongodb_dbs:
            raise ValueError("MongoDB client or databases not configured.")

        retrievers = []
        for db_name, collections in self.mongodb_dbs.items():
            database = self.mongo_client[db_name]
            for collection_name in collections:
                collection = database[collection_name]

                vector_store = MongoDBAtlasVectorSearch(
                    text_key=self.text_keys.get(collection_name, "text"),
                    embedding_key=self.embedding_keys.get(collection_name, "embedding"),
                    index_name=self.index_names.get(collection_name, [None, None])[1],
                    embedding=self.bedrock_embeddings,
                    collection=collection,
                )

                retriever = MongoDBAtlasHybridSearchRetriever(
                    vectorstore=vector_store,
                    search_index_name=self.index_names.get(
                        collection_name, [None, None]
                    )[0],
                    pre_filter=(
                        {"userId": user_id}
                        if user_id and collection_name == "maap_data_loader"
                        else None
                    ),
                    top_k=10,
                )
                retrievers.append(retriever)

        if not retrievers:
            return []

        final_retriever = (
            retrievers[0]
            if len(retrievers) == 1
            else MergerRetriever(retrievers=retrievers)
        )
        documents = final_retriever.invoke(query)
        return [doc.page_content for doc in documents]

    @MongoDBCache(ttl=1800, debug=True)
    def search_web(self, query: str) -> List[str]:
        """
        Performs a web search using Tavily API.

        Args:
            query (str): The search query string.

        Returns:
            List[str]: Retrieved document contents from the web.
        """
        if not self.tavily_client:
            raise ValueError("Tavily API client is not configured.")

        documents = self.tavily_client.search(query)
        return [doc["content"] for doc in documents.get("results", [])]
