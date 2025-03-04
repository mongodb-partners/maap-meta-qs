import hashlib
import json
import pymongo
from datetime import datetime, timezone
import os
import inspect
from logger import AsyncRemoteLogger
from dotenv import load_dotenv
import asyncio

load_dotenv()

logger = AsyncRemoteLogger(
    service_url="http://logger:8181", app_name="MAAP-AWS-Meta-Main"
)

# ========================
# MONGODB CONNECTION
# ========================
MONGODB_URI = os.getenv("MONGODB_URI")
client = pymongo.MongoClient(MONGODB_URI, maxPoolSize=100)
mongodb_database = client["cache"]
cache_collection = mongodb_database["data"]


# ========================
# INDEXING OPTIMIZATION
# ========================
def create_optimized_indexes():
    """
    Ensures that the MongoDB collection has optimized indexes for caching performance.
    """
    indexes = cache_collection.index_information()

    if "key_timestamp_idx" not in indexes:
        cache_collection.create_index(
            [("key", 1), ("timestamp", 1)],
            unique=True,
            name="key_timestamp_idx",
        )

    if "timestamp_ttl_idx" not in indexes:
        cache_collection.create_index(
            [("timestamp", 1)],
            expireAfterSeconds=3600,  # TTL of 1 hour
            name="timestamp_ttl_idx",
        )

    if "key_hashed_idx" not in indexes:
        cache_collection.create_index(
            [("key", "hashed")],
            name="key_hashed_idx",
        )

    logger.info("Optimized MongoDB indexes have been created.")


# Run this once at startup
create_optimized_indexes()


def serialize_args(args, kwargs):
    """Convert function arguments into a JSON-serializable format."""

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    return json.dumps(
        {"args": convert(args), "kwargs": convert(kwargs)}, sort_keys=True
    )


# ========================
# MONGODB CACHING CLASS (Supports both Sync & Async)
# ========================
class MongoDBCache:
    def __init__(self, ttl=3600, debug=False):
        self.ttl = ttl
        self.debug = debug

    def __call__(self, function):
        if inspect.iscoroutinefunction(function):
            return self._async_wrapper(function)
        else:
            return self._sync_wrapper(function)

    def _sync_wrapper(self, function):
        """Wrapper for synchronous functions."""

        def wrapper(*args, **kwargs):
            key_data = serialize_args(args, kwargs)
            key = hashlib.sha256(key_data.encode()).hexdigest()

            ele = cache_collection.find_one(
                {"key": key}, projection={"_id": 0, "response": 1}
            )
            if ele:
                if self.debug:
                    logger.info(
                        f"Cache HIT for {function.__name__} with args: {args}, kwargs: {kwargs}"
                    )
                return ele["response"]

            if self.debug:
                logger.info(
                    f"Cache MISS for {function.__name__} with args: {args}, kwargs: {kwargs}"
                )

            value = function(*args, **kwargs)

            cache_collection.update_one(
                {"key": key},
                {"$set": {"response": value, "timestamp": datetime.now(timezone.utc)}},
                upsert=True,
            )
            return value

        return wrapper

    def _async_wrapper(self, function):
        """Wrapper for asynchronous functions."""

        async def wrapper(*args, **kwargs):
            key_data = serialize_args(args, kwargs)
            key = hashlib.sha256(key_data.encode()).hexdigest()

            ele = cache_collection.find_one(
                {"key": key}, projection={"_id": 0, "response": 1}
            )
            if ele:
                if self.debug:
                    logger.info(
                        f"Cache HIT for {function.__name__} with args: {args}, kwargs: {kwargs}"
                    )
                return ele["response"]

            if self.debug:
                logger.info(
                    f"Cache MISS for {function.__name__} with args: {args}, kwargs: {kwargs}"
                )

            value = await function(*args, **kwargs)

            cache_collection.update_one(
                {"key": key},
                {"$set": {"response": value, "timestamp": datetime.now(timezone.utc)}},
                upsert=True,
            )
            return value

        return wrapper

    @staticmethod
    def invalidate_cache(*args, **kwargs):
        """Invalidate a specific cache entry."""
        key_data = serialize_args(args, kwargs)
        key = hashlib.sha256(key_data.encode()).hexdigest()
        result = cache_collection.delete_one({"key": key})
        if result.deleted_count:
            logger.info(f"Cache entry invalidated for args: {args}, kwargs: {kwargs}")
        else:
            logger.warning(f"No cache entry found for args: {args}, kwargs: {kwargs}")

    @staticmethod
    def clear_cache():
        """Clear the entire cache collection."""
        cache_collection.delete_many({})
        logger.info("All cache entries cleared.")


# # ========================
# # EXAMPLE USAGE
# # ========================
# @MongoDBCache(ttl=1800, debug=True)
# def expensive_sync_function(x: int, y: int):
#     """Expensive computation (synchronous)"""
#     time.sleep(2)  # Simulate long computation
#     return {"result": x + y}


# @MongoDBCache(ttl=1800, debug=True)
# async def expensive_async_function(x: int, y: int):
#     """Expensive computation (asynchronous)"""
#     await asyncio.sleep(2)  # Simulate long computation
#     return {"result": x + y}


# # ========================
# # TESTING
# # ========================
# if __name__ == "__main__":
#     # Sync function test
#     print("Sync Call 1:", expensive_sync_function(3, 4))  # First call (cache miss)
#     print("Sync Call 2:", expensive_sync_function(3, 4))  # Second call (cache hit)

#     # Async function test
#     async def test_async():
#         print("Async Call 1:", await expensive_async_function(3, 4))  # First call (cache miss)
#         print("Async Call 2:", await expensive_async_function(3, 4))  # Second call (cache hit)

#     asyncio.run(test_async())
