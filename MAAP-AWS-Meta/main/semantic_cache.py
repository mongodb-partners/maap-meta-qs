import asyncio
import httpx
from typing import Optional


class SemanticCache:
    """
    A singleton class for managing AI semantic caching with MongoDB.

    Attributes:
        base_url (str): The base URL of the semantic cache API.
        client (AsyncClient): The HTTP client used for requests.

    Methods:
        __new__(cls, *args, **kwargs):
            Ensures a single instance of the class (singleton pattern).

        _init_cache(self, base_url: str):
            Initializes the cache with the given API base URL.

        Asynchronous Methods:
            async save_to_cache(self, user_id: str, query: str, response: str):
                Saves a query-response pair in the cache.

            async get_from_cache(self, user_id: str, query: str):
                Retrieves a response from the cache.

            async aclose(self):
                Closes the HTTP client.

        Synchronous Methods:
            save(self, user_id: str, query: str, response: str):
                Synchronously saves data to cache.

            get(self, user_id: str, query: str):
                Synchronously retrieves data from cache.

            close(self):
                Synchronously closes the HTTP client.

    Author:
        Mohammad Daoud Farooqi
    """

    _instance: Optional["SemanticCache"] = None

    def __new__(cls, base_url: str):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_cache(base_url)
        return cls._instance

    def _init_cache(self, base_url: str):
        """Initialize the cache client with the specified base URL."""
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=10.0)

    async def asave(self, user_id: str, query: str, response: str):
        """Asynchronously save query-response pair to cache."""
        url = f"{self.base_url}/save_to_cache"
        payload = {
            "user_id": user_id,
            "query": query,
            "embedding": None,
            "response": response,
            "timestamp": None,
        }

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": str(e)}

    async def aget(self, user_id: str, query: str):
        """Asynchronously retrieve cached response."""
        url = f"{self.base_url}/read_cache"
        payload = {"user_id": user_id, "query": query}

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": str(e)}

    async def close(self):
        """Asynchronously close the HTTP client."""
        await self.client.aclose()

    def _run_in_thread_or_loop(self, coro):
        """Run coroutine synchronously in an event loop."""
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(coro)  # Running within an existing event loop
        except RuntimeError:
            asyncio.run(coro)

    # Synchronous Methods
    def save(self, user_id: str, query: str, response: str):
        """Synchronously save query-response pair to cache."""
        self._run_in_thread_or_loop(self.asave(user_id, query, response))

    def get(self, user_id: str, query: str):
        """Synchronously retrieve cached response."""
        return self._run_in_thread_or_loop(self.aget(user_id, query))

    def __del__(self):
        try:
            self._run_in_thread_or_loop(self.close())
        except RuntimeError:
            # If there's no running event loop, we can't close the client
            pass
