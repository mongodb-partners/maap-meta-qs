from fastapi import FastAPI, Form
from fastapi import Request
from typing import List
import uvicorn

from agents import MultiAgentSystem
from orchestrator import QueryRouter
import datetime
import os
from logger import AsyncRemoteLogger
import json
from memory import AIMemory
from semantic_cache import SemanticCache
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from fastapi import HTTPException
import base64

load_dotenv()
# Create instances
logger = AsyncRemoteLogger(
    service_url="http://logger:8181", app_name="MAAP-AWS-Meta-Main"
)
ai_memory = AIMemory(base_url="http://ai-memory:8182")
semantic_cache = SemanticCache(base_url="http://semantic-cache:8183")


async def lifespan(app: FastAPI):
    """Manage resources for the app lifespan."""
    logger.print("Starting the app...")
    yield  # This marks the beginning of the lifespan context

    # Cleanup actions on shutdown
    await ai_memory.close()
    await logger.close()
    await semantic_cache.close()
    logger.print("Shutting down the app...")
    yield  # This marks the end of the lifespan context


# Initialize FastAPI app
app = FastAPI(
    title="MAAP - MongoDB AI Applications Program",
    version="1.0",
    description="MongoDB AI Applications Program",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


multi_agent_system = MultiAgentSystem()
query_router = QueryRouter(multi_agent_system)


@app.post("/agents")
async def process_request(
    request: Request,
    json_input_params: str = Form(
        description="Pass all input pamaraters as a Json string."
    ),
):
    """
    Process incoming requests and route them to the appropriate agent(s).
    """

    async def event_generator():
        inputs = {}
        strResponse = ""
        message = ""
        history = ""
        userId = ""
        conversation_id = ""
        tools = ""
        query = ""
        new_files = []
        try:
            inputs = json.loads(json_input_params)
            message = inputs["message"]
            history = inputs["history"]
            userId = inputs["userId"]
            conversation_id = inputs["conversation_id"]
            tools = inputs["tools"]
            query = inputs["query"]
            images = inputs["images"]

            for key, value in inputs.items():
                logger.print(key, " = ", value)

            new_files = SaveImages(images)
            logger.print(str(new_files))
        except Exception as error:
            logger.error(str(error))
            yield {
                "event": "message",
                "data": "Error processing request.\n" + str(error),
            }
        yield {
            "event": "message",
            "data": "Searching in MongoDB Atlas Semantic Cache.....\n",
        }
        try:
            cached_response = await semantic_cache.aget(userId, query)
            result = cached_response.get("response")
            if result:
                logger.print("Cache hit for user " + userId)
                yield {"event": "message", "data": result}
                yield {"event": "message", "data": "EOS"}
                return
            
        except Exception as error:
            logger.error(str(error))
            yield {
                "event": "message",
                "data": "Error while fetching from Semantic Cache.\n" + str(error),
            }
        yield {
                "event": "message",
                "data": "Searching for related conversation history.....\n",
            }
        conversation_summary = ""
        try:
            response =  await ai_memory.aget(
                {
                    "user_id": userId,
                    "text": query,
                }
            )
            print(response)
            conversation_summary = response.get("conversation_summary", "")
            logger.print(conversation_summary)
        except Exception as error:
            logger.error(str(error))
            yield {
                "event": "message",
                "data": "No conversation history found.\n" + str(error),
            }

        try:
            num_files = len(new_files)
            if num_files > 0:
                image_path = new_files[0]
                async for part in query_router.route_query(
                    userId, message, conversation_summary, tools, image_path
                ):
                    strResponse += part + "\n"
                    yield {"event": "message", "data": part + "\n"}
            else:
                async for part in query_router.route_query(
                    userId, query, conversation_summary, tools, None
                ):
                    strResponse += part + "\n"
                    yield {"event": "message", "data": part + "\n"}
        except Exception as error:
            logger.error(str(error))
            yield {
                "event": "message",
                "data": "Error processing request.\n" + str(error),
            }

        finally:
            try:
                payload = {
                    "user_id": userId,
                    "conversation_id": conversation_id,
                    "type": "human",
                    "text": query,
                }
                response = await ai_memory.apost(data=payload)

                payload = {
                    "user_id": userId,
                    "conversation_id": conversation_id,
                    "type": "ai",
                    "text": strResponse,
                }
                response = await ai_memory.apost(data=payload)

                payload = {
                    "user_id": userId,
                    "conversation_id": conversation_id,
                    "type": "ai",
                    "text": strResponse,
                }
                response = await semantic_cache.asave(userId, query, strResponse)

                strResponse = ""
                yield {"event": "message", "data": "EOS"}
            except Exception as error:
                logger.error(str(error))
                yield {
                    "event": "message",
                    "data": "Error processing request.\n" + str(error),
                }

    return EventSourceResponse(event_generator())


def SaveImages(images: List) -> List[str]:
    strDatetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    new_files = []

    for file in images:
        try:
            if file["filename"] is None or file["filename"].strip() == "":
                raise ValueError("Invalid file name.")

            file_name, file_ext = os.path.splitext(file["filename"])
            safe_file_name = file_name.replace(" ", "-")
            full_path = os.path.join(
                os.getcwd(), "files", f"{safe_file_name}_{strDatetime}{file_ext}"
            )

            # Ensure the "files" directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "wb") as f:
                f.write(base64.b64decode(file["data"]))

            new_files.append(full_path)

        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    return new_files


# FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
