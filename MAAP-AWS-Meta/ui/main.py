import asyncio
import json
import mimetypes
import os
import re
import time
import uuid
import httpx
import asyncio
import gradio as gr
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from gradio import Markdown as m
import traceback
from images import PLUS
from logger import AsyncRemoteLogger
import base64


def generate_unique_conversation_id(prefix: str = "conversation_") -> str:
    """
    Generates a unique conversation ID using a UUID without dashes and a timestamp.

    :param prefix: (str) Optional prefix for the conversation ID.
    :return: (str) A unique conversation ID.
    """
    unique_id = uuid.uuid4().hex  # Generate UUID and remove dashes
    timestamp = int(time.time())  # Get current timestamp
    conversation_id = f"{prefix}{unique_id}_{timestamp}"

    return conversation_id


# Create an instance of the logger
logger = AsyncRemoteLogger(
    service_url="http://logger:8181", app_name="MAAP-AWS-Meta-UI"
)

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

app = FastAPI(
    title="MAAP - MongoDB AI Applications Program",
    version="1.0",
    description="MongoDB AI Applications Program",
)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def process_request(message, history, userId, conversation_id, tools):
    images_data = []
    try:
        await logger.aprint(userId, conversation_id, tools)
        url = "http://main:8000/agents"
        await logger.aprint(message, history)

        if message and len(message) > 0:
            query = message["text"].strip()
            urls = extract_urls(query)
            await logger.aprint(urls)
            num_files = len(message["files"])
            strTempResponse = ""
            if num_files > 0 or len(urls) > 0:
                for file in message["files"]:
                    file_name, file_ext = os.path.splitext(file)
                    file_name = os.path.basename(file)
                    image_file_types = [".jpg", ".jpeg", ".png"]
                    if file_ext in image_file_types:
                        images_data.append(
                            {
                                "filename": file_name,
                                "data": encode_image_to_base64(file),
                            }
                        )

                strTempResponse = ""
                for i in re.split(
                    r"(\s)",
                    "Initiating upload and content vectorization. \nPlease wait....",
                ):
                    strTempResponse += i
                    await asyncio.sleep(0.025)
                    yield strTempResponse

                uploadResult = await ingest_data(userId, urls, message["files"])

                if uploadResult:
                    for i in re.split(
                        r"(\s)",
                        "\nFile(s)/URL(s) uploaded  and ingested successfully. \nGiving time for Indexes to Update....",
                    ):
                        strTempResponse += i
                        await asyncio.sleep(0.025)
                        yield strTempResponse
                    await asyncio.sleep(5)
                else:
                    for i in re.split(
                        r"(\s)", "\nFile(s)/URL(s) upload exited with error...."
                    ):
                        strTempResponse += i
                        await asyncio.sleep(0.025)
                        yield strTempResponse

            if len(query) > 0:
                inputs = {
                    "message": message,
                    "history": history,
                    "userId": userId,
                    "conversation_id": conversation_id,
                    "tools": tools,
                    "query": query,
                    "images": images_data,
                }

                payload = {"json_input_params": json.dumps(inputs)}

                strResponse = ""
                async with httpx.AsyncClient(timeout=100000) as client:
                    async with client.stream("POST", url, data=payload) as response:
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if line.startswith("data:"):
                                    data = line[6:]  # Remove 'data:' prefix and print
                                    strResponse += data + "\n"

                                    # Stop the client if "End of the stream" is received
                                    if data == "EOS":
                                        print("Stream ended by the server.")
                                        break
                                    # Yield partial response for real-time updates
                                    yield strResponse

                        else:
                            logger.print(f"Failed to connect: {response.status_code}")

            else:
                yield "Hi, how may I help you?"
        else:
            yield "Hi, how may I help you?"
    except Exception as error:
        exc = traceback.TracebackException.from_exception(error)
        emsg = ''.join(exc.format())  # Includes stack + error message
        await logger.aerror(emsg)
        yield "There was an error.\n" + emsg



def extract_urls(string):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]


async def ingest_data(userId, urls, new_files):
    url = "http://loader:8001/upload"

    inputs = {
        "userId": userId,
        "MongoDB_URI": MONGODB_URI,
        "MongoDB_text_key": "document_text",
        "MongoDB_embedding_key": "document_embedding",
        "MongoDB_index_name": "document_vector_index",
        "MongoDB_database_name": "maap_data_loader",
        "MongoDB_collection_name": "document",
        "WebPagesToIngest": urls,
    }

    payload = {"json_input_params": json.dumps(inputs)}
    files = []

    for file in new_files:

        file_name, file_ext = os.path.splitext(file)
        file_name = os.path.basename(file)
        mime_type, encoding = mimetypes.guess_type(file)
        file_types = [
            ".bmp",
            ".csv",
            ".doc",
            ".docx",
            ".eml",
            ".epub",
            ".heic",
            ".html",
            ".jpeg",
            ".png",
            ".md",
            ".msg",
            ".odt",
            ".org",
            ".p7s",
            ".pdf",
            ".png",
            ".ppt",
            ".pptx",
            ".rst",
            ".rtf",
            ".tiff",
            ".txt",
            ".tsv",
            ".xls",
            ".xlsx",
            ".xml",
            ".vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".vnd.openxmlformats-officedocument.presentationml.presentation",
        ]
        if file_ext in file_types:
            files.append(("files", (file_name, open(file, "rb"), mime_type)))
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    logger.print(response.text)
    if "Successfully uploaded" in response.text:
        return True
    else:
        return False


def print_like_dislike(x: gr.LikeData):
    logger.print(x.index, x.value, x.liked)
    return


head = """
<link rel="shortcut icon" href="https://ok5static.oktacdn.com/bc/image/fileStoreRecord?id=fs0jq9i9e0E4EFpjn297" type="image/x-icon">
"""
mdblogo_svg = "https://ok5static.oktacdn.com/fs/bco/1/fs0jq9i9coLeryBSy297"
metalogo_svg = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Meta_Platforms_Inc._logo.svg/800px-Meta_Platforms_Inc._logo.svg.png"

custom_css = """
           
            .message-row img {
                margin: 0px !important;
            }

            .avatar-container img {
            padding: 0px !important;
            }

            footer {visibility: hidden}; 
        """

with gr.Blocks(
    head=head,
    fill_height=True,
    fill_width=True,
    css=custom_css,
    title="MongoDB AI Applications Program (MAAP)",
    theme=gr.themes.Soft(primary_hue=gr.themes.colors.green),
) as demo:
    with gr.Row():
        m(
            f"""
<center>
    <div style="display: flex; justify-content: center; align-items: center;">
        <a href="https://www.mongodb.com/">
            <img src="{mdblogo_svg}" width="250px" style="margin-right: 20px"/>
        </a>
    <img src="{PLUS}" width="30px" style="margin-right: 20px;margin-left: 5px;margin-top: 10px;"/>
        <a href="https://ai.meta.com/">
            <img src="{metalogo_svg}" width="215px" style="margin-top: 10px;"/>
        </a>
    </div>
    <h1>MongoDB AI Applications Program (<a href="https://www.mongodb.com/services/consulting/ai-applications-program">MAAP</a>)</h1>
    <h3>An integrated end-to-end technology stack in the form of MAAP Framework.</h3>
</center>
"""
        )
    with gr.Accordion(
        label="--- Inputs ---", open=True, render=True
    ) as AdditionalInputs:
        m(
            """<p>
    Enter a User ID to store and retrieve user-specific file data from MongoDB. 
    Upload files via the Attach (clip) button or submit URLs to extract and store information in the MongoDB Atlas Vector Database, enabling contextually relevant searches.
    Receive precise query responses from the Multi-Agent AI System, powered by Meta's Llama LLMs, leveraging data retrieved from MongoDB.
        </p>
        """
        )

        txtUserId = gr.Textbox(
            value="your.email@yourdomain.com", label="User Id", key="UserId"
        )

        txtConversationId = gr.Textbox(
            value="",
            label="Conversation Id (read-only)",
            key="ConversationId",
            info="Unique conversation ID for the current session. Changes on page refresh.",
            interactive=False,
        )
        demo.load(
            generate_unique_conversation_id, inputs=[], outputs=[txtConversationId]
        )

        chbkgTools = gr.CheckboxGroup(
            choices=["MongoDB Hybrid Search", "Web Search"],
            value=["MongoDB Hybrid Search", "Web Search"],
            label="Tools",
            info="Which tools should the agent use to extract relevant information?",
            key="tools",
        )
    txtChatInput = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Type your query and/or upload file(s) and interact with it...",
        label="User Query",
        show_label=True,
        render=False,
    )

    examples = [
        [
            "Recommend places to visit in India.",
            "your.email@yourdomain.com",
            ["MongoDB Hybrid Search", "Web Search"],
        ],
        [
            "Explain https://www.mongodb.com/services/consulting/ai-applications-program",
            "your.email@yourdomain.com",
            ["MongoDB Hybrid Search", "Web Search"],
        ],
        [
            "How can I improve my leadership skills?",
            "your.email@yourdomain.com",
            ["MongoDB Hybrid Search", "Web Search"],
        ],
        [
            "What are the best practices for creating a scalable AI architecture?",
            "your.email@yourdomain.com",
            ["MongoDB Hybrid Search", "Web Search"],
        ],
        [
            "Explain how I can manage my team better while solving technical challenges.",
            "your.email@yourdomain.com",
            ["MongoDB Hybrid Search", "Web Search"],
        ],
    ]
    bot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=True,
        type="messages",
        autoscroll=True,
        avatar_images=[
            "https://ca.slack-edge.com/E01C4Q4H3CL-U04D0GXU2B1-g1a101208f57-192",
            "https://avatars.slack-edge.com/2021-11-01/2659084361479_b7c132367d18b6b7ffa0_512.png",
        ],
        show_copy_button=True,
        render=False,
        min_height="550px",
        label="Type your query and/or upload file(s) and interact with it...",
    )
    bot.like(print_like_dislike, None, None, like_user_message=False)

    CI = gr.ChatInterface(
        fn=process_request,
        chatbot=bot,
        type="messages",
        title="",
        description="Interact with a multi-agent system to get responses tailored to your query.",
        multimodal=True,
        additional_inputs=[txtUserId, txtConversationId, chbkgTools],
        additional_inputs_accordion=AdditionalInputs,
        textbox=txtChatInput,
        fill_height=True,
        show_progress=False,
        concurrency_limit=None,
    )

    gr.Examples(
        examples,
        inputs=[txtChatInput, txtUserId, chbkgTools],
        examples_per_page=2,
    )

    with gr.Row():

        m(
            """
            <center><a href="https://www.mongodb.com/">MongoDB</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <a href="https://ai.meta.com/">Meta</a>
            </center>
       """
        )


if __name__ == "__main__":
    app = gr.mount_gradio_app(
        app, demo, path="/", server_name="0.0.0.0", server_port=7860
    )
    uvicorn.run(app, host="0.0.0.0", port=7860)
