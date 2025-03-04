import asyncio
import os
from PIL import Image
import io
import boto3
import filetype
from botocore.exceptions import ClientError
from tools import Tools
from logger import AsyncRemoteLogger
from dotenv import load_dotenv
from cache import MongoDBCache

load_dotenv()
# Create an instance of the logger
logger = AsyncRemoteLogger(
    service_url="http://logger:8181", app_name="MAAP-AWS-Meta-Main"
)


class AgentBase:
    """
    Base class for all agents.
    """

    bedrock_client = boto3.client(
        "bedrock-runtime", region_name=os.getenv("AWS_REGION")
    )  # Shared instance

    def __init__(self, name, model_id):
        self.name = name
        self.model_id = model_id
        # Initialize shared tools
        self.tools = Tools(
            mongodb_uri=os.getenv("MONGODB_URI"),
            mongodb_dbs={
                "travel_agency": ["trip_recommendation"],
                "maap_data_loader": ["document"],
            },
            aws_region=os.getenv("AWS_REGION"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            index_names={
                "trip_recommendation": ["travel_text_search_index", "vector_index"],
                "document": ["travel_text_search_index", "document_vector_index"],
            },
            text_keys={
                "trip_recommendation": "About Place",
                "document": "document_text",
            },
            embedding_keys={
                "trip_recommendation": "details_embedding",
                "document": "document_embedding",
            },
        )

        logger.info(f"{name} with {model_id} and Tools initialized successfully.")

    @MongoDBCache(ttl=3600, debug=True)  # Cache for 1 hour
    async def send_to_bedrock(self, prompt, image_path=None):
        """
        Send a prompt to the Bedrock model and return the response.
        """
        payload = await self._prepare_payload(prompt, image_path)

        try:
            response = await asyncio.to_thread(
                self.bedrock_client.converse,
                modelId=self.model_id,
                messages=payload["messages"],
            )
            model_response = response["output"]["message"]
            response_text = " ".join(i["text"] for i in model_response["content"])
            return response_text
        except ClientError as err:
            logger.error(
                "A client error occurred: %s", err.response["Error"]["Message"]
            )
            return None

    async def _prepare_payload(self, prompt, image_path=None):
        """
        Prepare the payload for AWS Converse.
        """
        if image_path:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                kind = filetype.guess(image_path)
                image_format = kind.extension if kind else "jpeg"

                # Convert jpg/jpeg to png and resize
                image = Image.open(io.BytesIO(image_bytes))
                if image_format in ["jpg", "jpeg"]:
                    image_format = "png"
                    image = image.convert("RGB")
                image.thumbnail((512, 512))
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                image_bytes = img_byte_arr.getvalue()

            message = {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": image_bytes},
                        }
                    },
                ],
            }
        else:
            message = {"role": "user", "content": [{"text": prompt}]}

        return {"messages": [message]}

    async def respond(
        self, userId, query, conversation_summary, tool_names, image_path=None
    ):
        """
        Abstract method to be implemented by each agent.
        """
        raise NotImplementedError("Each agent must implement its own 'respond' method.")


# Define various agent classes inheriting from AgentBase
class ReflectionAgent(AgentBase):
    async def respond(
        self, userId, query, conversation_summary, tool_names, image_path=None
    ):
        prompt = (
            f"You are a self-reflective agent. Reflect on the following input and provide feedback:\n"
            f"Input: {query}\n"
            f"Include strengths, areas for improvement, and suggestions for growth."
        )
        return await self.send_to_bedrock(prompt)


class SolutionAgent(AgentBase):
    async def respond(
        self, userId, query, conversation_summary, tool_names, image_path=None
    ):
        prompt = (
            f"You are a problem-solving agent. Solve the following problem step by step:\n"
            f"Problem: {query}\n"
            f"Provide a structured solution."
        )
        return await self.send_to_bedrock(prompt)


class InquiryAgent(AgentBase):
    async def respond(
        self, userId, query, conversation_summary, tool_names, image_path=None
    ):
        mongodb_results = ""
        websearch_results = ""
        if len(tool_names) > 0:
            for tool in tool_names:
                if tool == "MongoDB Hybrid Search":
                    mongodb_results = self.tools.search_mongodb(query, userId)
                elif tool == "Web Search":
                    if len(query) <= 400:
                        websearch_results = self.tools.search_web(query)

        logger.info(f"Web Search Results: {websearch_results}")
        prompt_mdb = (
            f"The response from MongoDB Hybrid search for the user query is: {mongodb_results}.\n"
            if len(mongodb_results) > 0
            else ""
        )
        prompt_ws = (
            f"The response from Web search for the user query is: {websearch_results}.\n"
            if len(websearch_results) > 0
            else ""
        )
        prompt_pc = (
            f"Previous related conversation summary: {conversation_summary}.\n"
            if conversation_summary
            else ""
        )

        prompt = (
            f"You are an answering agent. You have access to perform Hybrid search on MongoDB database and Web search. \n"
            f"{prompt_mdb}"
            f"{prompt_ws}"
            f"{prompt_pc}"
            f"Answer the following question:\n"
            f"Question: {query}\n"
            f"Provide a clear and concise response."
            f"Use the information retrieved from the MongoDB Hybrid search and Web search, if it is relevant."
        )
        return await self.send_to_bedrock(prompt)


class GuidanceAgent(AgentBase):
    async def respond(
        self, userId, query, conversation_summary, tool_names, image_path=None
    ):
        prompt = (
            f"You are a mentorship expert. Provide advice and guidance for the following:\n"
            f"Query: {query}\n"
            f"Offer actionable steps for personal or professional growth."
        )
        return await self.send_to_bedrock(prompt)


class VisualAgent(AgentBase):
    async def respond(
        self, userId, query, conversation_summary, tool_names, image_path=None
    ):
        prompt = (
            f"You are a highly capable AI assistant with perfect vision and exceptional attention to detail, "
            "specialized in analyzing images and extracting comprehensive information. "
            "Analyze and interpret the following visual data description:\n"
            f"Description: {query}\n"
            f"Provide insights or suggestions based on the visual data."
        )
        return await self.send_to_bedrock(prompt, image_path=image_path)


class CodingAgent(AgentBase):
    async def respond(
        self, userId, query, conversation_summary, tool_names, image_path=None
    ):
        prompt = (
            f"You are a coding expert. Review or generate code for the following task:\n"
            f"Task: {query}\n"
            f"Provide optimized and well-documented code."
        )
        return await self.send_to_bedrock(prompt)


class AnalyticsAgent(AgentBase):
    async def respond(
        self, userId, query, conversation_summary, tool_names, image_path=None
    ):
        prompt = (
            f"You are a data analytics expert. Analyze the following data and provide insights:\n"
            f"Data: {query}\n"
            f"Include key findings, trends, and recommendations."
        )
        return await self.send_to_bedrock(prompt)


class ReasoningAgent(AgentBase):
    async def respond(
        self, userId, query, conversation_summary, tool_names, image_path=None
    ):
        prompt = (
            f"You are a reasoning expert. Apply logical reasoning to the following scenario:\n"
            f"Scenario: {query}\n"
            f"Provide clear inferences and conclusions based on the scenario."
        )
        return await self.send_to_bedrock(prompt)


class MultiAgentSystem:
    """
    System to manage multiple agents and route queries to the appropriate agent.
    """

    def __init__(self):

        self.agents = {
            "ReflectionAgent": ReflectionAgent(
                "Reflection Agent",
                os.getenv("REFLECTION_AGENT", "us.meta.llama3-3-70b-instruct-v1:0"),
            ),
            "SolutionAgent": SolutionAgent(
                "Solution Agent",
                os.getenv("SOLUTION_AGENT", "us.meta.llama3-2-11b-instruct-v1:0"),
            ),
            "InquiryAgent": InquiryAgent(
                "Inquiry Agent",
                os.getenv("INQUIRY_AGENT", "us.meta.llama3-1-8b-instruct-v1:0"),
            ),
            "GuidanceAgent": GuidanceAgent(
                "Guidance Agent",
                os.getenv("GUIDANCE_AGENT", "us.meta.llama3-3-70b-instruct-v1:0"),
            ),
            "VisualAgent": VisualAgent(
                "Visual Agent",
                os.getenv("VISUAL_AGENT", "us.meta.llama3-2-90b-instruct-v1:0"),
            ),
            "CodingAgent": CodingAgent(
                "Coding Agent",
                os.getenv("CODING_AGENT", "us.meta.llama3-1-8b-instruct-v1:0"),
            ),
            "AnalyticsAgent": AnalyticsAgent(
                "Analytics Agent",
                os.getenv("ANALYTICS_AGENT", "us.meta.llama3-3-70b-instruct-v1:0"),
            ),
            "ReasoningAgent": ReasoningAgent(
                "Reasoning Agent",
                os.getenv("REASONING_AGENT", "us.meta.llama3-2-3b-instruct-v1:0"),
            ),
        }

    async def interact(
        self,
        userId,
        agent_type,
        query,
        conversation_summary=None,
        tool_names=None,
        image_path=None,
    ):
        """
        Interact with the specified agent type and return the response.
        """
        try:
            if agent_type in self.agents:
                response = await self.agents[agent_type].respond(
                    userId, query, conversation_summary, tool_names, image_path
                )
                return response
            else:
                return (
                    "Unknown agent type. Please choose from: Reflection, Solution, Inquiry, Guidance, "
                    "Visual, Coding, Analytics, or Reasoning."
                )
        except Exception as e:
            logger.error(str(e))
            return "An error occurred while processing the request." + str(e)
