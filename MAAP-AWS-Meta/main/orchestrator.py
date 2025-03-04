from agents import ReflectionAgent
import os
from cache import MongoDBCache

class QueryRouter:
    """
    Router to categorize and route queries to the appropriate agent(s).
    """

    def __init__(self, agent_system):
        self.agent_system = agent_system
        self.categorizer = ReflectionAgent(
            "Query Categorizer",
            os.getenv("REFLECTION_AGENT", "us.meta.llama3-3-70b-instruct-v1:0"),
        )
    @MongoDBCache(ttl=3600, debug=True)  # Cache for 1 hour
    async def categorize_query(self, query):
        """
        Categorize the query to determine the appropriate agent(s) to handle it.
        """
        prompt = (
            f"You are a query categorizer. "
            "Available Agents: [Reflection, Solution, Inquiry, Guidance, Visual, Coding, Analytics, Reasoning]. \n"
            "Here are short descriptions for each agent:"
            "1. **ReflectionAgent**: A self-reflective agent that analyzes input, providing feedback on strengths, areas for improvement, and suggestions for growth."
            "2. **SolutionAgent**: A problem-solving agent that breaks down problems step by step and provides structured solutions."
            "3. **InquiryAgent**: An answering agent that uses MongoDB Hybrid search to gather relevant information and provide clear, concise answers to user queries."
            "4. **GuidanceAgent**: A mentorship expert that offers actionable advice and steps for personal or professional development based on user input."
            "5. **VisualAgent**: A highly capable agent specialized in analyzing images, interpreting visual data, and offering insights or suggestions. Use VisualAgent only if there is an image attached with the query."
            "6. **CodingAgent**: A coding expert that reviews or generates optimized and well-documented code for specific tasks."
            "7. **AnalyticsAgent**: A data analytics expert that analyzes data and provides insights, trends, and recommendations based on key findings."
            "8. **ReasoningAgent**: A reasoning expert that applies logical reasoning to scenarios, providing clear inferences and conclusions based on the given context.\n\n"
            "Analyze the following query and determine:\n"
            f"1. Which type of agent is most suited to handle it. Choose only from the given list of available agents. "
            f"2. If the query requires collaboration between multiple agents.\n"
            f"3. Provide a reason and recommend initial collaborators.\n"
            f"Query: {query}\n"
            f"Provide your response in the format:\n"
            f"Category: <AgentType>\n"
            f"Collaboration: <Yes/No>\n"
            f"Reason: <Short explanation>\n"
            f"InitialCollaborators: [<AgentType1>, <AgentType2>, ...]. Include all required participating agents if Collaboration is 'Yes'."
            "If an image is given, use VisualAgent first for image analysis."
        )
        categorization = await self.categorizer.send_to_bedrock(prompt)
        return categorization

    async def parse_categorization(self, categorization):
        """
        Parse the categorization output to extract the category, collaboration status, reason, and collaborators.
        """
        try:
            category_line = next(
                line for line in categorization.splitlines() if "Category:" in line
            )
            collaboration_line = next(
                line for line in categorization.splitlines() if "Collaboration:" in line
            )
            reason_line = next(
                line for line in categorization.splitlines() if "Reason:" in line
            )
            collaborators_line = next(
                line
                for line in categorization.splitlines()
                if "InitialCollaborators:" in line
            )

            category = category_line.split(":")[1].strip()
            collaboration = collaboration_line.split(":")[1].strip().lower() == "yes"
            reason = reason_line.split(":")[1].strip()
            collaborators = (
                collaborators_line.split(":")[1].strip().strip("[]").split(", ")
            )

            return category, collaboration, reason, collaborators
        except Exception as e:
            raise ValueError(
                f"Failed to parse categorization output: {categorization}. Error: {e}"
            )

    async def collaborative_iteration(
        self,
        userId,
        query,
        collaborators,
        conversation_summary=None,
        tools=None,
        image_path=None,
    ):
        """
        Perform collaborative iterations among multiple agents to refine the response.
        """
        current_response = query
        iteration = 1
        while True:
            yield f"\n**--- Collaboration Iteration {iteration} ---**"
            next_responses = []
            for collaborator in collaborators:
                yield f"\n**Agent {collaborator} processing...**"
                response = await self.agent_system.interact(
                    userId,
                    collaborator,
                    current_response,
                    conversation_summary,
                    tools,
                    image_path,
                )
                current_response = response
                yield f"\n**Agent {collaborator} Response:** {response}"
                next_responses.append(response)

            # Combine responses for further collaboration
            combined_response = "\n".join(next_responses)

            # Ask agents if they are satisfied with the current answer
            satisfaction_prompt = (
                f"You are collaborating agents. Here is the combined response:\n"
                f"{combined_response}\n\n"
                f"Are you satisfied with this response? If not, list the areas that need further improvement and additional iterations required.\n"
                f"Provide your answer in the format:\n"
                f"Satisfied: <Yes/No>\n"
                f"NextSteps: <List of improvements or refinements>"
            )
            satisfaction_check = await self.categorizer.send_to_bedrock(
                satisfaction_prompt
            )
            yield f"**Satisfaction Check:** {satisfaction_check}"

            try:
                satisfied_line = next(
                    line
                    for line in satisfaction_check.splitlines()
                    if "Satisfied:" in line
                )
                satisfied = satisfied_line.split(":")[1].strip().lower() == "yes"
            except Exception as e:
                raise ValueError(
                    f"Failed to parse satisfaction check output: {satisfaction_check}. Error: {e}"
                )

            if satisfied:
                yield "\n**--- Final Answer Reached ---**"
                break

            # If not satisfied, prepare for the next iteration
            current_response = combined_response
            iteration += 1
            if iteration > 3:
                yield "\n**--- Maximum Iteration Limit Reached ---**"
                break

    async def route_query(
        self, userId, query, conversation_summary, tools, image_path=None
    ):
        """
        Route the query to the appropriate agent(s) based on categorization.
        """
        yield "Performing query categorization...\n"
        conversation = (
            f"<query>{query}</query>\n"
            f"<conversation_summary>{conversation_summary}</conversation_summary>"
        )
        categorization = await self.categorize_query(conversation)
        yield f"**Categorization Output:** {categorization}"

        try:
            parsed_data = await self.parse_categorization(categorization)
            category, collaboration, reason, collaborators = parsed_data

            if not collaboration:
                # Single-agent response
                response = await self.agent_system.interact(
                    userId, category, query, conversation_summary, tools, image_path
                )
                yield response
            else:
                # Multi-agent collaborative response
                yield f"**Collaboration Required. Initial Collaborators:** {collaborators}"
                async for response in self.collaborative_iteration(
                    userId,
                    query,
                    collaborators,
                    conversation_summary,
                    tools,
                    image_path,
                ):
                    yield response

        except Exception as e:
            yield f"Failed to handle query routing: {e}"
