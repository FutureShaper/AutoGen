import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(
    #model="deepseek-r1:7b",
    model = "mistral",
    base_url="http://localhost:11434/v1",
    api_key="placeholder",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
    },
)

async def hello() -> None:
    agent = AssistantAgent("assistant", model_client)
    print(await agent.run(task="Say 'Hello World!'"))

async def main() -> None:
    assistant = AssistantAgent("assistant", model_client)
    web_surfer = MultimodalWebSurfer("web_surfer", model_client)
    user_proxy = UserProxyAgent("user_proxy")
    termination = TextMentionTermination("exit") # Type 'exit' to end the conversation.
    team = RoundRobinGroupChat([web_surfer, assistant, user_proxy], termination_condition=termination)
    await Console(team.run_stream(task="Find information about AutoGen and write a short summary."))

asyncio.run(main())
