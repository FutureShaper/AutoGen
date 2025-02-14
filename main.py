import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

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

web_surfer = MultimodalWebSurfer("web_surfer", model_client)
user_proxy = UserProxyAgent("user_proxy")
termination = TextMentionTermination("TERMINATE")

async def hello_world() -> None:
    assistant = AssistantAgent("assistant", model_client)
    team = RoundRobinGroupChat([web_surfer, assistant, user_proxy], termination_condition=termination)
    await Console(team.run_stream(task="Find information about AutoGen and write a short summary."))

def arxiv_search(query: str, max_results: int = 2) -> list:  # type: ignore[type-arg]
    """
    Search Arxiv for papers and return the results including abstracts.
    """
    import arxiv

    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    results = []
    for paper in client.results(search):
        results.append(
            {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d"),
                "abstract": paper.summary,
                "pdf_url": paper.pdf_url,
            }
        )

    # # Write results to a file
    # with open('arxiv_search_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)

    return results

arxiv_search_tool = FunctionTool(
    arxiv_search, description="Search Arxiv for papers related to a given topic, including abstracts"
)

arxiv_search_agent = AssistantAgent(
    name="Arxiv_Search_Agent",
    tools=[arxiv_search_tool],
    model_client=model_client,
    description="An agent that can search Arxiv for papers related to a given topic, including abstracts",
    system_message="You are a helpful AI assistant. Solve tasks using your tools. Specifically, you can take into consideration the user's request and craft a search query that is most likely to return relevant academi papers.",
)

report_agent = AssistantAgent(
    name="Report_Agent",
    model_client=model_client,
    description="Generate a report based on a given topic",
    system_message="You are a helpful assistant. Your task is to synthesize data extracted into a high quality literature review including CORRECT references. You MUST write a final report that is formatted as a literature review with CORRECT references.  Your response should end with the word 'TERMINATE'",
)

team = RoundRobinGroupChat(
    participants=[arxiv_search_agent, report_agent], termination_condition=termination
)

async def main() -> None:
    await Console(
    team.run_stream(
        task="Write a literature review on ML Ops.",
        )
    )

asyncio.run(main())
