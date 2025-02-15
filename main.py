import asyncio
import faiss
import ollama
import numpy as np
from typing import Optional, List
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

llm_model_name = 'mistral'

# Custom LLM client for Ollama
class OllamaLLM(LLM, BaseModel):
    model_name: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response['text']

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    @property
    def _llm_type(self):
        return "ollama"

# Step 2: Placeholder text for RAG demo
texts = [
    "Simon says Pizza is the best.",
    "Pasta is best served with Pesto.",
    "Parmesan is a great source of Protein.",
]

# Step 3: Create embeddings for your data using the Ollama model
def get_embeddings(texts):
    embeddings = []
    for text in texts:
        response = ollama.embed(model='nomic-embed-text', input=text)
        #print(response)  # Print the response to inspect its structure
        if 'embeddings' in response:
            embeddings.append(response['embeddings'][0])  # Adjusted to extract from 'embeddings'
        else:
            raise KeyError(f"'embeddings' key not found in response: {response}")
    return embeddings

text_embeddings = np.array(get_embeddings(texts))

# Step 4: Build the FAISS index
dimension = len(text_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(text_embeddings)

# Create a docstore and index_to_docstore_id mapping
documents = [Document(page_content=text) for text in texts]
docstore = InMemoryDocstore(dict(enumerate(documents)))
index_to_docstore_id = {i: i for i in range(len(documents))}

# Wrap the FAISS index in a LangChain FAISS vector store
vector_store = FAISS(embedding_function=get_embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# Step 5: Save the FAISS index locally
vector_store.save_local("./faiss.index")

# Define a custom wrapper for the RetrievalQA chain
class CustomRetrievalQA:
    def __init__(self, retrieval_qa, name, description):
        self.retrieval_qa = retrieval_qa
        self.name = name
        self.description = description
        self.args_schema = None  # Add the args_schema attribute

    def _run(self, *args, **kwargs):
        return self.retrieval_qa(*args, **kwargs)

# Define the RAG tool
def create_rag_tool():
    # Create a FAISS vector store
    vector_store = FAISS.load_local("./faiss.index", embeddings=get_embeddings, allow_dangerous_deserialization=True)

    # Create a custom LLM client for Ollama
    llm = OllamaLLM(model_name=llm_model_name)

    # Create a RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Wrap the chain in a custom wrapper with a name and description
    custom_rag_chain = CustomRetrievalQA(rag_chain, name="RAG_Tool", description="Retrieve and generate information based on a given query")

    # Wrap the custom chain in a LangChainToolAdapter
    rag_tool = LangChainToolAdapter(custom_rag_chain)

    return rag_tool

rag_tool = create_rag_tool()

model_client = OpenAIChatCompletionClient(
    model = llm_model_name,
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

# Create an agent for the RAG tool
rag_agent = AssistantAgent(
    name="RAG_Agent",
    tools=[rag_tool],
    model_client=model_client,
    description="An agent that uses RAG to retrieve and generate information based on a given query",
    system_message="You are a helpful AI assistant. Use the RAG tool to retrieve additional information from local files. Then immediately create your response and append to your response the word 'TERMINATE'",
    #system_message="For testing purposes please ALWAYS only answer with the word 'TERMINATE'"
)

async def hello_world() -> None:
    assistant = AssistantAgent("assistant", model_client)
    team = RoundRobinGroupChat([web_surfer, assistant, user_proxy, rag_agent], termination_condition=termination)
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

    return results

arxiv_search_tool = FunctionTool(
    arxiv_search, description="Search Arxiv for papers related to a given topic, including abstracts"
)

arxiv_search_agent = AssistantAgent(
    name="Arxiv_Search_Agent",
    tools=[arxiv_search_tool],
    model_client=model_client,
    description="An agent that can search Arxiv for papers related to a given topic, including abstracts",
    system_message="You are a helpful AI assistant. Solve tasks using your tools. Specifically, you can take into consideration the user's request and craft a search query that is most likely to return relevant academic papers.",
)

report_agent = AssistantAgent(
    name="Report_Agent",
    model_client=model_client,
    description="Generate a report based on a given topic",
    system_message="You are a helpful assistant. Your task is to synthesize data extracted into a high quality literature review including CORRECT references. You MUST write a final report that is formatted as a literature review with CORRECT references. Your response should end with the word 'TERMINATE'",
)

team = RoundRobinGroupChat(
    #participants=[arxiv_search_agent, report_agent, rag_agent], termination_condition=termination
    participants=[rag_agent], termination_condition=termination
)

async def main() -> None:
    await Console(
        team.run_stream(
            task="Which italian food is a great source of protein?"
        )
    )

asyncio.run(main())