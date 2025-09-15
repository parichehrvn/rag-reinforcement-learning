from dotenv import load_dotenv
import os
import logging
import torch

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment. Make sure to set it if you want to call the Google LLM.")

# Check CUDA availability
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# Initialize embeddings, vector store and LLM
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={"device": 'cpu'})

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PERSIST_DIR = os.path.join(BASE_DIR, "storage")

vector_store = Chroma(embedding_function=embeddings,
                      persist_directory=PERSIST_DIR,
                      collection_name='rag_rl')

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)


# Define tools
@tool(response_format='content_and_artifact')
def retrieve(query: str):
    """Retrieve information related to a query"""
    try:
        retrieved_docs = vector_store.similarity_search(query, k=5)
        serialized = "\n\n".join(
            f"Source: {doc.metadata} \nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    except Exception as e:
        logger.error("Error during retrieval: %s", e)
        return {"context": []}


# Step1: Generate an AIMessage that may include a tool-call to be sent
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond"""
    llm_with_tools = llm.bind_tools(([retrieve]))
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": response}


# Step 2: Execute the retrieval
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content
def generate(state: MessagesState):
    """Generate final answer"""
    try:
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break

        tool_messages = recent_tool_messages[::-1]

        doc_content = "\n\n".join(doc.content for doc in tool_messages)

        system_message_content = f""" 
    You are an expert in reinforcement learning, acting as an instructor teaching a student. 
    Your task is to provide a very concise and short explanation of the answer, using the provided retrieved context from academic papers and blogs as the foundation. 
    Use a pedagogical tone to ensure the student understands the topic thoroughly. 
    If the context is insufficient or missing, acknowledge this and provide a general overview based on your expertise, noting the limitation.
    {doc_content}
    """
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Without streaming:
        response = llm.invoke(prompt)
        return {"messages": response}

    except Exception as e:
        logger.error("Error preparing generation: %s", e)
        return {"answer": iter([f"[Error preparing generation: {e}]"])}


# Build and compile graph
memory = MemorySaver()

graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"}
)

graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile(checkpointer=memory)

# Function to run the RAG pipeline
def run_rag(question: str, thread_id: str = "default_thread"):
    """
    Run the RAG pipeline with memory support and return the streamed answer.

    Args:
        question (str): The user's question.
        thread_id (str): Identifier for the conversation thread to maintain memory.

    Returns:
        Iterator[str]: Streamed response from the LLM.
    """
    try:
        # Initialize the input state with the user's question as a HumanMessage
        input_state = {
            "messages": [HumanMessage(content=question)]
        }

        config = {
            "configurable": {"thread_id": thread_id}
        }

        for message_chunk, metadata in graph.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="messages",
                config=config,
        ):
            if message_chunk.content and metadata['langgraph_node'] == 'generate' :
                yield message_chunk.content

        # state = graph.invoke(input_state, config=config)

    except Exception as e:
        logger.error("Error running RAG pipeline: %s", e)
        return iter([f"[Error running RAG pipeline: {e}]"])

# Example usage
# question = "What is Q-learning in reinforcement learning?"
# response = run_rag(question, thread_id="conversation_1")

