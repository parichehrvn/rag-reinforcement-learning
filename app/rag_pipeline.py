from typing import TypedDict, List, Iterator, Union
from dotenv import load_dotenv
import os
import logging
import torch

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph


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

# Define custom conversational prompt for reinforcement learning Q&A
prompt_template = """
You are an expert in reinforcement learning, acting as an instructor teaching a student. 
Your task is to provide a detailed, step-by-step explanation of the answer, using the provided context from academic papers and blogs as the foundation. 
Use a pedagogical tone to ensure the student understands the topic thoroughly. 
If the context is insufficient or missing, acknowledge this and provide a general overview based on your expertise, noting the limitation.

Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{question}"),
    ]
)


# Define state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define steps
def retrieve(state: State):
    try:
        retrieved_docs = vector_store.similarity_search(state["question"], k=5)
        return {"context": retrieved_docs}
    except Exception as e:
        logger.error("Error during retrieval: %s", e)
        return {"context": []}


def generate(state: State):
    try:
        doc_content = "\n\n".join(doc.page_content for doc in state["context"])
        message = prompt.invoke({"question": state["question"], "context": doc_content})

        def stream_answer():
            try:
                for chunk in llm.stream(message):
                    yield chunk.content
            except Exception as e:
                logger.error("Error during streaming: %s", e)
                yield f"[Error generating response: {e}]"

        return {"answer": stream_answer()}

    except Exception as e:
        logger.error("Error preparing generation: %s", e)
        return {"answer": iter([f"[Error preparing generation: {e}]"])}


# Build and compile graph
graph_builder = StateGraph(State)
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_edge('retrieve', 'generate')
graph_builder.set_entry_point('retrieve')
graph = graph_builder.compile()

# Function to run the RAG pipeline
def run_rag(question: str):
    # Run the RAG pipeline and stream the answer.
    state = graph.invoke({"question": question, "context": [], "answer": None})
    return state["answer"] if state["answer"] else iter(["[No answer generated]"])