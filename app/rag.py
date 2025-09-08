# app/rag_pipeline.py
from typing import List
from langgraph.graph import StateGraph
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Gemini
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from typing import TypedDict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize embeddings, vector store, and LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = Gemini(model_name="gemini-pro", api_key=gemini_api_key)

# Define custom prompt for reinforcement learning Q&A
prompt_template = """
You are an expert in reinforcement learning. 
Answer the question like an instructor with details, using the provided context from academic papers and blogs. 
If the context doesn't provide enough information, state that clearly.

Question: {question}
Context: {context}
Answer:
"""
prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

# Define state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=5)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build and compile graph
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge("retrieve", "generate")
graph_builder.set_entry_point("retrieve")
graph = graph_builder.compile()

# Function to run the RAG pipeline
def run_rag(question: str):
    return graph.invoke({"question": question, "context": [], "answer": ""})