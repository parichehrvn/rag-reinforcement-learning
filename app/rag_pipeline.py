from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
import torch
from typing import TypedDict, List
from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check CUDA availability
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# Initialize embeddings, vector store and LLM
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={"device": 'cpu'})

vector_store = Chroma(embedding_function=embeddings,
                      persist_directory='../storage',
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
    retrieved_docs = vector_store.similarity_search(state['question'], k=5)
    return {'context': retrieved_docs}


def generate(state: State):
    doc_content = "\n\n".join(doc.page_content for doc in state['context'])
    message = prompt.invoke(({'question': state['question'], 'context': doc_content}))
    # response = llm.invoke(message)
    # return {'answer': response.content}
    for chunk in llm.stream(message):
        yield chunk.content


# Build and compile graph
graph_builder = StateGraph(State)
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_edge('retrieve', 'generate')
graph_builder.set_entry_point('retrieve')
graph = graph_builder.compile()

# Function to run the RAG pipeline
def run_rag(question: str):
    return graph.invoke({"question": question, "context": [], "answer": ""})
