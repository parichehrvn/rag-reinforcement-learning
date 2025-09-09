# app/main.py
import streamlit as st
from rag_pipeline import run_rag

st.title("Reinforcement Learning Q&A")

# Input for user question
question = st.text_input("Ask a question about reinforcement learning:", "")

# Button to trigger query
if st.button("Get Answer"):
    if question:
        # Call the RAG pipeline (to be implemented in rag_pipeline.py)
        result = run_rag(question)
        # Display answer
        st.write("**Answer:**")
        st.write(result["answer"])
        # Display source documents
        st.write("**Sources:**")
        for doc in result["context"]:
            st.write(f"- {doc.page_content[:100]}... (Source: {doc.metadata.get('source', 'unknown')})")
    else:
        st.error("Please enter a question.")