import streamlit as st
from rag_pipeline import run_rag


def get_response(question):
    response = run_rag(question)
    return response




st.title('Reinforcement Learning Q&A')

def home():
    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    # 1) Render chat history (always first)
    for msg in st.session_state["messages"]:
        avatar = "🧑" if msg["role"] == "User" else "🤖"
        st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])

    # 2) Input
    question = st.chat_input("Ask a question to learn Reinforcement Learning:")
    if question:
        # 3) Append & show user's message immediately
        st.session_state["messages"].append({"role": "User", "content": question})
        st.chat_message("User", avatar="🧑").markdown(f"**{question}**")

        # 4) Append placeholder assistant msg and stream chunks into it
        with st.chat_message("Assistant", avatar="🤖"):
            placeholder = st.empty()
            st.session_state["messages"].append({"role": "Assistant", "content": ""})
            idx = len(st.session_state["messages"]) - 1  # index of assistant placeholder

            full_response = ""
            for chunk in get_response(question):  # yields chunks
                full_response += chunk
                st.session_state["messages"][idx]["content"] = full_response  # persist partials
                placeholder.markdown(full_response)


with st.spinner('Fetching your data...⌛'):
    home()

