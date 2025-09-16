import streamlit as st

from rag_pipeline import run_rag


def get_response(question):
    response = run_rag(question)
    return response


st.set_page_config(page_title="Reinforcement Learning Q&A",
                   page_icon="🤖",
                   layout="centered")

def home():
    # st.set_page_config(page_title="Reinforcement Learning Q&A",
    #                    page_icon="🤖",
    #                    layout="centered")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Initial Greeting
    if not st.session_state["messages"]:
        st.session_state["messages"].append({
            "role": "Assistant",
            "content": "👋 Hi! I’m your Reinforcement Learning assistant. Ask me anything to get started."
        })

    st.markdown(
        """
        <div style="text-align: center; padding: 2rem;">
            <h1>Reinforcement Learning Q&A</h1>
            <p style="color: gray;">Ask anything about Reinforcement Learning. Powered by RAG.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1) Render chat history (always first)
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            avatar = "🧑" if msg["role"] == "user" else "🤖"
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

            # with st.spinner('Thinking...⌛'):
            for chunk in get_response(question):  # yields chunks
                full_response += chunk
                st.session_state["messages"][idx]["content"] = full_response  # persist partials
                placeholder.markdown(full_response)


home()

