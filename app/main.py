import streamlit as st
from rag_pipeline import run_rag


@st.cache_data
def get_response(question):
    response = run_rag(question)
    return response


st.title('Reinforcement Learning Q&A')

# Input for user question
question = st.text_input('Ask a question to learn Reinforcement Learning:', '')

# Button to trigger query
if st.button('Get Answer 🤖'):
    if question:
        placeholder = st.empty()
        full_response = ""

        # Call the rag pipeline
        for chunk in get_response(question):
            full_response += chunk
            placeholder.markdown(full_response)  # update progressively

    else:
        st.error("Please enter a question.")



# import streamlit as st
#
# st.title("RAG Chatbot")
#
# question = st.chat_input("Ask a question")
# if question:
#     with st.chat_message("user"):
#         st.write(question)
#
#     with st.chat_message("assistant"):
#         response_stream = generate({"question": question, "context": [...]})
#         st.write_stream(response_stream)
