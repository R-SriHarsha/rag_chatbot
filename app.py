import streamlit as st
from search_pipeline.rag_chain import build_rag_chain


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Document-Based RAG Chatbot")
st.write("Ask questions from your uploaded PDFs")


@st.cache_resource
def load_chain():
    return build_rag_chain()

rag_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.sidebar.title("⚙️ Options")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask your question...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = rag_chain(user_input)

    
    with st.chat_message("assistant"):
        st.markdown(response["answer"])

        
        with st.expander("🔍 Retrieved Context"):
            for chunk in response["context"]:
                st.write(chunk)

    st.session_state.messages.append(
        {"role": "assistant", "content": response["answer"]}
    )


    if response.get("needs_confirmation"):
        st.session_state.pending_question = user_input


if st.session_state.pending_question:

    st.warning("This question may be outside uploaded documents.")

    col1, col2 = st.columns(2)

    if col1.button("Answer using General Knowledge"):
        question = st.session_state.pending_question

        with st.spinner("Generating general answer..."):
            response = rag_chain(question, allow_general=True)

        with st.chat_message("assistant"):
            st.markdown(response["answer"])

        st.session_state.messages.append(
            {"role": "assistant", "content": response["answer"]}
        )

        st.session_state.pending_question = None

    if col2.button("Cancel"):
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Ok, Please ask a question related to the uploaded documents."
            }
        )

        st.session_state.pending_question = None
        st.rerun()
