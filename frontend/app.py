import streamlit as st
import requests

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Set page title and chatbot UI
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.markdown("""
<style>
    .chat-container {
        max-width: 600px;
        margin: auto;
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        height: 80vh;
    }
    .chat-box {
        display: flex;
        flex-direction: column;
        overflow-y: auto;
        height: 65vh;
        flex-grow: 1;
        padding: 10px;
    }
    .user-message {
        background-color: #C80E0E;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        max-width: 75%;
        align-self: flex-end;
    }
    .bot-message {
        background-color: #ffffff;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        max-width: 75%;
        align-self: flex-start;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 600px;
        background-color: #f8f9fa;
        padding: 10px;
        border-top: 1px solid #ccc;
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 Ooredoo Financial Chatbot")


for role, text in st.session_state.chat_history:
    if role == "User":
        st.markdown(f'<div class="user-message"> {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message"> {text}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input container fixed at the bottom
st.markdown('<div class="input-container">', unsafe_allow_html=True)
user_query = st.text_input("Type a message...", placeholder="Ask me any question about Ooredoo's financial data.")
st.markdown('</div>', unsafe_allow_html=True)

# Chatbot interaction
if st.button("Send") and user_query:
    response = requests.post("http://localhost:8000/query/", json={"input_query": user_query})
    bot_response = response.json().get("response", "I do not have enough information to answer your question.")
    
    # Save chat history
    st.session_state.chat_history.append(("User", user_query))
    st.session_state.chat_history.append(("Bot", bot_response))

    st.session_state.user_input = ""
    
    # Rerun to display updated chat
    st.experimental_rerun()
