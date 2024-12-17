# streamlit
# google-generativeai
# python-dotenv
# langchain
# PyPDF2
# faiss-cpu
# langchain_google_genai
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Define the AI legal assistant class
class LegalAI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=self.api_key
        )
        self.msgs = StreamlitChatMessageHistory(key="legal_case_history")

    def process_legal_case(self, case_text):
        # Add the user's input to the message history
        self.msgs.add_message(HumanMessage(content=case_text))
        messages = [
            {"type": "system", "content": "You are a legal assistant."},
            {"type": "human", "content": case_text},
        ]
        # Get the AI response
        response = self.model.invoke(messages)
        
        # Handle response based on its type
        if isinstance(response, dict) and "content" in response:
            ai_content = response["content"]
        elif hasattr(response, "content"):  # For AIMessage or similar objects
            ai_content = response.content
        else:
            raise TypeError("Unexpected response type: {}".format(type(response)))

        # Add the response to the message history
        self.msgs.add_message(AIMessage(content=ai_content))
        return ai_content

    def answer_question(self, user_query):
        # Add the user's query to the message history
        self.msgs.add_message(HumanMessage(content=user_query))
        messages = [
            {"type": "system", "content": "You are a legal assistant."},
            *[
                {"type": msg.type, "content": msg.content}
                for msg in self.msgs.messages
            ],
        ]
        # Get the AI response
        response = self.model.invoke(messages)
        
        # Handle response based on its type
        if isinstance(response, dict) and "content" in response:
            ai_content = response["content"]
        elif hasattr(response, "content"):  # For AIMessage or similar objects
            ai_content = response.content
        else:
            raise TypeError("Unexpected response type: {}".format(type(response)))

        # Add the response to the message history
        self.msgs.add_message(AIMessage(content=ai_content))
        return ai_content


# Set up Streamlit UI
st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️")
st.title("Legal AI Assistant")
st.write("Welcome! This chatbot helps you analyze and interact with legal cases.")

# Get the API key
def get_api_key():
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    api_key = st.text_input("Enter your API key", type="password")
    return api_key

api_key = get_api_key()

# If API key is provided, start the assistant
if api_key:
    legal_ai = LegalAI(api_key)
    st.write("You can now interact with the legal AI assistant.")
    
    # Input for legal case text
    case_text = st.text_area("Paste the legal case text here:")
    
    if case_text:
        case_summary = legal_ai.process_legal_case(case_text)
        st.write(f"Case Summary: {case_summary}")
    
    # User's legal query
    user_query = st.text_input("Ask a legal question:")
    if user_query:
        response = legal_ai.answer_question(user_query)
        st.write(f"Answer: {response}")

else:
    st.warning("Please enter your API key.")