import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

import os
from dotenv import load_dotenv

load_dotenv()

# Set environment variables
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A CHATBOT with OpenAI"

# Define the chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant, respond to user query"),
        ("user", "Question:{question}")
    ]
)


def generate_response(question, llm, temperature, max_tokens):
    # Initialize the LLM
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()

    # Create a chain from LLM and output parser
    chain = llm | output_parser

    # Provide the question as a string input
    answer = chain.invoke(question)

    return answer


# Streamlit interface
st.title("Q and A chatbot with Ollama")

# Sidebar inputs
llm = st.sidebar.selectbox("Select the model", ["llama3"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max_tokens", min_value=50, max_value=300, value=150)

# User input
st.write("Go ahead ask any question")
user_input = st.text_input("you:")

# Display the response
if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")
