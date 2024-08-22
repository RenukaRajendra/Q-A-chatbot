import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## langsmith Tracking

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A CHATBOT with OpenAI"


prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant, respond to user query"),
        ("user","Question:{question}")

    ]
)

def generate_response(question,api_key,llm,temperature,amx_tokens):
    openai.api_key=api_key
    llm = ChatOpenAI(model=llm)
    output_parser=StrOutputParser()
    chain=llm|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer


st.title("Q and A chatbot with OPenai")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your openAI api key", type="password")

llm = st.sidebar.selectbox("Select the model",["gpt-4o","gpt-4 turbo","gpt-4"])

temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max_tokens",min_value=50,max_value=300,value=150)

st.write("Go ahead ask any question")
user_input = st.text_input("you:")

if user_input:
    response = generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")

