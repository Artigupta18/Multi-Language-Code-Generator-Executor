import sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import google.generativeai as genai
import subprocess
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.callbacks import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler

# Set up Gemini API
callback_manager = CallbackManager([BaseCallbackHandler()])

api_key=st.secrets["API_KEY"]
genai.configure(api_key=api_key)

# Load Free Sentence Transformer Model for Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to execute code securely
def execute_code(code: str,  language: str) -> str:
    """Executes Python code and returns the output."""
    try:
        if language == "Python":
            result = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=5)
        elif language == "JavaScript":
            result = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=5)
        elif language == "Bash":
            result = subprocess.run(["bash", "-c", code], capture_output=True, text=True, timeout=5)
        elif language == "Java":
            with open("Main.java", "w") as f:
                f.write(code)
            subprocess.run(["javac", "Main.java"], capture_output=True, text=True)  # Compile
            result = subprocess.run(["java", "Main"], capture_output=True, text=True, timeout=5)
        elif language == "C++":
            with open("main.cpp", "w") as f:
                f.write(code)
            subprocess.run(["g++", "main.cpp", "-o", "main"], capture_output=True, text=True)  # Compile
            result = subprocess.run(["./main"], capture_output=True, text=True, timeout=5)
        elif language == "C":
            with open("main.c", "w") as f:
                f.write(code)
            subprocess.run(["gcc", "main.c", "-o", "main"], capture_output=True, text=True)  # Compile
            result = subprocess.run(["./main"], input="5\n", capture_output=True, text=True, timeout=10)
        elif language == "R":
            with open("script.R", "w") as f:
                f.write(code)
            result = subprocess.run(["Rscript", "script.R"], capture_output=True, text=True, timeout=5)
        else:
            return f"Error: Unsupported language '{language}' selected."

        return result.stdout if result.stdout else result.stderr
    except Exception as e:
        return str(e)

# LangChain Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define function for code generation using Gemini API
def generate_code(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")  # Load the Gemini model
    response = model.generate_content(prompt)  # Correct method for text generation
    return response.text

# Streamlit UI
st.title("Multi-Language Code Generator & Executor")

# User input for code generation
user_prompt = st.text_area("Describe the code you need:")

if st.button("Generate Code"):
    with st.spinner("Generating code..."):
        generated_code = generate_code(user_prompt)
        st.code(generated_code, language="python")

# User input for code execution
language_options = ["Python","C++","Java","C","JavaScript","Bash","R"]
selected_language = st.selectbox("Select the Programming Language:", language_options)
user_code = st.text_area("Enter Code to Execute:")

if st.button("Run Code"):
    with st.spinner("Executing code..."):
        result = execute_code(user_code, selected_language)
        st.write("Output:", result)
