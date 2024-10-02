import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Set up the Streamlit app interface
st.title("MindEase - Mental Health Assistant")

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

# Function to create a retriever from the chunks of text
def get_retriever(chunks, db_dir):
    vector_store = Chroma.from_texts(
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        texts=chunks,
        persist_directory=db_dir
    )
    retriever = vector_store.as_retriever()
    return retriever

# Set up the form for user input
with st.form("input_form"):
    user_input = st.text_area("Enter your mental health-related question:")
    submitted = st.form_submit_button("Submit")

    # Process the input if the form is submitted
    if submitted and user_input:
        # Example context to simulate retriever result
        context = """
        Mental health includes emotional, psychological, and social well-being. It affects how we think, feel, and act. It also helps determine how we handle stress, relate to others, and make choices.
        """
        
        # Define LLM (Gemini)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        # Define prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a mental health assistant. Your job is to provide emotional support, coping strategies, and guidance for mental health-related issues.
            Use the following context to help guide your response.

            Context: {context}  
            User Question: {question}

            Response:
            """
        )

        # Generate the response using the prompt and the LLM
        prompt = prompt_template.format(context=context, question=user_input)
        response = llm.predict(prompt)

        # Display the response in the Streamlit app
        st.write("Response:")
        st.write(response)
        