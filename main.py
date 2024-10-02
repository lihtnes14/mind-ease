import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Inject custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #191825;
            color: #865DFF;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
        }
        .stApp {
            background: #191825;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.5);
        }
        .title h1 {
            color: #865DFF;
            text-align: center;
            font-weight: 600;
        }
        textarea {
            border: 1px solid #E384FF;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
            background-color: #191825;
            color: #865DFF;
            outline: none; 
        }
        textarea:focus {
            border-color: #E384FF; 
            box-shadow: 0 0 5px rgba(227, 132, 255, 0.5); 
        }
        .stButton>button {
            background-color: #FFA3FD;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            margin: 10px 0;
        }
        .stButton>button:hover {
            background-color: #E384FF;
            color: white;
            border-color:white;
        }
        .response {
            background-color: #865DFF;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 18px;
            line-height: 1.5;
            color: #191825;
        }
        .form-header {
            text-align: center;
            margin-bottom: 20px;
        }
        [data-testid="stWidgetLabel"] {
            background-color: #E384FF;
            font-size: 20px;
            color: white;
            border-radius: 15px; 
            padding: 5px 10px; 
            display: inline-block;
            margin-bottom: 10px;
        }
        .question {
            background-color: #dbc6f4;
            color: #191825;
            padding: 10px;
            margin-top: 10px;
            border-radius: 8px;
        }
        .response {
            background-color: #865DFF;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            color: #191825;
        }
    </style>
""", unsafe_allow_html=True)

# Set up the Streamlit app interface
st.markdown("<div class='title'><center><h1>MindEase</h1></center></div>", unsafe_allow_html=True)
st.markdown("<div class='subheading'><center><h3>Your Mental Health Companion🫂❤️</h3></center></div>", unsafe_allow_html=True)

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

# Store questions and responses
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Set up the form for user input
with st.form("input_form"):
    user_input = st.text_area("What's on your mind today? Let us help.")
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

        # Store the question and response in session state
        st.session_state.conversation_history.append((user_input, response))

# Display the conversation history (questions and responses)
if st.session_state.conversation_history:
    for idx, (question, response) in enumerate(st.session_state.conversation_history):
        # Display the question with different background and margin
        st.markdown(f"<div class='question'>Question {idx + 1}: {question}</div>", unsafe_allow_html=True)
        
        # Display the response with different background and margin
        st.markdown(f"<div class='response'>{response}</div>", unsafe_allow_html=True)
