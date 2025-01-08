import os
from io import BytesIO
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['MODEL_NAME'] = os.getenv('MODEL_NAME')

# Streamlit app UI setup
st.title('Welcome to CloverAI ')

# Directory path selection using file uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    try:
        # Process uploaded PDF files
        all_docs = []
        for uploaded_file in uploaded_files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            # Load the PDF using PyPDFLoader
            doc_loader = PyPDFLoader(temp_file_path)
            docs = doc_loader.load()
            all_docs.extend(docs)

            # Clean up the temporary file
            os.unlink(temp_file_path)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunk_docs = text_splitter.split_documents(all_docs)

        # Apply vector embedding on chunked documents
        model_name = os.environ.get('MODEL_NAME')
        if not model_name:
            raise ValueError("MODEL_NAME environment variable is not set.")

        embedding_model = OllamaEmbeddings(model=model_name)
        db = FAISS.from_documents(chunk_docs, embedding_model)

        # Design ChatPrompt template
        my_prompt = ChatPromptTemplate.from_template(
            """
            Answer the question below based strictly on the provided context.
            - Use a step-by-step reasoning approach.
            - Provide a concise and detailed answer.
            - If the context does not contain relevant information to answer the question, clearly state that the answer cannot be provided.
            <context>
                {context}
            </context>
            Question: {input}
            """
        )

        # Load LLM model
        llm_model = OllamaLLM(model=model_name)

        # Create chain to process documents
        doc_chain = create_stuff_documents_chain(llm_model, my_prompt)

        # Create retriever from vector store
        retriever = db.as_retriever()

        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        # User input for question
        input_text = st.text_input("Ask Your Question about the uploaded PDFs:", key="user_input")

        # Display the output of the model
        if input_text:
            result = retrieval_chain.invoke({'input': input_text})
            st.write(result['answer'])

    except Exception as e:
        st.error(f"An error occurred: {type(e).__name__} - {e}")

# streamlit run app4.py