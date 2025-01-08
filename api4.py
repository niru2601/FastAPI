import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import faiss

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['MODEL_NAME'] = os.getenv('MODEL_NAME')
os.environ['DIR_PATH'] = os.getenv('DIR_PATH')
os.environ['INDEX_DIR_PATH'] = os.getenv('INDEX_DIR_PATH')

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow frontend interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever
retriever = None

# Ensure the documents directory exists
documents_path = os.environ['DIR_PATH']
os.makedirs(documents_path, exist_ok=True)

# Ensure the index directory exists
index_path = os.environ['INDEX_DIR_PATH']
os.makedirs(index_path, exist_ok=True)

faiss_index_file = os.path.join(index_path, 'faiss_index_file.index')

def initialize_retriever(files):
    try:
        all_docs = []

        for uploaded_file in files:
            if uploaded_file.filename.lower().endswith('.pdf'):
                file_path = os.path.join(documents_path, uploaded_file.filename)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.file.read())

                doc_loader = PyPDFLoader(file_path)
                docs = doc_loader.load()
                all_docs.extend(docs)
            else:
                raise Exception(f"{uploaded_file.filename} (not a PDF)")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunk_docs = text_splitter.split_documents(all_docs)

        # Apply vector embedding on chunked documents
        embedding_model = OllamaEmbeddings(model=os.environ['MODEL_NAME'])
        db = FAISS.from_documents(chunk_docs, embedding_model)

        faiss.write_index(db.index, faiss_index_file)
        logger.info(f"FAISS index saved to {faiss_index_file}")

        return db.as_retriever()

    except Exception as e:
        logger.error("Error initializing retriever", exc_info=True)
        raise e

# Define an endpoint for uploading and processing PDFs
@app.post("/documents/upload/")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    global retriever
    if not files:
        logger.warning("No files provided")
        return JSONResponse(content={"error": "No files provided"}, status_code=400)
    try:
        logger.info(f"Processing {len(files)} file(s)")
        retriever = initialize_retriever(files)
        return {"message": f"{len(files)} PDFs processed successfully"}
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Define an endpoint for posting questions and getting answers
@app.post("/ask/question/")
async def ask_question(question: str = Form(...)):
    global retriever
    if not retriever:
        logger.warning("No retriever initialized")
        return JSONResponse(content={"error": "No retriever initialized. Please upload PDFs first."}, status_code=400)

    if not question:
        logger.warning("No question provided")
        return JSONResponse(content={"error": "No question provided"}, status_code=400)

    try:
        # Creating prompt template and LLM model
        my_prompt = ChatPromptTemplate.from_template(
            """
            Answer the question below based strictly on the provided context.
            - Use a step-by-step reasoning approach.
            - Provide a concise and detailed answer from provided context only.
            - If the context does not contain relevant information to answer the question, clearly state that the answer cannot be provided.
            - Add citations from the provided context if possible.
            <context>
                {context}
            </context>
            Question: {input}
            """
        )
        llm_model = OllamaLLM(model=os.environ['MODEL_NAME'])
        doc_chain = create_stuff_documents_chain(llm_model, my_prompt)

        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        # Get the answer
        result = retrieval_chain.invoke({"input": question})

        # retrieved_docs = result.get('context')
        #
        # # Inspect a retrieved document
        # for doc in retrieved_docs:
        #     print(doc.page_content)  # Content of the document
        #     print(doc.metadata)

        db_index = faiss.read_index(faiss_index_file)

        return {"answer": result["answer"]}

    except Exception as e:
        logger.error(f"Error handling question: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn api4:app --reload
