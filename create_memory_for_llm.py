from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#step 1: load raw pdfs
data_path=r"D:\VS Code\ChatBot with RAG\data"
def load_pdf_files(data):
    loader=DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents=loader.load()
    return documents
documents=load_pdf_files(data=data_path)


 #step 2: create chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks
chunks=create_chunks(extracted_data=documents)


#step 3:create vector embessings
def get_embedding_model():
   embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
   return embedding_model
embedding_model=get_embedding_model()

#step 4: store embeddings in FAISS
def store_embeddings(chunks,embedding_model):
    DB_FAISS_PATH="vectorstore/db_faiss"
    db=FAISS.from_documents(chunks,embedding_model)
    db.save_local(DB_FAISS_PATH)
store_embeddings(chunks,embedding_model)