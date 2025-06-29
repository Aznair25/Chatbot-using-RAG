# 🧠 ChatRAG

An interactive chatbot powered by LangChain, Mistral LLM, HuggingFace, and FAISS — designed to answer questions based strictly on a set of handouts from the DBMS course by Virtual University Pakistan. This project demonstrates Retrieval-Augmented Generation (RAG) using sentence-transformer embeddings and a Streamlit-based frontend.

---

## 📂 Project Structure

```
├── create_memory_for_llm.py       # Loads PDFs, chunks text, stores embeddings
├── connect_memory_with_llm.py     # Loads Mistral LLM and sets up RAG chain
├── main.py                        # Streamlit frontend for chatbot interface
├── requirements.txt               # Project dependencies
├── .gitignore                     # Ignored files and directories
├── data/                          # Contains DBMS course handouts (PDFs)
└── vectorstore/                   # FAISS vector database (auto-generated)
```

---

## ⚙️ Features

- 📄 **PDF-Based Memory**: Ingests pdf files using LangChain document loaders  
- 🔍 **RAG Pipeline**: Retrieves context-relevant chunks from FAISS  
- 🧠 **Mistral LLM**: Uses HuggingFace Inference Endpoint  
- 💬 **Streamlit UI**: Simple chat interface with session memory  
- 🔐 **Env-based Token Management**: Keeps HuggingFace API key secure
- 🚫 **Smart Context Handling**: If a question falls outside the DBMS scope, the bot responds with a clear message like:  
  _"I don't know based on the provided information. The context provided is about Database Management Systems and does not contain information about cancer."_

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Aznair25/ChatRAG.git
cd ChatRAG
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add `.env` File

Create a `.env` file in the root directory:

```
HF_TOKEN=your_huggingface_token
```

> ⚠️ Replace `your_huggingface_token` with your actual [Hugging Face API token](https://huggingface.co/settings/tokens).

### 4. Add data files

Place the PDF files in the `data/` directory. This content forms the chatbot's memory.

### 5. Generate Embeddings

Run:

```bash
python create_memory_for_llm.py
```

This will split documents and store them as FAISS vectors.

### 6. Launch Chatbot UI

```bash
streamlit run main.py
```

---

## 📸 Screenshot

> ![alt text](<Chatbot UI (1).jpeg>) ![alt text](<Chatbot UI (2).jpeg>)
---

## ❓ How It Works

- **PDF Loading**: LangChain loads the PDF files using `PyPDFLoader`
- **Chunking**: Text is split using `RecursiveCharacterTextSplitter`
- **Embedding**: Uses `sentence-transformers/all-MiniLM-L12-v2`
- **Vector Store**: Stored with FAISS locally
- **LLM**: Mistral-7B Instruct via HuggingFace Inference Endpoint
- **Chain**: A `RetrievalQA` chain restricts responses strictly to provided context

---


## 🙌 Acknowledgments

- [LangChain](https://www.langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- [Virtual University of Pakistan](https://www.vu.edu.pk/) – for the DBMS course handouts(PDF file used)

```
