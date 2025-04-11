import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from connect_memory_with_llm import load_llm, set_custom_prompt, chaining


custom_prompt_template = """
You are an AI assistant. You can ONLY answer using the information provided in the context below.
If the answer is not in the context, reply strictly with: "I don't know based on the provided information."

Do NOT use any external knowledge.
Do NOT guess or make up information.
Do NOT include any small talk.

Context:
{context}

Question:
{question}

Answer:
"""

hf_repo_id="mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def load_database(DB_FAISS_PATH):
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
    return db

def main():
    st.title("Ask ChatBot!")
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    query=st.chat_input("Pass your query here!")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({'role':'user','content':query})
        prompt=set_custom_prompt(custom_prompt_template)
        llm=load_llm(hf_repo_id)
        try:
            db=load_database(DB_FAISS_PATH)
            if db is None:
                st.error("Failed to load the DataBase.")
            qa_chain=chaining(prompt,db,llm)
            response=qa_chain.invoke({'query':query})
            result=response["result"]             
            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({'role':'assistant','content':result})
        except Exception as e:
            st.error(f"Error: {str(e)}") 

if __name__=="__main__":
    main()