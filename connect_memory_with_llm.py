from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

#step 1: set up LLM(Mistral with huggingface)

load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")
def load_llm(hf_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=hf_repo_id,
        temperature=0.5,
        task="text-generation",
        model_kwargs={
            "token":HF_TOKEN,
            "max_length":"512"}
    )
    return llm

#step 2: connect LLM with FAISS and create chain

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context","question"]
    )
    return prompt


def chaining(prompt,db,llm):
        qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':3}),
        chain_type_kwargs={"prompt":prompt}
        )
        return qa_chain