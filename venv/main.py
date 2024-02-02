import openai
import uvicorn
from fastapi import FastAPI, Form, Request
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from config import api_key as OPENAI_API_KEY  # Ensure this is correct

app = FastAPI()

#set your OpenAI API key here
#OPEN_API_KEY = "key here"

def setup_chain():
    #Define file path and template
    file = 'Mental_Health_FAQ.csv'
    template = "How you doin?"  # Replace with your template

    #Initialize embeddings, loader, and prompt
    embeddings = OpenAIEmbeddings(OPENAI_API_KEY)
    loader = CSVLoader(filepath=file, encoding='utf-8')
    docs = loader.load()
    prompt = PromptTemplate(template=template, input_variables=OPENAI_API_KEY)

    #Create DocArrayInMemorySearch and retriever
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    chain_type_kwargs={"prompt": prompt}

    #Initialize ChatOpenAI
    llm = ChatOpenAI(temperature=0.7)

    #Setup RetrievalQA chain 
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    return chain

agent = setup_chain()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Chatbot"}

@app.post("/prompt")
def process_prompt(prompt: str = Form(...)):
    response = agent.run(prompt)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
