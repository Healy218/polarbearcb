import openai
import uvicorn
from fastapi import FastAPI, Request, Form
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch

app = FastAPI()

#set your OpenAI key here

OPENAI_API_KEY  = "your_api_key"

def setup_chain():
    #define file path and template"
    file = 'Mental_Health_FAQ.csv'
    template = """ #Template contents here """

    #Intialize embeddings, loader, and prompt
    embeddings = OpenAIEmbeddings()
    loader = CSVLoader(filepath=file, encoding='utf-8')
    docs = loader.load()
    prompt = PromptTemplate(template=template, input_variables=docs)

    #Create DocArrayInMemorySearch and retriever
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    chaim_type_kwargs = {"prompt":prompt}

    #initialize ChatOpenAI
    llm = ChatOpenAI(
        tempurature=0
    )

    #Setup RetievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    return chain

agent = setup_chain