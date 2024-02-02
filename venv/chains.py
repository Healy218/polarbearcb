import openai
from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from config import api_key as OPENAI_API_KEY  # Correctly import the API key

app = FastAPI()

def setup_chain():
    # Define file path and template
    file_path = 'Mental_Health_FAQ.csv'
    template = "Your template here"  # Define your prompt template

    # Initialize embeddings with OpenAI API Key
    embeddings = OpenAIEmbeddings(api_key="sk-...BleU")

    # Load documents from CSV
    loader = CSVLoader(filepath=file_path, encoding='utf-8')
    docs = loader.load()

    # Initialize prompt template
    prompt = PromptTemplate(template=template, input_variables=docs)

    # Create document search and retriever
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # Initialize ChatOpenAI with corrected parameter name 'temperature'
    llm = ChatOpenAI(temperature=0.7)

    # Setup RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use a valid chain type
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},  # Corrected variable name
        verbose=True
    )

    return chain

# Initialize the agent by calling the setup function
agent = setup_chain()
