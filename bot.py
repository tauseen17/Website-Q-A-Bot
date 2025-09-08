import os
from dotenv import load_dotenv
load_dotenv()
os.environ["USER_AGENT"] = "WebsiteQA-Bot/1.0"


import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEndpointEmbeddings,ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate




# Initialize the language model
llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
  
)
model=ChatHuggingFace(llm=llm)

#create the output parser
parser=StrOutputParser()


#create the prompt
prompt=PromptTemplate(
    template="""
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context:{context}
    Question: {question}""",
    input_variables=["context","question"]
)

chain= prompt | model | parser 



st.title("Website Q&A Bot")

url=st.text_input("Enter website URL")

if url:
    loader=WebBaseLoader(url)
    docs=loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunk=splitter.split_documents(docs)

    #embed the documents
    embeddings=HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction"
        )
    vector_store=FAISS.from_documents(chunk,embeddings)

    retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})

    query=st.text_input("Enter the question about the content of the website")

    if query:
        docs=retriever.invoke(query)
        context="\n\n".join([docs.page_content for docs in docs])

        answer=chain.invoke({"context":context,"question":query})

        st.write(answer)


