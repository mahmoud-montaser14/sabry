# Create VectorDB Function
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain_community.document_loaders import WebBaseLoader
import bs4
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


def Create_vectordb(PDF, embeddings):
    # loading the document
    text = ""
    pdf_reader = PdfReader(PDF)
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=300)

    texts = text_splitter.split_text(text)
    vectordb = FAISS.from_texts(texts=texts, embedding=embeddings)
    return vectordb


def Load_vectordb(Folder_Path, embeddings):
    return FAISS.load_local(folder_path=Folder_Path, embeddings=embeddings, allow_dangerous_deserialization=True)


def save_vectordb(vectordb, Folder_Path):
    vectordb.save_local(folder_path=Folder_Path)
    print("save successful")


def Update_VectorDB(Folder_Path, embeddings, Data, Type):
    vectordb = Load_vectordb(Folder_Path, embeddings)

    # splitting into text chunks
    if Type == 1:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len)
        text = ""
        pdf_reader = PdfReader(Data)
        for page in pdf_reader.pages:
            text += page.extract_text()
        texts = text_splitter.split_text(text)
        vectordb.add_texts(texts)
        print("VectorDB updated")
    if Type == 2:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300)
        loader = WebBaseLoader(web_paths=(Data,),
                               bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                                   class_=("mw-body"))))
        text_documents = loader.load()

        # Convert the list to an iterable
        documents = text_splitter.split_documents(text_documents)
        vectordb.add_documents(documents)
    return vectordb


def load_LLM_groq(vectordb, Model_name):
    retriever = vectordb.as_retriever()
    llm = ChatGroq(
        model=Model_name,
        temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True)
    return qa_chain


def chatbot(LLM, user_prompt: str):
    # append the question of user to message as a user roke
    # all_messages.append({'role': 'user', 'content': user_prompt})
    # ## also the question to questions list
    # all_prompts.append(user_prompt)
    response = LLM({"query": user_prompt})
    bot_response = f"Groq: {response['result']}"
    return bot_response
