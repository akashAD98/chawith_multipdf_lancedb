
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
import os
from langchain.chains import RetrievalQA
import lancedb
import openai
from langchain.llms import CTransformers
from langchain.vectorstores import LanceDB
import gradio as gr


from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import lancedb
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

import os
import re
import getpass
import langchain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from typing import List, Union

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os

import os
import re
import getpass
import langchain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from typing import List, Union

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Set OpenAI API key as an environment variable
#os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
#OPENAI_API_KEY  =' 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
# '

def find_urls(text: str) -> List:

    # Regular expression to match common URLs and ones starting with 'www.'
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.findall(text)



def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def load_pdf(path):
    # Specify the path to your PDF folder
    pdf_folder_path = '/content/pdf'

    # Load PDF documents
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    documents = loader.load()

    return documents


def website_loader(website: Union[str, list[str]]) -> List[langchain.schema.document.Document]:

    print("Loading website(s) into Documents...")
    documents = WebBaseLoader(web_path=website).load()
    print("Done loading website(s).")
    return documents


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def split_text(documents: List) -> List[langchain.schema.document.Document]:

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=120,
                                                   chunk_overlap=20,
                                                   length_function=len
                                                   )
    chunks = text_splitter.transform_documents(documents)
    print("Done splitting documents.")
    return chunks

def get_document_embeddings(chunks: List) :

    print("Creating embedder...")
    
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device':"cpu"})
    
    #OpenAIEmbeddings()
    # embedder = CacheBackedEmbeddings.from_bytes_store(
    #     core_embeddings_model,
    #     store,
    #     namespace=core_embeddings_model.model
    # )
    print("Done creating embedder")
    return embedder

def create_vector_store(chunks: List[langchain.schema.document.Document],
                        embedder):
  
    print("Creating vectorstore...")
    #vectorstore = FAISS.from_documents(chunks, embedder)
    #return vectorstore

    db = lancedb.connect('/tmp/lancedb')
    table = db.create_table("pdf_sear1ch", data=[
        {"vector": embedder.embed_query("Hello World"), "text": "Hello World", "id": "1"}
    ], mode="overwrite")
    vectorstore = LanceDB.from_documents(chunks, embedder, connection=table)
    return vectorstore


def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "mistral-7b-instruct-v0.1.Q5_K_M.gguf",    
        model_type="mistral"
    )
    return llm
#llm=load_llm()



def create_retriever(vectorstore) :

    print("Creating vectorstore retriever...")
    retriever = vectorstore.as_retriever()
    return retriever


def embed_user_query(query: str) -> List[float]:

    core_embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device':"cpu"})
    
    #OpenAIEmbeddings()
    embedded_query = core_embeddings_model.embed_query(query)
    return embedded_query

def similarity_search(vectorstore: langchain.vectorstores,
                      embedded_query: List[float]) -> List[langchain.schema.document.Document]:

    response = vectorstore.similarity_search_by_vector(embedded_query, k=4)
    return response


def create_chatbot(retriever: langchain.vectorstores) -> langchain.chains.conversational_retrieval:

    #llm = ChatOpenAI(model="gpt-3.5-turbo")
    llm = load_llm()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
        )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
        )
    return conversation_chain

def chat(conversation_chain, input: str) -> str:

    return conversation_chain.run(input)

text = """
I need you to go to the following URLs and get information about them
https://medium.com/cometheartbeat/how-to-prevent-yourself-from-wasting-time-on-your-data-science-project-728db69a4afc
and this one https://medium.com/mlearning-ai/langchain-is-the-past-here-is-the-future-of-llm-based-apps-46663f532c19
"""


######@below working code   ##########

# This chatbot_instance will be initialized once a URL is provided.
chatbot_instance = None

def respond(message, chat_history):
    global chatbot_instance
    urls = find_urls(message)

    # If the chatbot is not yet initialized and we have URLs, initialize it
    if not chatbot_instance and urls:
        #documents = load_pdf(path)
        documents = website_loader(urls)
        chunks = split_text(documents)
        embedder = get_document_embeddings(chunks)
        vectorstore = create_vector_store(chunks, embedder)
        retriever = create_retriever(vectorstore)
        chatbot_instance = create_chatbot(retriever)
        bot_message = "Chatbot initialized! How can I help you?"
    else:
        if chatbot_instance:
            bot_message = chat(chatbot_instance, message)
        else:
            bot_message = "Please provide a URL to initialize the chatbot first."

    chat_history.append((message, bot_message))
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    user_query = gr.Textbox(label="Your Query", placeholder="What would you like to chat about?")
    clear = gr.ClearButton([user_query, chatbot])

    user_query.submit(respond, [user_query, chatbot], [user_query, chatbot])

demo.launch(share=True, debug=True)



########## correct ui vcode ##########@@@

# Initialize chatbot_instance as None
#chatbot_instance = None

# Create a Gradio interface with PDF input option and query box
# def respond(pdf_paths, query, chat_history):
#     global chatbot_instance

#     text = get_pdf_text(pdf_paths)

#     if not chatbot_instance:
#         chunks = split_text(text)
#         embedder = get_document_embeddings(chunks)
#         vectorstore = create_vector_store(chunks, embedder)
#         retriever = create_retriever(vectorstore)
#         chatbot_instance = create_chatbot(retriever)
#         bot_message = "Chatbot initialized! How can I help you?"
#     else:
#         bot_message = chat(chatbot_instance, query)
    
#     chat_history.append((query, bot_message))
#     return bot_message, chat_history



# def respond(pdf_paths, query, chat_history):
#     global chatbot_instance

#     text = get_pdf_text(pdf_paths)

#     if not chatbot_instance:
#         chunks = split_text(text)
#         embedder = get_document_embeddings(chunks)
#         vectorstore = create_vector_store(chunks, embedder)
#         retriever = create_retriever(vectorstore)
#         chatbot_instance = create_chatbot(retriever)
#         bot_message = "Chatbot initialized! How can I help you?"
#     else:
#         bot_message = chat(chatbot_instance, query)
    
#     chat_history.append((query, bot_message))
#     return bot_message, chat_history


# iface = gr.Interface(
#     fn=respond,
#     inputs=[
#         gr.File(type="file", label="Upload PDF Files", multiple=True),
#         gr.Textbox(label="Your Query", placeholder="What would you like to ask the chatbot?"),
#     ],
#     outputs="text",
# )

# iface.launch(debug=True)
