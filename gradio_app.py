
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
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
#OPENAI_API_KEY  =' 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'

def find_urls(text: str) -> List:
    """
    Extract URLs from a given text.

    This function looks for patterns starting with 'http://', 'https://', or 'www.'
    followed by any non-whitespace characters. It captures common URL formats
    but might not capture all possible URL variations.

    Args:
    - text (str): The input string from which URLs need to be extracted.

    Returns:
    - list: A list containing all the URLs found in the input text.
    """
    # Regular expression to match common URLs and ones starting with 'www.'
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.findall(text)

def website_loader(website: Union[str, list[str]]) -> List[langchain.schema.document.Document]:
    """
    Loads the specified website(s) into Document objects.

    This function initiates the WebBaseLoader with the provided website or list of websites,
    loads them, and returns the resulting Document objects.

    Parameters:
    - website (Union[str, list[str]]): A single website URL as a string or a list of website URLs to be loaded.

    Returns:
    - List[langchain.schema.document.Document]: A list of Document objects corresponding to the loaded website(s).
    """

    print("Loading website(s) into Documents...")
    documents = WebBaseLoader(web_path=website).load()
    print("Done loading website(s).")
    return documents

def split_text(documents: List) -> List[langchain.schema.document.Document]:
    """
    Splits the provided documents into chunks using RecursiveCharacterTextSplitter.

    This function takes a list of documents, splits each document into smaller chunks
    of a specified size with a specified overlap, and returns the chunks as a list of
    Document objects.

    Parameters:
    - documents (List): A list of Document objects to be split into chunks.

    Returns:
    - List[langchain.schema.document.Document]: A list of Document objects representing the chunks.

    Note:
    - The chunk size, overlap, and length function are set to 1000, 50, and len respectively. Adjust
      these values if necessary.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=50,
                                                   length_function=len
                                                   )
    chunks = text_splitter.transform_documents(documents)
    print("Done splitting documents.")
    return chunks

def get_document_embeddings(chunks: List) :
    """
    Generates and retrieves embeddings for the given document chunks using CacheBackedEmbeddings.

    This function initializes an embedder backed by a local cache and a core embeddings model
    from OpenAI. It then uses this embedder to generate embeddings for the given document chunks.

    Parameters:
    - chunks (List): A list of Document chunks for which embeddings are to be generated.

    Returns:
    - langchain.embeddings.cache.CacheBackedEmbeddings: An embedder which can be used to get
      embeddings for the document chunks.
    """
    print("Creating embedder...")
    
    embedder= OpenAIEmbeddings()
    # embedder = CacheBackedEmbeddings.from_bytes_store(
    #     core_embeddings_model,
    #     store,
    #     namespace=core_embeddings_model.model
    # )
    print("Done creating embedder")
    return embedder

def create_vector_store(chunks: List[langchain.schema.document.Document],
                        embedder):
    """
    Creates a FAISS vector store from the given document chunks using the provided embedder.

    This function uses the provided embedder to transform the document chunks into vectors
    and then stores them in a FAISS vector store.

    Parameters:
    - chunks (List[langchain.schema.document.Document]): A list of Document chunks to be vectorized.
    - embedder (langchain.embeddings.cache.CacheBackedEmbeddings): An embedder used to generate embeddings
      for the document chunks.

    Returns:
    - langchain.vectorstores.faiss.FAISS: A FAISS vector store containing the vectors of the document chunks.
    """
    print("Creating vectorstore...")
    #vectorstore = FAISS.from_documents(chunks, embedder)
    #return vectorstore

    db = lancedb.connect('/tmp/lancedb')
    table = db.create_table("pdf_sear1ch", data=[
        {"vector": embedder.embed_query("Hello World"), "text": "Hello World", "id": "1"}
    ], mode="overwrite")
    vectorstore = LanceDB.from_documents(chunks, embedder, connection=table)
    return vectorstore



def create_retriever(vectorstore: langchain.vectorstores) :
    """
    Creates a retriever for the provided FAISS vector store.

    This function initializes a retriever for the given vector store, allowing for efficient
    querying and retrieval of similar vectors/documents from the vector store.

    Parameters:
    - vectorstore (langchain.vectorstores): A FAISS vector store containing vectors of document chunks.

    Returns:
    - langchain.vectorstores.base.VectorStoreRetriever: A retriever object that can be used to query
      and retrieve similar vectors/documents from the vector store.

    """
    print("Creating vectorstore retriever...")
    retriever = vectorstore.as_retriever()
    return retriever

def embed_user_query(query: str) -> List[float]:
    """
    Embeds the provided user query using the OpenAIEmbeddings model.

    This function takes a user query as input and transforms it into a vector representation
    using the OpenAIEmbeddings model.

    Parameters:
    - query (str): The user query to be embedded.

    Returns:
    - List[float]: A list of floats representing the embedded vector of the user query.
    """
    core_embeddings_model = OpenAIEmbeddings()
    embedded_query = core_embeddings_model.embed_query(query)
    return embedded_query

def similarity_search(vectorstore: langchain.vectorstores,
                      embedded_query: List[float]) -> List[langchain.schema.document.Document]:
    """
    Performs a similarity search on the provided FAISS vector store using an embedded query.

    This function takes an embedded query and searches the FAISS vector store for the most
    similar vectors/documents based on the embedded query.

    Parameters:
    - vectorstore (langchain.vectorstores): A FAISS vector store containing vectors of document chunks.
    - embedded_query (List[float]): A list of floats representing the embedded vector of the user query.

    Returns:
    - List[langchain.schema.document.Document]: A list of Document objects that are the most similar to
      the embedded query.

    Note:
    - The function currently retrieves the top 4 most similar documents (k=4). Adjust the value of 'k'
      if a different number of results is desired.
    """
    response = vectorstore.similarity_search_by_vector(embedded_query, k=4)
    return response


def create_chatbot(retriever: langchain.vectorstores) -> langchain.chains.conversational_retrieval:
    """
    Initializes and returns a conversational chatbot using the provided retriever and the OpenAI model.

    This function sets up a chatbot based on the ConversationalRetrievalChain from LangChain,
    which leverages the OpenAI model for conversational interactions and uses the given retriever
    for document retrieval.

    Parameters:
    - retriever (langchain.vectorstores): A retriever object used for document retrieval based on similarity searches.

    Returns:
    - langchain.chains.conversational_retrieval: A ConversationalRetrievalChain instance which acts as the chatbot.

    Note:

    - The conversation history is stored in the 'chat_history' memory key and is used for context in
      subsequent interactions.
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo")

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
    """
    Interacts with the chatbot using the provided input and returns its response.

    This function takes a user input, passes it to the chatbot for processing,
    and retrieves the chatbot's response.

    Parameters:
    - input (str): The user's input/question to the chatbot.

    Returns:
    - str: The chatbot's response to the user's input.

    """
    return conversation_chain.run(input)

text = """
I need you to go to the following URLs and get information about them
https://medium.com/cometheartbeat/how-to-prevent-yourself-from-wasting-time-on-your-data-science-project-728db69a4afc
and this one https://medium.com/mlearning-ai/langchain-is-the-past-here-is-the-future-of-llm-based-apps-46663f532c19
"""


# This chatbot_instance will be initialized once a URL is provided.
chatbot_instance = None

def respond(message, chat_history):
    global chatbot_instance
    urls = find_urls(message)
    # If the chatbot is not yet initialized and we have URLs, initialize it
    if not chatbot_instance and urls:
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
