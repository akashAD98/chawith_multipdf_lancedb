

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
import os
from langchain.chains import RetrievalQA
import lancedb
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import LanceDB


pdf_folder_path = '/content/pdf'


from langchain.document_loaders import PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader(pdf_folder_path)
data = loader.load()

#EMBEDDING
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

#VECTORDB
db = lancedb.connect('/tmp/lancedb')
table = db.create_table("pdf_search", data=[
    {"vector": embeddings.embed_query("Hello World"), "text": "Hello World", "id": "1"}
], mode="overwrite")
docsearch = LanceDB.from_documents(data, embeddings, connection=table)


retriever = docsearch.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm = OpenAI(openai_api_key=openai.api_key),
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)


#USER INPUT
query = "what is resnet ?"
qa.run(query)
