import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

import tiktoken

os.environ['OPENAI_API_KEY'] = ""

loader = TextLoader("davem.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("Testing the embedding model")

doc_vectors = embeddings.embed_documents([t.page_content for t in texts[:5]])

CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5433/linkedin"
COLLECTION_NAME = "linkedin"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

query = ("summarize this person in two sentences")

print(embeddings.embed_query(query))

