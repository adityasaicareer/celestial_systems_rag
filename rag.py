from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings

import pprint

filepath="../data/example.pdf"

loader=PyPDFLoader(filepath)
print(loader)

docs=loader.load()

pprint.pp(docs[0].metadata)

for i in docs:
  pprint.pp(i.metadata)

""" we use the RecursiveCharacterTextSplitter to maintain the context and paragraphs intact"""

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)



chunks=text_splitter.split_documents(docs)

for idx,chunk in enumerate(chunks):
  chunk.metadata["chunk_id"]=idx


embedings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts=[chunk.page_content for chunk in chunks]
metadata=[chunk.metadata for chunk in chunks]
ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks]

vectors=embedings.embed_documents(texts)

client=chromadb.Client(Settings(persist_directory="./vectordb/chroma"))
collection=client.create_collection(name="my_collection")

vectors=embedings.embed_documents(texts)

collection.upsert(documents=texts,embeddings=vectors,metadatas=metadata,ids=ids)


print(f"Numebr of Inserted were : {collection.count()}")

query="How does top management demonstrate leadership and commitment to the ISMS??"

query_vector=embedings.embed_query(query)

results=collection.query(
  query_embeddings=[query_vector],
  n_results=5
)


for i in results['documents']:
  print(i)

print(len(results["documents"]))
print(results)
for i in results:
  print(i)
print(f"Score of the results {results['distances'][0][0]}")

