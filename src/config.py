from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

class Config:
    vectorstore = Chroma(
        collection_name="mm_rag_doc_gpt",
        embedding_function=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        persist_directory="./chroma_store" 
    )