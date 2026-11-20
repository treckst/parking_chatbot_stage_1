from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from pathlib import Path

llm = ChatOpenAI(model='gpt-6-luna')

response_llm = ChatOpenAI(model='gpt-6.2-sol')

embedding = OpenAIEmbeddings(model='text-embedding-3-small')
def get_vector(path: Path):
    '''path without chroma_db'''
    return Chroma(
    persist_directory = path / 'chroma_db',
    collection_name="parking_documents",
    embedding_function=embedding
)