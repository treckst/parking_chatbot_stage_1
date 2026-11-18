from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_chroma import Chroma

load_dotenv()

from src.models_vector import embedding

vectordb_path = Path(__file__).parent.parent

def ingest_docs():
    loader = DirectoryLoader(
        path=str(Path(__file__).parent / 'pdfs'),
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    docs = loader.load()


    Chroma.from_documents(
        persist_directory=vectordb_path / 'data' / 'chroma_db',
        documents=docs,
        embedding=embedding,
        collection_name='parking_documents'
    )

if __name__ == '__main__':
    ingest_docs()