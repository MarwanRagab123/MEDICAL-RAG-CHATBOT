from langchain_community.vectorstores import FAISS

from app.config.config import DP_FAISS_PATH

from app.common.exption import CustomException
from app.common.logger import get_logger
import os
from app.components.embeddings import embedding_layer

embedding = embedding_layer()

loader = get_logger(__name__)
def create_vectordb(chunks):
    try:
        if not chunks:
            raise CustomException("No chunks Found!!")

        loader.info("Generating Vector Data base ...")

        

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding= embedding
        )

        loader.info("Save the Vector data base..")

        vectorstore.save_local(DP_FAISS_PATH)

        loader.info("vector data base successfully created..")

        return vectorstore
    except Exception as e:
        raise CustomException("Error occurred while creating vector database.", e)

def load_db():

    try:
        if not os.path.exists(DP_FAISS_PATH):
            raise CustomException("Data base not exist!!")
        loader.info("Waiting to Load database..........")

        return FAISS.load_local(
            DP_FAISS_PATH,
            embeddings= embedding,
            allow_dangerous_deserialization=True

        )
    except Exception as e:
        raise CustomException("Error occurred while loading vector database.", e)
    
def add_documents_to_db(new_chunks):
    try:
        if not new_chunks:
            raise CustomException("No new chunks to add to the database.")
        loader.info("Loading existing vector database...")
        vectorstore = load_db()

        loader.info("Adding new documents to the vector database...")
        vectorstore.add_documents(new_chunks)

        loader.info("Saving updated vector database...")
        vectorstore.save_local(DP_FAISS_PATH)

        loader.info("New documents successfully added to the vector database.")
    except Exception as e:
        raise CustomException("Error occurred while adding documents to the vector database.", e)