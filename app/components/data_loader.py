from app.components.pdf_loader import pdf_loader , pdf_splitter
from app.components.vector_store import create_vectordb , load_db
from app.components.llm import llm_chain

import os
from app.common.logger import get_logger
from app.common.exption import CustomException

logger = get_logger(__name__)

def process_store_pdf():
    try:
        logger.info("test llm...")
        
        vector_store = load_db()
        rag_model = llm_chain(vector_store)
        res = rag_model.invoke("who is marwn ragab?")

        print(res)
        # document = pdf_loader()
        # chunks = pdf_splitter(document)
        # create_vectordb(chunks)

        logger.info("PDF processing and vector store creation completed successfully.")
    except Exception as e:
        raise CustomException("Error occurred during PDF processing and vector store creation.", e)


if __name__ == "__main__":
    process_store_pdf()