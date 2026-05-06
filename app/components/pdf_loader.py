import os


from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



from app.config.config import DATA_PATH , CHUNK_SIZE , CHUNK_OVERLAP
from app.common.logger import get_logger
from app.common.exption import CustomException


def pdf_loader():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data path {DATA_PATH} does not exist.")
        logger = get_logger(__name__)
        logger.info(f"Loading PDF documents from {DATA_PATH}...")
        Loader = DirectoryLoader(
            DATA_PATH,
            glob = "*.pdf",
            loader_cls = PyPDFLoader,
            show_progress = True 
        )

        documents = Loader.load()

        if not documents:
            logger.warning(f"No PDF documents found in {DATA_PATH}.")
        else:
            logger.info(f"Successfully loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        raise CustomException("Error occurred while loading PDF documents.", e)

def pdf_splitter(documents):
    try:
        logger = get_logger(__name__)
        logger.info("Splitting documents into chunks...")
        if not documents:
            raise CustomException("No document to split.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            separators = ["\n\n", "\n", " ", ""],
        ) 
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            logger.warning("No chunks were created from the documents.")
        else:
            logger.info(f"Successfully split documents into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        raise CustomException("Error occurred while splitting documents into chunks.", e)