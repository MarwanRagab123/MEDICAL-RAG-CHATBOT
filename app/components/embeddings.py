from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from app.common.logger import get_logger
from app.common.exption import CustomException

from app.config.config import EMBEDDING_REPO

def embedding_layer():
    try:
        logger = get_logger(__name__)
        logger.info(f"Embedding Layer Loading..")

        embedding = HuggingFaceBgeEmbeddings(model_name = EMBEDDING_REPO)
        logger.info(f"Success Install Embedding Layer..")
        return embedding
    
    except Exception as e:
        raise CustomException("Error occurred while install embedding layer.", e)