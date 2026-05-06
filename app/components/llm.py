from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.common.exption import CustomException
from app.common.logger import get_logger
from app.components.vector_store import load_db
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

logger = get_logger(__name__)

# Global chain variable (cached after first creation)
_chain = None


# Format documents for the prompt
def format_docs(docs: List[Document]) -> str:
    """Format documents for insertion into prompt"""
    if not docs or len(docs) == 0:
        return "No relevant documents found in the knowledge base."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content.strip()
        if content:  # Only add non-empty documents
            formatted.append(f"[Document {i}]\n{content}")
    
    if not formatted:
        return "No relevant documents found in the knowledge base."
    
    return "\n\n".join(formatted)


def create_llm():
    try:
        logger.info("Creating LLM model...")
        llm = ChatGroq(model="llama-3.1-8b-instant")
        logger.info("LLM model created successfully.")
        return llm
    except Exception as e:
        raise CustomException("Error occurred while creating LLM model.", e)


# def _handle_empty_context(query: str, llm) -> str:
#     """Handle case when no relevant documents are found"""
#     logger.warning(f"No relevant documents found for query: {query[:50]}...")
    
#     fallback_prompt = ChatPromptTemplate.from_template(
#         """You are a medical information assistant.
        
#         The user asked: {question}
        
#         However, I don't have relevant information about this topic in my knowledge base.
        
#         Please provide a brief, general response based on common medical knowledge,
#         and suggest that the user consult a healthcare professional for specific medical advice.
        
#         Keep your answer to 2-3 sentences maximum."""
#     )
    
#     chain = fallback_prompt | llm | StrOutputParser()
#     return chain.invoke({"question": query})


def _create_chain():
    """Internal function to create the RAG chain"""
    try:
        logger.info("Creating LLM chain....")
        
        # Load the vector store
        vectorstore = load_db()
        llm = create_llm()
        
        # Create retriever with MMR (Maximum Marginal Relevance) for better diversity
        # This helps avoid redundant documents and ensures more diverse results
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Changed from similarity to MMR
            search_kwargs={
                "k": 5,           # Number of documents to retrieve
                "fetch_k": 20,    # Fetch more candidates before filtering
                "lambda_mult": 0.5  # Balance between relevance and diversity
            }
        )
        
        # Create prompt
        simple_prompt = ChatPromptTemplate.from_template(
            """You are a precise medical information assistant based on medical encyclopedia.

            Your task is to answer the user's question using ONLY the provided context.
            
            Guidelines:
            - Use only information from the provided context/documents
            - If the context contains the answer, provide it clearly and concisely
            - If the context does NOT contain sufficient information, respond with: "I don't have detailed information about this in my knowledge base. Please consult a healthcare professional."
            - Keep the answer to 3 sentences maximum
            - Do NOT make up information or use outside knowledge
            - Be precise and factual
            
            Context from Medical Encyclopedia:
            {context}

            User Question:
            {question}

            Answer:"""
        )
        
        # Create chain
        llm_rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | simple_prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("LLM chain created successfully with MMR retrieval.")
        return llm_rag_chain
        
    except Exception as e:
        raise CustomException("Error occurred while creating LLM chain.", e)


def llm_chain(query: str) -> str:
    """
    Process a user query and return response from the RAG chain
    
    Args:
        query: User's medical question
        
    Returns:
        str: Response from the LLM
    """
    global _chain
    
    try:
        # Create chain if not already created (lazy loading)
        if _chain is None:
            _chain = _create_chain()
        
        logger.info(f"Processing query: {query[:50]}...")
        
        # Invoke the chain with the query
        response = _chain.invoke(query)
        
        logger.info("Query processed successfully.")
        return response
        
    except CustomException as ce:
        logger.error(f"Custom error in llm_chain: {str(ce)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in llm_chain: {str(e)}")
        raise CustomException("Error occurred while processing query.", e)