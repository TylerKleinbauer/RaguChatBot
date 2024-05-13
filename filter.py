import os

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever, MergerRetriever
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever

from basic_chain import get_model
from ensemble import ensemble_retriever_from_docs
from remote_loader import load_web_page
from vector_store import create_vector_db

from dotenv import load_dotenv


def create_retriever(texts):
    # Create openAI embeddings
    # Can be customized to use other forms of embeddings
    openai_api_key = os.environ["OPENAI_API_KEY"]
    openAI_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    
    # Creating a vector store with the openAI embeddings
    openAI_vs = create_vector_db(texts, collection_name="dense", embeddings=openAI_embeddings)   
    
    # Applying filters and comrpessors to the openAI embeddings to reduce the size of the context
    emb_filter = EmbeddingsRedundantFilter(embeddings=openAI_embeddings)
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[emb_filter, reordering])
    
    # Two different retrievers
    openAI_vs_retriever = openAI_vs.as_retriever()
    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])

    lotr = MergerRetriever(retrievers=[openAI_vs_retriever, bm25_retriever])

    compression_retriever_reordered = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr, search_kwargs={"k": 5, "include_metadata": True}
    )
    return compression_retriever_reordered