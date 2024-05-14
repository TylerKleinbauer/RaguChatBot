import os

from dotenv import load_dotenv
from basic_chain import basic_chain, get_model
from remote_loader import get_wiki_docs
from splitter import split_documents
from vector_store import create_vector_db
from rag_chain import make_rag_chain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

def main():
    load_dotenv()
    model = get_model("ChatGPT")
    docs = get_wiki_docs(query="Bertrand Russell", load_max_docs=5)
    texts = split_documents(docs)
    vs = create_vector_db(texts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professor who teaches philosophical concepts to beginners."),
        ("user", "{input}")
    ])
    # Besides similarly search, you can also use maximal marginal relevance (MMR) for selecting results.
    # retriever = vs.as_retriever(search_type="mmr")
    retriever = vs.as_retriever()

    output_parser = StrOutputParser()
    chain = basic_chain(model, prompt)
    base_chain = chain | output_parser
    rag_chain = make_rag_chain(model, retriever) | output_parser

    questions = [
        "What were the most important contributions of Bertrand Russell to philosophy?",
        "What was the first book Bertrand Russell published?",
        "What was most notable about \"An Essay on the Foundations of Geometry\"?",
    ]
    for q in questions:
        print("\n--- QUESTION: ", q)
        print("* BASE:\n", base_chain.invoke({"input": q}))
        print("* RAG:\n", rag_chain.invoke(q))