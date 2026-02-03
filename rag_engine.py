
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app_config import LLM_MODEL, EMBEDDING_MODEL, PERSIST_DIRECTORY

def get_rag_chain(api_key, db_path=None):
    """
    OPTIMIZED RAG Engine with improved prompt and retrieval settings.
    """
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 2. Vector DB
    from app_config import get_active_db_path
    import chromadb
    active_path = db_path if db_path else get_active_db_path()
    
    # If no path available yet, return a dummy or wait
    if not active_path or not os.path.exists(active_path):
        class DummyChain:
            def invoke(self, input_dict):
                return {"answer": "The Knowledge Base is not yet initialized. Please upload a file."}
        return DummyChain()

    # Use the same client approach as ingest.py for consistency and stability
    client = chromadb.PersistentClient(
        path=active_path,
        settings=chromadb.config.Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
    )
    
    vectorstore = Chroma(
        client=client,
        collection_name="employee_kb",
        embedding_function=embeddings,
    )

    # IMPROVED: Increase retrieval count for better coverage
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # 3. LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=LLM_MODEL,
        temperature=0
    )

    # 4. IMPROVED: More Detailed Prompt
    system_prompt = (
        "You are an expert Employee Data Analyst with access to a company database.\n\n"
        "INSTRUCTIONS:\n"
        "1. Answer questions using ONLY the provided context below\n"
        "2. Be specific with numbers, names, and details\n"
        "3. If asked for counts/totals, calculate accurately from the data\n"
        "4. If asked about specific people/projects, search the context carefully\n"
        "5. If the answer is not in the context, say 'I don't have that information in the database'\n"
        "6. Format answers clearly with bullet points or numbers when listing multiple items\n\n"
        "CONTEXT FROM DATABASE:\n{context}\n\n"
        "Remember: Only use information from the context above. Be precise and factual."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 5. Chain Construction (LCEL)
    def format_docs(docs):
        if not docs:
            return "No relevant data found."
        # Number the context for better LLM comprehension
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"[{i}] {doc.page_content}")
        return "\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return wrapper
    class WrappedChain:
        def invoke(self, input_dict):
            query = input_dict.get("input", "")
            result = rag_chain.invoke(query)
            return {"answer": result}
            
    return WrappedChain()
