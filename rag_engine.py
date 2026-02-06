
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app_config import LLM_MODEL, EMBEDDING_MODEL, PERSIST_DIRECTORY

def get_rag_chain(api_key, db_path=None, username=None):
    """
    OPTIMIZED RAG Engine with improved prompt and retrieval settings.
    """
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 2. Vector DB
    from app_config import get_active_db_path
    import chromadb
    active_path = db_path if db_path else get_active_db_path(username)
    
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

    # Updated prompt to include chat history
    system_prompt = (
        "You are a friendly and professional Employee Data Assistant. "
        "Your role is to help users understand their workforce data through clear, accurate, and conversational responses.\n\n"
        
        "**Your Communication Style:**\n"
        "- Be warm, polite, and supportive in your tone\n"
        "- Speak naturally like a helpful colleague, not a robot\n"
        "- Use simple, direct language ( \"I see that...\", \"Looking at this...\")\n"
        "- Provide context when helpful, but stay focused on the answer\n"
        "- Acknowledge the user's question naturally\n"
        "- Offer follow-up suggestions when appropriate\n"
        "- **IMPORTANT**: Remember the conversation history and answer follow-up questions based on previous context\n\n"
        
        "**Accuracy Standards:**\n"
        "- Base all answers strictly on the provided database context below\n"
        "- **CRITICAL**: Perform proper numerical calculations for time, hours, and dates with precision\n"
        "- When calculating time differences, hours worked, or date ranges, use exact arithmetic\n"
        "- Double-check all numerical results before presenting them\n"
        "- Perform calculations carefully and verify results\n"
        "- Calculate answers internally but don't show calculations\n"
        "- If data is unclear or missing, politely acknowledge it\n"
        "- Present numbers and facts with appropriate context\n\n"
        
        "**Response Format:**\n"
        "- Start with a direct answer to the question\n"
        "- Add brief context or explanation if it helps understanding\n"
        "- If user asks for a list, present it clearly\n"
        "- Use natural, flowing sentences rather than bullet points\n"
        "- **AVOID** robotic phrases like:\n"
        "  ❌ 'based on the context'\n"
        "  ❌ 'according to the data'\n"
        "  ❌ 'I found this information consistently across all the entries'\n"
        "  ❌ 'in the database'\n"
        "  ❌ 'the dataset shows'\n\n"
        
        "**Examples of Good vs Bad Responses:**\n"
        "❌ Robotic: \"Total hours: 320\"\n"
        "✅ Natural: \"The total hours worked across all projects is 320 hours.\"\n\n"
        
        "❌ Robotic: \"There are 12 employees in the dataset\"\n"
        "✅ Natural: \"There are 12 employees. Would you like me to break this down by department or role?\"\n\n"
        
        "❌ Robotic: \"I found this information consistently across all the entries in the database\"\n"
        "✅ Natural: \"That would be Aiyub Munshi. Would you like to know more about the project?\"\n\n"
        
        "**Conversation History:**\n"
        "{chat_history}\n\n"
        
        "**Database Context:**\n"
        "{context}\n\n"
        
        "Remember: Be accurate, be helpful, be human. Answer like you're talking to a colleague, not reading from a database report."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 5. Chain Construction (LCEL) with chat history support
    def format_docs(docs):
        if not docs:
            return "No relevant data found."
        # Number the context for better LLM comprehension
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"[{i}] {doc.page_content}")
        return "\n".join(formatted)

    # Helper function to extract query from input dict
    def get_query(input_data):
        if isinstance(input_data, dict):
            return input_data.get("input", "")
        return str(input_data)

    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(get_query(x))), 
            "input": lambda x: get_query(x),
            "chat_history": lambda x: x.get("chat_history", "") if isinstance(x, dict) else ""
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return wrapper that accepts chat_history
    class WrappedChain:
        def invoke(self, input_dict):
            query = input_dict.get("input", "")
            chat_history = input_dict.get("chat_history", [])
            
            # Format chat history as a string
            history_str = ""
            if chat_history:
                history_lines = []
                # Get last 5 exchanges to keep context manageable
                recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
                for msg in recent_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_lines.append(f"{role}: {msg['content']}")
                history_str = "\n".join(history_lines)
            
            result = rag_chain.invoke({"input": query, "chat_history": history_str})
            return {"answer": result}
            
    return WrappedChain()
