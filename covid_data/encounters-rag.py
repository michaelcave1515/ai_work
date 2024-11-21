from swarm import Swarm, Agent, Result
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd
import chromadb
import os

# Initialize Swarm client
client = Swarm()

# Set up vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def load_data():
    """Load and index the encounters data"""
    loader = CSVLoader(file_path="snippet-encounters.csv")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    vector_store.add_documents(texts)
    return "Data loaded and indexed"

def query_data(query: str):
    """Query the vector store and return relevant contexts"""
    results = vector_store.similarity_search(query, k=3)
    contexts = [doc.page_content for doc in results]
    return "\n\n".join(contexts)

def analyze_encounters(context_variables, query: str):
    """Analyze encounters data based on the query"""
    contexts = query_data(query)
    prompt = f"""Analyze the following encounters data to answer this query: {query}
    
    Relevant data:
    {contexts}
    
    Provide a clear, data-driven analysis."""
    
    return Result(
        value=prompt,
        agent=analysis_agent
    )

# Define agents
triage_agent = Agent(
    name="Triage Agent",
    instructions="""You are a data triage specialist. Your role is to:
    1. Understand user queries about border encounters
    2. Use the analyze_encounters function to get relevant data
    3. Hand off analysis to the analysis agent""",
    model="llama2:3b",
    functions=[analyze_encounters]
)

analysis_agent = Agent(
    name="Analysis Agent",
    instructions="""You are a border encounters data analyst. Your role is to:
    1. Review the provided data context
    2. Provide clear, factual analysis
    3. Focus on specific numbers and trends""",
    model="llama2:3b"
)

def main():
    # Load initial data
    load_data()
    
    # Example usage
    response = client.run(
        agent=triage_agent,
        messages=[{
            "role": "user", 
            "content": "What are the trends in encounters at the Boston Field Office?"
        }]
    )
    
    print(response.messages[-1]["content"])

if __name__ == "__main__":
    main()
