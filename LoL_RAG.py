"""
League of Legends Matchup RAG Chatbot - One-Day MVP Implementation
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import requests
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
from openai import OpenAI


# --- STEP 1: DATA COLLECTION ---

def create_sample_data():
    """Create sample matchup data for demonstration purposes"""
    
    matchups = [
        {
            "champion": "Darius",
            "role": "top",
            "counter": "Quinn",
            "counter_reason": "Quinn can kite Darius with her range and Vault ability to prevent him from stacking his passive.",
            "strong_against": "Garen",
            "strong_reason": "Darius out-trades Garen in extended fights and can prevent his passive healing with bleed.",
            "general_tips": "Darius excels in extended trades where he can stack his passive. His strongest power spike is at level 6.",
        },
        {
            "champion": "Ahri",
            "role": "mid",
            "counter": "Kassadin",
            "counter_reason": "Kassadin's magic shield reduces Ahri's burst and he outscales her.",
            "strong_against": "Lux",
            "strong_reason": "Ahri has more mobility to dodge Lux's skillshots and can engage with ultimate.",
            "general_tips": "Ahri is strongest when using charm to set up her damage combo. Roam to side lanes after pushing.",
        },
        {
            "champion": "Jinx",
            "role": "adc",
            "counter": "Draven",
            "counter_reason": "Draven's early game damage overwhelms Jinx before she can scale.",
            "strong_against": "Ezreal",
            "strong_reason": "Jinx can push Ezreal under tower and outscale him in team fights.",
            "general_tips": "Jinx is weak early but scales incredibly well. Farm safely until you have Runaan's Hurricane.",
        },
        # Add more sample data as needed
    ]
    
    # Create a dataframe
    df = pd.DataFrame(matchups)
    
    # Save to CSV
    df.to_csv("lol_matchups.csv", index=False)
    print("Sample data created and saved to lol_matchups.csv")
    
    return df

# --- STEP 2: VECTOR DATABASE SETUP ---

class SimpleVectorDB:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.texts = []
        
    def add_documents(self, documents: List[str]):
        """Add documents to the vector database"""
        if not documents:
            return
            
        self.texts.extend(documents)
        
        # Create embeddings
        embeddings = self.model.encode(documents)
        
        # Initialize or update FAISS index
        dimension = embeddings.shape[1]
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
            
        # Add vectors to index
        self.index.add(np.array(embeddings).astype('float32'))
        
    def search(self, query: str, top_k: int = 3):
        """Search for similar documents"""
        query_embedding = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(distances[0][i])
                })
                
        return results

def prepare_matchup_documents(df):
    """Convert dataframe to documents for vector storage"""
    documents = []
    metadata = []
    
    for _, row in df.iterrows():
        # Create different document chunks for different aspects of champion info
        counter_doc = f"Champion: {row['champion']} (Role: {row['role']}) is countered by {row['counter']}. {row['counter_reason']}"
        strong_doc = f"Champion: {row['champion']} (Role: {row['role']}) is strong against {row['strong_against']}. {row['strong_reason']}"
        tips_doc = f"Champion: {row['champion']} (Role: {row['role']}) - Tips: {row['general_tips']}"
        
        documents.extend([counter_doc, strong_doc, tips_doc])
        
        # Store metadata for each document
        for _ in range(3):  # One for each document type
            metadata.append({"champion": row['champion'], "role": row['role']})
    
    return documents, metadata

# --- STEP 3: RETRIEVAL SYSTEM ---

def query_matchup(vector_db, query, top_k=3):
    """Query the vector database for matchup information"""
    results = vector_db.search(query, top_k=top_k)
    return results

# --- STEP 4: LLM INTEGRATION ---

def generate_response(query, retrieved_contexts):
    """Generate response using OpenAI API"""
    
    # Format retrieved contexts
    context_str = "\n".join([res["text"] for res in retrieved_contexts])
    
    prompt = f"""You are a League of Legends expert specializing in champion matchups.
    Answer the user's question based on the following champion information:
    
    {context_str}
    
    If the information doesn't contain what you need to answer the question,
    provide general advice but make it clear you're giving general guidance.
    
    User Question: {query}
    """
    
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = "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"
    )

    response = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
        messages=[
            {"role": "system", "content": "You are a League of Legends matchup expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )
    
    
    return(response.choices[0].message.content)

    #return response.choices[0].message["content"]

# --- STEP 5: USER INTERFACE ---

def create_ui(vector_db):
    """Create a simple Gradio interface"""
    
    def answer_query(query):
        """Process user query and return response"""
        # Get relevant contexts
        results = query_matchup(vector_db, query)
        
        # Generate response
        response = generate_response(query, results)
        
        return response
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=answer_query,
        inputs=gr.Textbox(lines=2, placeholder="Ask about a League of Legends matchup..."),
        outputs="text",
        title="League of Legends Matchup Assistant",
        description="Ask questions about champion matchups, counters, and strategies!"
    )
    
    return interface

# --- MAIN EXECUTION ---

def main():
    """Main function to run the application"""
    # Create sample data (in a real app, you'd load external data)
    df = create_sample_data()
    
    # Prepare documents for vector database
    documents, metadata = prepare_matchup_documents(df)
    
    # Initialize and populate vector database
    vector_db = SimpleVectorDB()
    vector_db.add_documents(documents)
    
    # Create and launch UI
    ui = create_ui(vector_db)
    ui.launch(share=True)  # Set share=False in production

if __name__ == "__main__":
    main()