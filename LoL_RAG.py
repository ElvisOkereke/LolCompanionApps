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


#--- STEP 1: ENHANCED DATA STRUCTURE ---

def create_matchup_dataset():
    """Create a comprehensive matchup dataset that captures lane interactions"""
    
    # Define sample matchup data - in a real implementation, this would be much more extensive
    matchups = [
        {
            "champion1": "Darius",
            "champion2": "Garen",
            "lane": "top",
            "difficulty": 3,  # Scale of 1-10, where 10 is extremely difficult
            "winner_favored": "Darius",
            "win_rate": 56.4,
            "matchup_details": {
                "early_game": "Darius wins early trades due to his passive. Garen should avoid extended fights.",
                "mid_game": "Darius continues to have the advantage if even. Garen needs jungle assistance.",
                "late_game": "Darius falls off slightly, but still has the edge in 1v1 scenarios.",
                "team_fights": "Garen can be more useful in team fights due to his tankiness and single-target execution."
            },
            "champion1_strategy": "Focus on landing outer Q, stack passive, and force extended trades. Zone Garen from CS.",
            "champion2_strategy": "Use Q to trade quickly, then back off. Max W second for damage mitigation. Build armor early.",
            "key_abilities": {
                "champion1": "Apprehend (E) to prevent escape, Decimate (Q) for damage and healing",
                "champion2": "Decisive Strike (Q) for short trades, Courage (W) to mitigate Darius's burst"
            },
            "itemization": {
                "champion1": "Early Phage into Black Cleaver. Consider Sterak's Gage for survivability.",
                "champion2": "Rush Bramble Vest, Plated Steelcaps, then Stridebreaker or Trinity Force."
            }
        },
        {
            "champion1": "Ahri",
            "champion2": "Zed",
            "lane": "mid",
            "difficulty": 7,
            "winner_favored": "Zed",
            "win_rate": 52.1,
            "matchup_details": {
                "early_game": "Even matchup pre-6. Ahri can harass with auto attacks.",
                "mid_game": "Zed has kill pressure after 6. Ahri must play defensively.",
                "late_game": "Ahri scales better for team fights, Zed better for picks.",
                "team_fights": "Ahri offers more utility and AoE damage in team fights."
            },
            "champion1_strategy": "Save charm for when Zed ults. Build Zhonya's second item. Push and roam.",
            "champion2_strategy": "Bait out Ahri's charm before all-in. Track her ultimate cooldown. Roam when pushed in.",
            "key_abilities": {
                "champion1": "Charm (E) to disrupt Zed's combo, Spirit Rush (R) to dodge Death Mark",
                "champion2": "Living Shadow (W) to poke, Death Mark (R) for all-in potential"
            },
            "itemization": {
                "champion1": "Everfrost for additional CC, Zhonya's Hourglass to counter ultimate",
                "champion2": "Serrated Dirk early, Eclipse or Duskblade, then Edge of Night"
            }
        },
        {
            "champion1": "Jinx",
            "champion2": "Lucian",
            "lane": "bot",
            "difficulty": 8,
            "winner_favored": "Lucian",
            "win_rate": 54.3,
            "matchup_details": {
                "early_game": "Lucian dominates with early aggression. Jinx must surrender CS.",
                "mid_game": "Still Lucian favored, but Jinx starts scaling with first item.",
                "late_game": "Jinx outscales significantly in team fights.",
                "team_fights": "Jinx provides much stronger team fighting with AoE rockets."
            },
            "champion1_strategy": "Farm safely with rockets. Wait for ganks. Scale into late game.",
            "champion2_strategy": "Freeze lane and zone Jinx. Punish when she approaches for CS. End game early.",
            "key_abilities": {
                "champion1": "Fishbones (Q rocket form) for safe farming, Flame Chompers (E) for self-peel",
                "champion2": "Lightslinger (passive) for burst damage, Relentless Pursuit (E) for all-ins"
            },
            "itemization": {
                "champion1": "Kraken Slayer, Runaan's Hurricane, Infinity Edge",
                "champion2": "Kraken Slayer, Essence Reaver, Navori Quickblades"
            }
        },
        {
            "champion1": "Lee Sin",
            "champion2": "Elise",
            "lane": "jungle",
            "difficulty": 6,
            "winner_favored": "Elise",
            "win_rate": 51.2,
            "matchup_details": {
                "early_game": "Elise has stronger early dueling and ganking potential.",
                "mid_game": "Even matchup. Both excel at picks and skirmishes.",
                "late_game": "Lee Sin scales slightly better with more utility.",
                "team_fights": "Lee Sin offers better team fight presence with kick potential."
            },
            "champion1_strategy": "Focus on counter-ganking and vision control. Invade when Elise's cooldowns are down.",
            "champion2_strategy": "Aggressive early ganks. Counter-jungle when Lee is spotted on opposite side.",
            "key_abilities": {
                "champion1": "Sonic Wave/Resonating Strike (Q) for mobility, Dragon's Rage (R) for playmaking",
                "champion2": "Cocoon (E) for ganks, Rappel (E) to dodge Lee Sin's Q"
            },
            "itemization": {
                "champion1": "Goredrinker, Black Cleaver, Sterak's Gage",
                "champion2": "Hextech Rocketbelt, Zhonya's Hourglass, Void Staff"
            }
        },
        {
            "champion1": "Leona",
            "champion2": "Morgana",
            "lane": "support",
            "difficulty": 9,
            "winner_favored": "Morgana",
            "win_rate": 58.7,
            "matchup_details": {
                "early_game": "Morgana counters Leona's engage with Black Shield.",
                "mid_game": "Morgana continues to deny Leona's playmaking potential.",
                "late_game": "Morgana scales better with utility, Leona still offers better engage.",
                "team_fights": "Both provide good CC, but Morgana can completely negate Leona's impact."
            },
            "champion1_strategy": "Bait out Black Shield before engaging. Consider roaming mid.",
            "champion2_strategy": "Save Black Shield for Leona's engage. Poke with W when safe.",
            "key_abilities": {
                "champion1": "Zenith Blade (E) for engage, Solar Flare (R) for AoE stun",
                "champion2": "Black Shield (E) to counter CC, Dark Binding (Q) for picks"
            },
            "itemization": {
                "champion1": "Locket of the Iron Solari, Thornmail, Zeke's Convergence",
                "champion2": "Imperial Mandate, Zhonya's Hourglass, Redemption"
            }
        }
    ]
    
    return matchups

def create_bidirectional_matchups(matchups):
    """
    Create a bidirectional dataset that views each matchup from both perspectives
    """
    bidirectional_matchups = []
    
    for matchup in matchups:
        # Original direction
        bidirectional_matchups.append(matchup)
        
        # Reversed direction
        reversed_matchup = {
            "champion1": matchup["champion2"],
            "champion2": matchup["champion1"],
            "lane": matchup["lane"],
            "difficulty": 10 - matchup["difficulty"],  # Invert difficulty
            "winner_favored": matchup["champion2"] if matchup["winner_favored"] == matchup["champion1"] else matchup["champion1"],
            "win_rate": 100 - matchup["win_rate"],
            "matchup_details": {
                "early_game": matchup["matchup_details"]["early_game"],  # Reinterpret for reversed perspective
                "mid_game": matchup["matchup_details"]["mid_game"],
                "late_game": matchup["matchup_details"]["late_game"],
                "team_fights": matchup["matchup_details"]["team_fights"]
            },
            "champion1_strategy": matchup["champion2_strategy"],
            "champion2_strategy": matchup["champion1_strategy"],
            "key_abilities": {
                "champion1": matchup["key_abilities"]["champion2"],
                "champion2": matchup["key_abilities"]["champion1"]
            },
            "itemization": {
                "champion1": matchup["itemization"]["champion2"],
                "champion2": matchup["itemization"]["champion1"]
            }
        }
        
        bidirectional_matchups.append(reversed_matchup)
    
    # Convert to DataFrame
    bidir_df = pd.DataFrame(bidirectional_matchups)
    
    # Save to CSV
    bidir_df.to_csv("lol_bidirectional_matchups.csv", index=False)
    
    print("Bidirectional matchup dataset created and saved!")
    return bidir_df

def create_document_chunks(matchups):
    """
    Create text chunks optimized for RAG retrieval
    """
    documents = []
    metadata = []
    
    for matchup in matchups:
        # Create different chunks for different aspects of the matchup
        
        # Basic matchup information
        basic_info = f"""
        Matchup: {matchup['champion1']} vs {matchup['champion2']} in {matchup['lane']} lane
        Difficulty: {matchup['difficulty']}/10 for {matchup['champion1']}
        Favored Champion: {matchup['winner_favored']} (Win Rate: {matchup['win_rate']}%)
        """
        
        # Phase-specific information
        phases_info = f"""
        Matchup Phases for {matchup['champion1']} vs {matchup['champion2']} in {matchup['lane']}:
        Early Game: {matchup['matchup_details']['early_game']}
        Mid Game: {matchup['matchup_details']['mid_game']}
        Late Game: {matchup['matchup_details']['late_game']}
        Team Fights: {matchup['matchup_details']['team_fights']}
        """
        
        # Strategy information for first champion
        champ1_strategy = f"""
        Strategy for {matchup['champion1']} against {matchup['champion2']} in {matchup['lane']}:
        Overall Strategy: {matchup['champion1_strategy']}
        Key Abilities: {matchup['key_abilities']['champion1']}
        Recommended Items: {matchup['itemization']['champion1']}
        """
        
        # Strategy information for second champion
        champ2_strategy = f"""
        Strategy for {matchup['champion2']} against {matchup['champion1']} in {matchup['lane']}:
        Overall Strategy: {matchup['champion2_strategy']}
        Key Abilities: {matchup['key_abilities']['champion2']}
        Recommended Items: {matchup['itemization']['champion2']}
        """
        
        # Add chunks to documents with corresponding metadata
        chunks = [
            basic_info.strip(),
            phases_info.strip(),
            champ1_strategy.strip(),
            champ2_strategy.strip()
        ]
        
        documents.extend(chunks)
        
        # Add metadata for each chunk
        for _ in range(len(chunks)):
            metadata.append({
                "champion1": matchup["champion1"],
                "champion2": matchup["champion2"],
                "lane": matchup["lane"]
            })
    
    return documents, metadata

# --- STEP 2: VECTOR DATABASE SETUP ---

class SimpleVectorDB:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.texts = []
        self.metadata = []
        
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the vector database"""
        if not documents:
            return
            
        self.texts.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        
        # Create embeddings
        embeddings = self.model.encode(documents)
        
        # Initialize or update FAISS index
        dimension = embeddings.shape[1]
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
            
        # Add vectors to index
        self.index.add(np.array(embeddings).astype('float32'))
        
    def search(self, query: str, top_k: int = 5):
        """Search for similar documents"""
        query_embedding = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                result = {
                    "text": self.texts[idx],
                    "score": float(distances[0][i])
                }
                
                # Add metadata if available
                if idx < len(self.metadata):
                    result["metadata"] = self.metadata[idx]
                    
                results.append(result)
                
        return results

# --- STEP 3: RETRIEVAL SYSTEM ---

def extract_champion_names(query):
    """
    Extract champion names from user query to improve retrieval
    A more robust implementation could use NER or a dictionary lookup
    """
    # This is a simplified version - in a real implementation, you'd have a complete
    # list of champions and use NLP techniques for better extraction
    common_champions = [
        "Darius", "Garen", "Ahri", "Zed", "Jinx", "Lucian", "Lee Sin", "Elise",
        "Leona", "Morgana", "Yasuo", "Lux", "Thresh", "Ezreal", "Jhin"
    ]
    
    found_champions = []
    
    for champion in common_champions:
        if champion.lower() in query.lower():
            found_champions.append(champion)
    
    return found_champions

def query_matchup(vector_db, query, top_k=5):
    """Query the vector database for matchup information with enhanced logic"""
    
    # Extract champion names to improve the query
    champions = extract_champion_names(query)
    
    # Perform the basic search
    results = vector_db.search(query, top_k=top_k)
    
    # If champions were mentioned, boost the relevance of documents about those champions
    if champions and len(champions) > 0:
        # This is a simple boosting mechanism
        # A more sophisticated approach would re-rank based on champion mentions
        boosted_results = []
        regular_results = []
        
        for result in results:
            is_boosted = False
            
            if "metadata" in result:
                for champion in champions:
                    if (result["metadata"].get("champion1") == champion or 
                        result["metadata"].get("champion2") == champion):
                        boosted_results.append(result)
                        is_boosted = True
                        break
            
            if not is_boosted:
                regular_results.append(result)
        
        # Combine boosted and regular results, with boosted ones first
        combined_results = boosted_results + regular_results
        
        # Trim to the original top_k size
        return combined_results[:top_k]
    
    return results

# --- STEP 4: LLM INTEGRATION ---

def generate_response(query, retrieved_contexts):
    """Generate response using OpenAI API with enhanced context"""
    
    # Format retrieved contexts
    context_str = ""
    for i, res in enumerate(retrieved_contexts):
        context_str += f"Context {i+1}:\n{res['text']}\n\n"
    
    # Extract champion names for more targeted response
    champions = extract_champion_names(query)
    champion_context = ""
    
    if champions and len(champions) > 0:
        champion_context = f"The user is asking about the following champion(s): {', '.join(champions)}. "
    
    prompt = f"""You are a League of Legends expert specializing in champion matchups.
    {champion_context}Answer the following question like you are talking to a person, based on the following champion information:
    
    {context_str}
    
    If the information doesn't contain what you need to answer the question,
    provide general advice but make it clear you're giving general guidance.
    
    Be specific about game phases (early, mid, late) and provide concrete strategies.
    When discussing matchups, mention both sides of the interaction.

    Do not mention the context/source of the information or that you are an AI.

    Answer the question in paragraph form in a friendly and engaging manner.
    
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
        max_tokens=4096,
        temperature=0.7,
    )
    
    
    return(response.choices[0].message.content)

    #return response.choices[0].message["content"]

# --- STEP 5: USER INTERFACE ---

def create_ui(vector_db):
    """Create a simple Gradio interface"""
    
    def answer_query(query, additional_context=""):
        """Process user query and return response"""
        
        # Combine query with additional context if provided
        full_query = query
        if additional_context:
            full_query = f"{query} Additional context: {additional_context}"
        
        # Get relevant contexts
        results = query_matchup(vector_db, full_query)
        
        # Generate response
        response = generate_response(full_query, results)
        
        # For debugging: return contexts that were used
        context_info = "\n\n---DEBUG: Retrieved Contexts---\n"
        for i, res in enumerate(results):
            context_info += f"Context {i+1} (score: {res['score']:.2f}):\n{res['text'][:100]}...\n"
        
        # In production, you would remove the debug info
        return response + context_info if debug_mode else response
    
    # Set debug mode (set to False for production)
    debug_mode = True
    
    # Create Gradio interface with improved layout
    with gr.Blocks(title="League of Legends Matchup Assistant") as interface:
        gr.Markdown("# League of Legends Matchup Assistant")
        gr.Markdown("Ask questions about champion matchups, counters, and strategies!")
        
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    lines=2, 
                    placeholder="Ask about a League of Legends matchup (e.g., 'How does Darius vs Garen matchup work in top lane?')",
                    label="Your Question"
                )
                
            with gr.Column(scale=1):
                submit_btn = gr.Button("Get Advice", variant="primary")
        
        with gr.Accordion("Advanced Options", open=False):
            context_input = gr.Textbox(
                lines=2,
                placeholder="Optional: Add more context about your rank, playstyle, etc.",
                label="Additional Context"
            )
        
        output = gr.Textbox(label="Matchup Advice", lines=10)
        
        submit_btn.click(
            fn=answer_query,
            inputs=[query_input, context_input],
            outputs=output
        )
        
        # Add example queries for users to try
        gr.Examples(
            examples=[
                ["How do I play Darius vs Garen top lane?"],
                ["Is Ahri good against Zed in mid?"],
                ["What items should I build as Jinx against Lucian?"],
                ["What's the best strategy for Zed vs Ahri early game?"],
            ],
            inputs=query_input
        )
    
    return interface

# --- MAIN EXECUTION ---

def main():
    """Main function to run the application"""
    print("Initializing League of Legends Matchup RAG bot...")
    
    # Create dataset
    print("Creating matchup dataset...")
    matchups = create_matchup_dataset()

    bidrect = create_bidirectional_matchups(matchups)
    
    # Create document chunks for embedding
    print("Processing data into retrievable chunks...")
    documents, metadata = create_document_chunks(matchups)
    #bi_documents, bi_metadata = create_document_chunks(bidrect)
    
    # Initialize and populate vector database
    print("Setting up vector database...")
    vector_db = SimpleVectorDB()
    vector_db.add_documents(documents, metadata)
    #vector_db.add_documents(bi_documents, bi_metadata)
    print(f"Added {len(documents)} document chunks to vector database")
    
    # Create and launch UI
    print("Creating user interface...")
    ui = create_ui(vector_db)
    print("Launching interface. This may take a moment...")
    ui.launch(share=True)  # Set share=False in production

if __name__ == "__main__":
    main()