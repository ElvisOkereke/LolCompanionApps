import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
from openai import OpenAI
import PyPDF2
import re
from pathlib import Path
import logging
import shutil


from lolalytics_scraper import LolalyticsScraper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#--- STEP 1: PDF PARSER ---
class PDFParser:
    def __init__(self, pdf_directory: str = "matchup_pdfs"):
        self.pdf_directory = pdf_directory
        os.makedirs(pdf_directory, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    # else: # Optional: Log if a page has no extractable text
                    #     logger.debug(f"Page {page_num + 1} in {pdf_path} had no extractable text.")
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def _extract_section_content(self, text: str, start_keyword: str, end_keywords: List[str]) -> str:
        """Helper to extract content for a specific item within a larger section."""
        # Ensure start_keyword is escaped for regex if it contains special characters
        escaped_start_keyword = re.escape(start_keyword)
        
        # Pattern: StartKeyword\s*(.*?)(?=EndKeyword1|EndKeyword2|...|$)
        # The positive lookahead ensures we capture everything until the next keyword or end of text.
        # Adding \n? before end_keywords to handle cases where they might be on a new line.
        end_pattern = "|".join([r"\n?" + re.escape(kw) for kw in end_keywords]) if end_keywords else "$"
        
        pattern = rf"{escaped_start_keyword}\s*(.*?)(?=(?:{end_pattern}))"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        return match.group(1).strip() if match and match.group(1) else "Not found"


    def extract_matchup_info(self, matchup_text_block: str) -> Optional[Dict]:
        """
        Extract structured matchup information from a text block corresponding to ONE matchup.
        Assumes the new PDF format.
        """
        try:
            # 1. Basic Information
            # Matchup: Fiora vs Darius in top lane Difficulty: 5/10 Favored Champion: Fiora Win Rate: 51.94%
            basic_info_pattern = (
                r"Matchup:\s*(?P<champion1>[\w\s'.&-]+?)\s*vs\s*(?P<champion2>[\w\s'.&-]+?)\s*in\s*"
                r"(?P<lane>[\w\s]+?)\s*lane\s*Difficulty:\s*(?P<difficulty>\d+/\d+)\s*"
                r"Favored Champion:\s*(?P<favored>[\w\s'.&-]+?)\s*Win Rate:\s*(?P<win_rate>[\d.]+%?)"
            )
            basic_match = re.search(basic_info_pattern, matchup_text_block, re.IGNORECASE)

            if not basic_match:
                logger.warning("Could not parse 'Matchup:' line from PDF text block.")
                return None

            data = basic_match.groupdict()
            champion1 = data['champion1'].strip()
            champion2 = data['champion2'].strip()
            lane = data['lane'].strip().lower()
            difficulty_str = data['difficulty'].split('/')[0] # "5/10" -> "5"
            difficulty = int(difficulty_str) if difficulty_str.isdigit() else 5
            favored_champion = data['favored'].strip()
            win_rate_str = data['win_rate'].replace('%', '')
            win_rate = float(win_rate_str) if win_rate_str else 50.0
            
            # Determine winner_favored based on "Favored Champion" field directly
            # If that field somehow matched champion1 or champion2 exactly, fine.
            # Otherwise, it could be "Fiora" or "Aatrox" etc.
            winner_favored = favored_champion # Use the parsed favored champion

            # 2. Game Phases
            # Define keywords that mark the end of the "Game Phases" section content for each phase.
            # For "Early Game", it ends before "Mid Game:", "Late Game:", "Team Fights:", or a strategy section.
            # Strategy sections look like "Strategy for Fiora"
            game_phase_section_end_keywords = [
                f"Strategy for {champion1}", f"Strategy for {champion2}", "Basic Information" # End of current matchup block
            ]
            
            early_game = self._extract_section_content(matchup_text_block, "Early Game:", ["Mid Game:", "Late Game:", "Team Fights:"] + game_phase_section_end_keywords)
            mid_game = self._extract_section_content(matchup_text_block, "Mid Game:", ["Early Game:", "Late Game:", "Team Fights:"] + game_phase_section_end_keywords) # Added Early Game in case order varies
            late_game = self._extract_section_content(matchup_text_block, "Late Game:", ["Early Game:", "Mid Game:", "Team Fights:"] + game_phase_section_end_keywords)
            team_fights = self._extract_section_content(matchup_text_block, "Team Fights:", ["Early Game:", "Mid Game:", "Late Game:"] + game_phase_section_end_keywords)


            # 3. Strategy, Runes, Items for Champion1 and Champion2
            # Helper function to parse these details for a given champion
            def get_champion_details(champ_name: str, text: str, other_champ_name: str) -> Dict:
                details = {"overall_strategy": "Not found", "runes": "Not found", "recommended_items": "Not found"}
                # Strategy for Fiora (this is the section header)
                # Fiora Overall Strategy: ...
                # Fiora Runes: ...
                # Fiora Recommended Items: ...
                
                # Define where this champion's strategy section ends
                # It ends before the other champion's strategy, or "Basic Information" (next matchup) or end of text.
                section_end_keywords = [f"Strategy for {other_champ_name}", "Basic Information"]

                # Find the block for "Strategy for [ChampName]"
                strategy_block_pattern = rf"Strategy for {re.escape(champ_name)}\s*(.*?)(?=(?:Strategy for {re.escape(other_champ_name)}|Basic Information|$))"
                strategy_block_match = re.search(strategy_block_pattern, text, re.IGNORECASE | re.DOTALL)

                if strategy_block_match:
                    champ_block_text = strategy_block_match.group(1)
                    # Now extract specific parts from this block
                    details["overall_strategy"] = self._extract_section_content(champ_block_text, f"{re.escape(champ_name)} Overall Strategy:", [f"{re.escape(champ_name)} Runes:", f"{re.escape(champ_name)} Recommended Items:"])
                    details["runes"] = self._extract_section_content(champ_block_text, f"{re.escape(champ_name)} Runes:", [f"{re.escape(champ_name)} Overall Strategy:", f"{re.escape(champ_name)} Recommended Items:"])
                    details["recommended_items"] = self._extract_section_content(champ_block_text, f"{re.escape(champ_name)} Recommended Items:", [f"{re.escape(champ_name)} Overall Strategy:", f"{re.escape(champ_name)} Runes:"])
                else:
                    logger.warning(f"Could not find strategy block for {champ_name}")
                return details

            champion1_details = get_champion_details(champion1, matchup_text_block, champion2)
            champion2_details = get_champion_details(champion2, matchup_text_block, champion1)
            
            # Construct the matchup dictionary
            matchup_dict = {
                "champion1": champion1,
                "champion2": champion2,
                "lane": lane,
                "difficulty": difficulty, # This is for champion1 vs champion2, as per "5/10" (implicitly for C1)
                "winner_favored": winner_favored, # This is the explicit "Favored Champion"
                "win_rate": win_rate, # This is for champion1 (Fiora in the example)
                "matchup_details": {
                    "early_game": early_game if early_game != "Not found" else "No data available",
                    "mid_game": mid_game if mid_game != "Not found" else "No data available",
                    "late_game": late_game if late_game != "Not found" else "No data available",
                    "team_fights": team_fights if team_fights != "Not found" else "No data available"
                },
                "champion1_strategy_details": champion1_details,
                "champion2_strategy_details": champion2_details,
                "source": "pdf_document"
            }
            return matchup_dict

        except Exception as e:
            logger.error(f"Error parsing new PDF matchup format: {e}\nProblematic text block snippet:\n{matchup_text_block[:500]}...")
            return None

    def parse_pdf_files(self) -> List[Dict]:
        all_matchups = []
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            logger.info(f"Processing PDF: {pdf_file}")
            full_pdf_text = self.extract_text_from_pdf(pdf_path)

            if not full_pdf_text:
                logger.warning(f"Could not extract text from {pdf_file}")
                continue

            # Split the PDF text by "Basic Information" to handle multiple matchups per PDF
            # The delimiter itself will be lost, so we need to re-add it if it's not the first block.
            # A more robust way is to find all occurrences of the start of a matchup.
            # The start could be "League of Legends Matchup Information" (once at the top)
            # or "Basic Information" for subsequent ones.
            
            # Use a regex that captures the start of a matchup block.
            # A matchup block starts with "Basic Information" or the very beginning of the document
            # if it starts with "League of Legends Matchup Information" followed by "Basic Information"
            
            # Split points will be "Basic Information" or the initial "League of Legends Matchup Information"
            # Need to ensure the split includes the "Basic Information" text in each chunk.
            
            # Find all starting positions of "Basic Information"
            matchup_starts = [m.start() for m in re.finditer(r"Basic Information", full_pdf_text, re.IGNORECASE)]
            
            # Also consider the document start if it has the main header
            if full_pdf_text.lower().startswith("league of legends matchup information"):
                # Check if "Basic Information" immediately follows or is close
                first_basic_info_match = re.search(r"Basic Information", full_pdf_text, re.IGNORECASE)
                if first_basic_info_match and first_basic_info_match.start() < 200: # Arbitrary small distance
                     # If "Basic Information" is already found and is the first, don't add 0
                     if not matchup_starts or matchup_starts[0] != first_basic_info_match.start():
                        # This case is tricky; if "Basic Info" is the first real matchup_start, we don't want 0.
                        # Let's simplify: if the doc starts with the main header, and Basic Info is near,
                        # we assume the first matchup_start is the "Basic Information".
                        pass # The existing matchup_starts should cover it.
                elif 0 not in matchup_starts: # If no "Basic Information" found early, but doc starts with main header
                     # This implies a format where the first matchup doesn't have "Basic Information"
                     # but starts directly after the main header. This contradicts new PDF.
                     # For now, assume "Basic Information" is always present for a matchup.
                     pass


            if not matchup_starts: # No "Basic Information" sections found
                # Try to parse the whole document as one matchup if it contains a "Matchup:" line
                if "Matchup:" in full_pdf_text:
                     logger.info(f"No 'Basic Information' sections in {pdf_file}, attempting to parse as single matchup.")
                     matchup_blocks = [full_pdf_text]
                else:
                    logger.warning(f"No 'Basic Information' sections found in {pdf_file}, and no 'Matchup:' line. Skipping.")
                    continue
            else:
                matchup_blocks = []
                # Create blocks from start indices
                for i in range(len(matchup_starts)):
                    start_index = matchup_starts[i]
                    # The text for the current matchup starts at "Basic Information"
                    # and ends just before the next "Basic Information" or at the end of the document.
                    # We need to find the "Matchup:" line that belongs to this "Basic Information"
                    
                    # The actual content for a matchup starts *after* "Basic Information" line
                    # but the "Matchup:" line is critical.
                    # Let's assume the "Basic Information" header is immediately followed by the "Matchup:" line.
                    
                    # The text block for `extract_matchup_info` should contain the "Matchup:" line.
                    # `matchup_text_block` should start from where "Matchup: ..." is.
                    # So we need to find "Matchup:" after "Basic Information"
                    
                    block_text_start_search_area = full_pdf_text[start_index:]
                    matchup_line_within_block_match = re.search(r"Matchup:.*Win Rate:[\s\d.%]+", block_text_start_search_area, re.IGNORECASE | re.DOTALL)
                    
                    if not matchup_line_within_block_match:
                        logger.warning(f"Found 'Basic Information' in {pdf_file} but no subsequent 'Matchup:' line. Skipping this block.")
                        continue

                    # The actual content starts from the "Matchup:" line found.
                    actual_block_start_offset = matchup_line_within_block_match.start()
                    
                    # End of this block is start of next "Basic Information" or end of doc.
                    end_index = matchup_starts[i+1] if (i + 1) < len(matchup_starts) else len(full_pdf_text)
                    
                    # The text for parsing is from current "Matchup:" line to end_index (relative to start_index for text content)
                    current_matchup_text = block_text_start_search_area[actual_block_start_offset : (end_index - start_index)]
                    matchup_blocks.append(current_matchup_text)

            logger.info(f"Found {len(matchup_blocks)} potential matchup blocks in {pdf_file}.")

            for i, block in enumerate(matchup_blocks):
                logger.debug(f"Parsing block {i+1} from {pdf_file}...")
                # logger.debug(f"Block content snippet for parsing:\n{block[:500]}\n---")
                matchup = self.extract_matchup_info(block)
                if matchup:
                    all_matchups.append(matchup)
                    logger.info(f"Successfully parsed matchup {i+1} ({matchup['champion1']} vs {matchup['champion2']}) from {pdf_file}")
                else:
                    logger.warning(f"Failed to parse matchup block {i+1} from {pdf_file}.")
        
        if not all_matchups and not pdf_files: # No files, no problem if scraper works
            pass
        elif not all_matchups and pdf_files: # Files existed but no matchups parsed
            logger.warning("No matchups extracted from any PDF files. Check PDF format and parsing logic.")


        return all_matchups
    
    def _get_sample_matchups(self) -> List[Dict]:
        logger.info("Using sample matchup data (new format)")
        return [
            {
                "champion1": "Fiora", "champion2": "Darius", "lane": "top", "difficulty": 5,
                "winner_favored": "Fiora", "win_rate": 51.94,
                "matchup_details": {
                    "early_game": "Darius wins from levels 1-6 due to his all in potential assuming he has ghost and flash. Darius can easily zone Fiora off the wave pre-6 so freezing the wave in front of his turret is a very advantageous position for him. Ideally Fiora wants to 2 or 3 wave crash, then play safe for the bounce back and recall for tiamat(waveclear item) and tp back to lane. Darius wants to get a kill pre 6",
                    "mid_game": "Post level 6, Fiora should win fights assuming she dodges Darius Decimate(Q) healing at the tip, gets at least 1 free early vital, procs all 4 ults vitals, and lastly Riposte(W) the Darius Ultimate. Fiora should win but there is room for error Darius can win with Fiora messes up or wastefully uses the Riposte(W).",
                    "late_game": "Fiora continues to win in the side lane, same fight pattern and win conditions.",
                    "team_fights": "Fiora and Darius go even in team fights Fiora can either target Darius and deny his resets or assassinate low health carries with her strong damage, Darius should focus on getting Noxian Guillotine (R) Resets to win. Fiora should focus on split pushing and drawing pressure away from objectives over team fighting however there is a time for both. If either champion has a lead they should try to help in objective fights when splitting isn't worth it."
                },
                "champion1_strategy_details": {
                    "overall_strategy": "Main Focus on dodging Darius Decimate(Q) and Riposte(W) on Darius' Noxian Guillotine(R). Split Push with your lead can easily win 1 versus 2 when ahead.",
                    "runes": "Conqueror, Absorb Life, Alacrity, Last Stand, Revitalize, Bone Plating, Attack Speed, Adaptive Damage, Flat Health.",
                    "recommended_items": "Ravenous Hydra, Voltaic Cyclosword, Triforce, Hullbreaker, Death's Dance, Serylda's Grudge, Guardian Angel."
                },
                "champion2_strategy_details": {
                    "overall_strategy": "Try to control the wave and freeze near your turret early. Go for all in Pre-6 off of a freeze, you get outscaled if you go even or are slightly ahead so try to punish the early weakness of Fiora.",
                    "runes": "Conqueror, Triumph, Alacrity, Last Stand, (Nimbus Cloak, Clerity for offensive) or (Bone Plating, Unflinching for defensive), Attack Speed, Adaptive Damage, Flat Health.", # From page 2
                    "recommended_items": "Trinity Force, Stridebreaker, Dead Man's Plate,, Death's Dance, Sterak's Gage, Force of Nature." # From page 2
                },
                "source": "sample_data_new_format"
            }
        ]

    def create_pdf_template(self, output_dir: str = "templates") -> str:
        try:
            import fpdf
        except ImportError:
            logger.error("FPDF (fpdf2) library not found. Please install it: pip install fpdf2")
            return "Error: FPDF (fpdf2) library not found. Cannot create template."

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "matchup_template_v2.pdf")
        
        pdf = fpdf.FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, txt="League of Legends Matchup Information", ln=True, align='C')
        pdf.ln(5)

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, txt="Basic Information", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 7, txt="Matchup: ChampionOne vs ChampionTwo in lane_name lane Difficulty: X/10 Favored Champion: ChampionOne Win Rate: XX.X%")
        pdf.ln(5)

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, txt="Game Phases", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 7, txt="Early Game: [Describe early game dynamics here, laning, kill pressure, zoning, recall timings, etc.]")
        pdf.multi_cell(0, 7, txt="Mid Game: [Describe mid game dynamics, power spikes, objective control, roaming, skirmishes, etc.]")
        pdf.multi_cell(0, 7, txt="Late Game: [Describe late game dynamics, scaling, team fight roles, split pushing, win conditions, etc.]")
        pdf.multi_cell(0, 7, txt="Team Fights: [Describe team fight specific interactions, target priority, peel/dive considerations, etc.]")
        pdf.ln(5)

        # Champion One Strategy
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, txt="Strategy for ChampionOne", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 7, txt="ChampionOne Overall Strategy: [Detailed overall strategy for ChampionOne in this matchup. Key abilities to use/dodge, wave management, trading patterns, win conditions.]")
        pdf.multi_cell(0, 7, txt="ChampionOne Runes: [List of recommended runes for ChampionOne, e.g., Conqueror, Electrocute, Phase Rush, secondary tree choices.]")
        pdf.multi_cell(0, 7, txt="ChampionOne Recommended Items: [List of core and situational items for ChampionOne, e.g., Mythic, Legendary items, boots, starting items.]")
        pdf.ln(5)

        # Champion Two Strategy
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, txt="Strategy for ChampionTwo", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 7, txt="ChampionTwo Overall Strategy: [Detailed overall strategy for ChampionTwo.]")
        pdf.multi_cell(0, 7, txt="ChampionTwo Runes: [List of recommended runes for ChampionTwo.]")
        pdf.multi_cell(0, 7, txt="ChampionTwo Recommended Items: [List of core and situational items for ChampionTwo.]")
        
        pdf.output(output_path)
        logger.info(f"Created new template PDF (v2) at {output_path}")
        return output_path


#--- STEP 2: DOCUMENT CHUNKING ---
def create_document_chunks(matchups: List[Dict]) -> Tuple[List[str], List[Dict]]:
    documents = []
    metadata_list = []
    
    for matchup in matchups:
        # Basic matchup information chunk
        basic_info_chunk = f"""
        Matchup: {matchup['champion1']} vs {matchup['champion2']} in {matchup['lane']} lane.
        Difficulty (for {matchup['champion1']}): {matchup.get('difficulty', 'N/A')}/10.
        Favored Champion: {matchup.get('winner_favored', 'N/A')}.
        Win Rate for {matchup['champion1']} (if specified): {matchup.get('win_rate', 'N/A')}%.
        Source: {matchup.get('source', 'N/A')}.
        {f"URL: {matchup['url_queried']}" if matchup.get('url_queried') and matchup['source'] == 'lolalytics.com' else ""}
        """.strip()
        documents.append(basic_info_chunk)
        current_metadata = {
            "champion1": matchup["champion1"], "champion2": matchup["champion2"],
            "lane": matchup["lane"], "source": matchup.get("source", "N/A"),
            "url_queried": matchup.get("url_queried") if matchup.get('source') == 'lolalytics.com' else None
        }
        metadata_list.append(current_metadata)

        # Game phases chunk (if details exist)
        phases_detail = matchup.get('matchup_details', {})
        early = phases_detail.get('early_game', 'No data available').strip()
        mid = phases_detail.get('mid_game', 'No data available').strip()
        late = phases_detail.get('late_game', 'No data available').strip()
        team_fights = phases_detail.get('team_fights', 'No data available').strip()

        if not all(s == "No data available" or s == "Not found" or not s for s in [early, mid, late, team_fights]):
            game_phases_chunk = f"""
            Game Phases for {matchup['champion1']} vs {matchup['champion2']} ({matchup['lane']}):
            Early Game: {early}
            Mid Game: {mid}
            Late Game: {late}
            Team Fights: {team_fights}
            """.strip()
            documents.append(game_phases_chunk)
            metadata_list.append(current_metadata)

        # Strategy for Champion1 (if details exist)
        c1_strat_details = matchup.get('champion1_strategy_details', {})
        c1_overall = c1_strat_details.get('overall_strategy', 'No data available').strip()
        c1_runes = c1_strat_details.get('runes', 'No data available').strip()
        c1_items = c1_strat_details.get('recommended_items', 'No data available').strip()

        if not all(s == "No data available" or s == "Not found" or not s for s in [c1_overall, c1_runes, c1_items]):
            c1_strategy_chunk = f"""
            Strategy for {matchup['champion1']} against {matchup['champion2']} ({matchup['lane']}):
            Overall Strategy: {c1_overall}
            Runes: {c1_runes}
            Recommended Items: {c1_items}
            """.strip()
            documents.append(c1_strategy_chunk)
            metadata_list.append(current_metadata)

        # Strategy for Champion2 (if details exist)
        c2_strat_details = matchup.get('champion2_strategy_details', {})
        c2_overall = c2_strat_details.get('overall_strategy', 'No data available').strip()
        c2_runes = c2_strat_details.get('runes', 'No data available').strip()
        c2_items = c2_strat_details.get('recommended_items', 'No data available').strip()
        
        if not all(s == "No data available" or s == "Not found" or not s for s in [c2_overall, c2_runes, c2_items]):
            c2_strategy_chunk = f"""
            Strategy for {matchup['champion2']} against {matchup['champion1']} ({matchup['lane']}):
            Overall Strategy: {c2_overall}
            Runes: {c2_runes}
            Recommended Items: {c2_items}
            """.strip()
            documents.append(c2_strategy_chunk)
            metadata_list.append(current_metadata)
            
    return documents, metadata_list

# --- STEP 3: VECTOR DATABASE SETUP ---

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

# --- STEP 4: RETRIEVAL SYSTEM ---

def extract_champion_names(query):
    """
    Extract champion names from user query to improve retrieval.
    A more robust implementation could use NER or a dictionary lookup.
    
    Args:
        query: User query string
        
    Returns:
        List of identified champion names
    """
    
    common_champions = [
        "Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Amumu", "Anivia", "Annie", "Aphelios", 
        "Ashe", "Aurelion Sol", "Azir", "Bard", "Blitzcrank", "Brand", "Braum", "Caitlyn", 
        "Camille", "Cassiopeia", "Cho'Gath", "Corki", "Darius", "Diana", "Dr. Mundo", "Draven", 
        "Ekko", "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz", "Galio", "Gangplank", 
        "Garen", "Gnar", "Gragas", "Graves", "Gwen", "Hecarim", "Heimerdinger", "Illaoi", "Irelia", 
        "Ivern", "Janna", "Jarvan IV", "Jax", "Jayce", "Jhin", "Jinx", "Kai'Sa", "Kalista", "Karma", 
        "Karthus", "Kassadin", "Katarina", "Kayle", "Kayn", "Kennen", "Kha'Zix", "Kindred", "Kled", 
        "Kog'Maw", "LeBlanc", "Lee Sin", "Leona", "Lillia", "Lissandra", "Lucian", "Lulu", "Lux", 
        "Malphite", "Malzahar", "Maokai", "Master Yi", "Miss Fortune", "Mordekaiser", "Morgana", 
        "Nami", "Nasus", "Nautilus", "Neeko", "Nidalee", "Nocturne", "Nunu", "Olaf", "Orianna", 
        "Ornn", "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn", "Rakan", "Rammus", "Rek'Sai", 
        "Rell", "Renekton", "Rengar", "Riven", "Rumble", "Ryze", "Samira", "Sejuani", "Senna", 
        "Seraphine", "Sett", "Shaco", "Shen", "Shyvana", "Singed", "Sion", "Sivir", "Skarner", 
        "Sona", "Soraka", "Swain", "Sylas", "Syndra", "Tahm Kench", "Taliyah", "Talon", "Taric", 
        "Teemo", "Thresh", "Tristana", "Trundle", "Tryndamere", "Twisted Fate", "Twitch", "Udyr", 
        "Urgot", "Varus", "Vayne", "Veigar", "Vel'Koz", "Vi", "Viego", "Viktor", "Vladimir", 
        "Volibear", "Warwick", "Wukong", "Xayah", "Xerath", "Xin Zhao", "Yasuo", "Yone", "Yorick", 
        "Yuumi", "Zac", "Zed", "Ziggs", "Zilean", "Zoe", "Zyra"
    ]
    
    found_champions = []
    
    for champion in common_champions:
        if champion.lower() in query.lower():
            found_champions.append(champion)
    
    return found_champions

def query_matchup(vector_db, query, top_k=5):
    """
    Query the vector database for matchup information with enhanced logic.
    
    Args:
        vector_db: Vector database instance
        query: User query string
        top_k: Number of results to return
        
    Returns:
        List of relevant document chunks with metadata
    """
    
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

# --- STEP 5: LLM INTEGRATION ---

def generate_response(query, retrieved_contexts, api_key=None):
    """
    Generate response using OpenAI API with enhanced context.
    
    Args:
        query: User query string
        retrieved_contexts: List of relevant document chunks
        api_key: API key for OpenAI (can be None if using environment variable)
        
    Returns:
        Generated response string
    """
    
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
    
    # Check if API key is provided, otherwise try to get from environment
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "$API_KEY") 
        #Set a user or system env variable to name OPENAI_API_KEY and value "$API_KEY"
    
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = api_key
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-ai/deepseek-r1",
            messages=[
                {"role": "system", "content": "You are a League of Legends matchup expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Sorry, I encountered an error while generating a response. Please try again later. Error: {str(e)}"

# --- STEP 6: PDF UPLOAD INTERFACE ---
def create_upload_interface(vector_db_instance):
    def upload_pdf_files_process(pdf_files_list): # pdf_files_list is a list of TemporaryFileWrapper objects
        if not pdf_files_list:
            return "No files uploaded."
        
        pdf_parser_instance = PDFParser() # pdf_directory is "matchup_pdfs" by default
        newly_processed_matchups = []
        results_log = []

        for pdf_file_obj in pdf_files_list:
            # pdf_file_obj is a tempfile._TemporaryFileWrapper
            # pdf_file_obj.name is the actual path to the temporary file
            temp_file_path = pdf_file_obj.name
            
            # Use the basename of the temporary file path for saving.
            # Note: This will be a name like "tmpxyz123.pdf", not the user's original filename.
            # Getting the original filename robustly requires handling gr.UploadData or similar,
            # which is more involved. For now, using the temp name for the saved copy.
            # If you need the original filename, you might need to adjust how Gradio handles uploads
            # or inspect `pdf_file_obj` for attributes like `orig_name` if your Gradio version provides it.
            filename_for_saving = os.path.basename(temp_file_path)
            
            save_path = os.path.join(pdf_parser_instance.pdf_directory, filename_for_saving)

            logger.info(f"Processing uploaded PDF (original temp path: {temp_file_path})")
            
            try:
                # --- CORRECTED FILE COPYING ---
                # Ensure the destination directory exists
                os.makedirs(pdf_parser_instance.pdf_directory, exist_ok=True)
                
                # Copy the temporary file to your persistent 'matchup_pdfs' directory
                # shutil.copyfile() is a good way to do this. It takes source and destination paths.
                shutil.copyfile(temp_file_path, save_path)
                logger.info(f"Copied uploaded PDF to: {save_path}")

                # Now, parse the PDF from its new, persistent location (save_path)
                pdf_text = pdf_parser_instance.extract_text_from_pdf(save_path)
                
                if not pdf_text:
                    results_log.append(f"Could not extract text from {filename_for_saving} (copied to {save_path})")
                    continue
                
                matchup = pdf_parser_instance.extract_matchup_info(pdf_text)
                if matchup:
                    newly_processed_matchups.append(matchup)
                    results_log.append(f"Successfully processed and added {filename_for_saving}: {matchup['champion1']} vs {matchup['champion2']}")
                else:
                    results_log.append(f"Could not extract structured matchup information from {filename_for_saving}")
            
            except Exception as e:
                # If original_filename_for_saving was defined, it would be better for logging here.
                results_log.append(f"Error processing file (temp name: {filename_for_saving}): {str(e)}")
                logger.error(f"Error processing uploaded file (temp name: {filename_for_saving}, temp path: {temp_file_path}): {e}", exc_info=True)
        
        if newly_processed_matchups:
            logger.info(f"Adding {len(newly_processed_matchups)} new matchups from uploaded PDFs to VectorDB.")
            # Ensure variable names are consistent with what create_document_chunks expects
            documents, metadata_for_db = create_document_chunks(newly_processed_matchups)
            vector_db_instance.add_documents(documents, metadata_for_db)
            results_log.append(f"Added {len(documents)} document chunks from uploaded PDFs to the knowledge base.")
        elif pdf_files_list: # Files were uploaded, but nothing processed
            results_log.append("No new matchups were successfully processed from the uploads to add to the knowledge base. Check logs and PDF format.")
        
        return "\n".join(results_log)

    # ... (rest of the create_upload_interface function, like template generation) ...
    def trigger_template_creation():
        pdf_parser_instance = PDFParser()
        try:
            template_path = pdf_parser_instance.create_pdf_template()
            return template_path 
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return None 

    with gr.Blocks(title="LoL Matchup PDF Uploader") as interface:
        gr.Markdown("# League of Legends Matchup PDF Uploader")
        gr.Markdown("Upload PDFs (following the template format) to add detailed matchup information to the knowledge base. Uploaded PDFs will be copied to the 'matchup_pdfs' folder.")
        
        with gr.Row():
            with gr.Column(scale=2):
                pdf_input_files = gr.File(file_count="multiple", label="Upload PDF Files", file_types=[".pdf"])
                upload_button_ui = gr.Button("Upload and Process PDFs", variant="primary")
            with gr.Column(scale=1):
                 gr.Markdown("### PDF Template")
                 template_download_button_ui = gr.Button("Generate & Download Template")
                 template_file_output_ui = gr.File(label="Download Template Here", interactive=False)

        output_textbox_ui = gr.Textbox(label="Upload Log", lines=10, interactive=False)
        
        upload_button_ui.click(fn=upload_pdf_files_process, inputs=[pdf_input_files], outputs=output_textbox_ui)
        
        template_download_button_ui.click(
            fn=trigger_template_creation, 
            inputs=[], 
            outputs=template_file_output_ui
        ).then(
            fn=lambda x: "Template PDF generated. Click 'Download Template Here' above to download." if x else "Failed to generate template. Check logs.",
            inputs=[template_file_output_ui],
            outputs=output_textbox_ui
        )
    return interface

# --- STEP 7: USER INTERFACE ---

def create_ui(vector_db):
    """
    Create the main Gradio interface for querying matchup information.
    
    Args:
        vector_db: Vector database instance
        
    Returns:
        Gradio interface
    """
    
    def answer_query(query, additional_context="", api_key="", debug_mode=False):
        """
        Process user query and generate response.
        
        Args:
            query: User query string
            additional_context: Optional additional context
            api_key: Optional API key for OpenAI
            debug_mode: Whether to include debug information in response
            
        Returns:
            Generated response string
        """
        
        # Combine query with additional context if provided
        full_query = query
        if additional_context:
            full_query = f"{query} Additional context: {additional_context}"
        
        # Get relevant contexts
        results = query_matchup(vector_db, full_query)
        
        # Generate response
        response = generate_response(full_query, results, api_key)
        
        # For debugging: return contexts that were used
        if debug_mode:
            context_info = "\n\n---DEBUG: Retrieved Contexts---\n"
            for i, res in enumerate(results):
                context_info += f"Context {i+1} (score: {res['score']:.2f}):\n{res['text'][:100]}...\n"
            
            return response + context_info
        
        return response
    
    def toggle_debug(debug_state):
        """Toggle debug mode"""
        return not debug_state
    
    with gr.Blocks(title="League of Legends Matchup Assistant") as interface:
        gr.Markdown("# League of Legends Matchup Assistant")
        gr.Markdown("Ask questions about champion matchups, counters, and strategies!")
        
        debug_state = gr.State(False)
        
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
            with gr.Row():
                with gr.Column():
                    context_input = gr.Textbox(
                        lines=2,
                        placeholder="Optional: Add more context about your rank, playstyle, etc.",
                        label="Additional Context"
                    )
                
                with gr.Column():
                    api_key_input = gr.Textbox(
                        placeholder="Optional: Enter your OpenAI API key",
                        label="API Key",
                        type="password"
                    )
            
            with gr.Row():
                debug_btn = gr.Button("Toggle Debug Mode")
                debug_indicator = gr.Textbox(value="Debug Mode: OFF", label="Debug Status")
        
        output = gr.Textbox(label="Matchup Advice", lines=10)
        
        submit_btn.click(
            fn=answer_query,
            inputs=[query_input, context_input, api_key_input, debug_state],
            outputs=output
        )
        
        debug_btn.click(
            fn=toggle_debug,
            inputs=[debug_state],
            outputs=debug_state
        ).then(
            fn=lambda x: f"Debug Mode: {'ON' if x else 'OFF'}",
            inputs=[debug_state],
            outputs=debug_indicator
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

# --- STEP 8: MAIN APPLICATION ---

def main():
    """
    Main application entry point.
    """
    
    logger.info("Initializing League of Legends Matchup RAG bot...")
    
    # Initialize PDF parser
    pdf_parser = PDFParser()
    
    # Parse available PDFs
    logger.info("Loading matchup data from PDFs...")
    matchups = pdf_parser.parse_pdf_files()
    logger.info(f"Loaded {len(matchups)} matchups")
    
    # Process data into retrievable chunks
    logger.info("Processing data into retrievable chunks...")
    documents, metadata = create_document_chunks(matchups)
    
    # Initialize and populate vector database
    logger.info("Setting up vector database...")
    vector_db = SimpleVectorDB()
    vector_db.add_documents(documents, metadata)
    logger.info(f"Added {len(documents)} document chunks to vector database")
    
    # Create interface
    logger.info("Creating Gradio interfaces...")
    
    # Create upload interface
    upload_ui = create_upload_interface(vector_db)
    
    # Create chat interface
    chat_ui = create_ui(vector_db)
    
    # Combine interfaces in a TabItem
    with gr.Blocks(title="League of Legends Matchup System") as app:
        with gr.Tabs():
            with gr.Tab("Chat Assistant"):
                chat_ui.render()
            with gr.Tab("Upload PDFs"):
                upload_ui.render()
    
    logger.info("Launching application...")
    app.launch(share=True)  # Set share=False in production

if __name__ == "__main__":
    main()