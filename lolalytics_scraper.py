# lolalytics_scraper.py
import requests
from bs4 import BeautifulSoup
import re
import logging
import time
from typing import Optional, Dict
import json # For testing

# Configure logging for the scraper module
# This basicConfig will apply if the module is run directly.
# If imported, the main app's logging config might take precedence or need adjustment.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


class LolalyticsScraper:
    BASE_URL = "https://lolalytics.com/lol/{champion1}/vs/{champion2}/build/"
    # Champion names that need special mapping for lolalytics URLs
    CHAMPION_NAME_TO_KEY_MAP = {
        "wukong": "monkeyking",
        "nunu & willump": "nunu",
        "nunu and willump": "nunu",
        "nunu": "nunu",
        "dr. mundo": "drmundo",
        "dr mundo": "drmundo",
        "miss fortune": "missfortune",
        "twisted fate": "twistedfate",
        "master yi": "masteryi",
        "xin zhao": "xinzhao",
        "jarvan iv": "jarvaniv",
        "kog'maw": "kogmaw",
        "cho'gath": "chogath",
        "kha'zix": "khazix",
        "vel'koz": "velkoz",
        "rek'sai": "reksai",
        "kai'sa": "kaisa",
        # Add other known special cases here (e.g. LeBlanc -> leblanc, already handled by default)
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        })

    def _normalize_champion_name_for_url(self, champion_name: str) -> str:
        """Normalizes champion name for Lolalytics URL path."""
        name_lower = champion_name.lower().strip()
        if name_lower in self.CHAMPION_NAME_TO_KEY_MAP:
            return self.CHAMPION_NAME_TO_KEY_MAP[name_lower]
        # Default normalization: lowercase, remove non-alphanumerics (except if already mapped)
        return re.sub(r"[^a-z0-9]", "", name_lower)

    def _build_url(self, champion1: str, champion2: str, lane: Optional[str] = None) -> str:
        c1_key = self._normalize_champion_name_for_url(champion1)
        c2_key = self._normalize_champion_name_for_url(champion2)
        
        url = self.BASE_URL.format(champion1=c1_key, champion2=c2_key)
        if lane:
            # Lolalytics uses specific lane names like: top, jungle, middle, bottom, support
            lane_norm = lane.lower().strip()
            if lane_norm == "adc" or lane_norm == "bot": # common alternatives
                lane_norm = "bottom"
            url += f"?vslane={lane_norm}"
        return url

    def get_matchup_win_rate(self, champion1: str, champion2: str, lane: Optional[str] = None) -> Optional[Dict]:
        url = self._build_url(champion1, champion2, lane)
        logger.info(f"Scraping: {champion1} vs {champion2} (Lane: {lane if lane else 'Overall'}) from {url}")

        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            # Being polite to the server
            time.sleep(1 + random.uniform(0.5, 1.5)) # Sleep 1.5 to 2.5s with randomness
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching {url}")
            return None
        except requests.exceptions.RequestException as e:
            status_code = response.status_code if 'response' in locals() and hasattr(response, 'status_code') else 'N/A'
            logger.error(f"Error fetching {url}: {e} (Status code: {status_code})")
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        
        # --- New Win Rate Extraction Logic based on q:key="w9_0" ---
        target_div = soup.find('div', attrs={'q:key': 'w9_0'})

        if not target_div:
            logger.warning(f"Could not find the target div with q:key='w9_0' on {url} for {champion1} vs {champion2}.")
            # Optional: Log a snippet of the HTML for debugging
            # with open(f"debug_page_{self._normalize_champion_name_for_url(champion1)}_{self._normalize_champion_name_for_url(champion2)}.html", "w", encoding="utf-8") as f:
            #    f.write(soup.prettify())
            # logger.info(f"Saved HTML snapshot for debugging.")
            return None

        # The div q:key="w9_0" has two child divs:
        # 1. Child div [0]: Contains the label text (e.g., "Win Rate")
        # 2. Child div [1]: Contains the win rate value (e.g., "52.34%")
        child_divs = target_div.find_all('div', recursive=False) 

        if not child_divs or len(child_divs) < 2:
            logger.warning(f"Target div q:key='w9_0' found, but it does not have the expected two child divs on {url}. Found {len(child_divs) if child_divs else 0} child divs.")
            return None
        
        # The win rate value is in the second child div.
        win_rate_text_container = child_divs[1] 
        win_rate_value_str = win_rate_text_container.get_text(strip=True)

        if not win_rate_value_str:
            logger.warning(f"Second child div of q:key='w9_0' found, but it contains no text on {url}.")
            return None
            
        # Expected format: "52.34%"
        match = re.search(r'(\d{1,2}\.\d{1,2})%', win_rate_value_str)
        if match:
            win_rate = float(match.group(1))
            # champion1 is the one whose perspective the stats are from.
            winner_favored = champion1 if win_rate > 50.0 else (champion2 if win_rate < 50.0 else "Even")
            
            logger.info(f"Successfully scraped win rate for {champion1} vs {champion2} (Lane: {lane if lane else 'Overall'}): {win_rate}%")
            return {
                "champion1": champion1, 
                "champion2": champion2, 
                "lane": lane.lower().strip() if lane else "overall",
                "win_rate": win_rate, # This is champion1's win rate in the matchup
                "winner_favored": winner_favored,
                "source": "lolalytics.com",
                "url_queried": url
            }
        else:
            logger.warning(f"Could not parse win rate from text: '{win_rate_value_str}' (from q:key='w9_0' path) for {champion1} on {url}.")
            return None

if __name__ == '__main__':
    import random # for sleep variation
    # Example usage for testing the scraper directly
    # This ensures that if you run `python lolalytics_scraper.py`, it tests itself.
    scraper = LolalyticsScraper()
    
    test_matchups = [
        ("Darius", "Garen", "top"),
        ("Ahri", "Zed", "middle"),
        ("Jinx", "Caitlyn", "bottom"),
        ("Wukong", "Sett", "top"),
        ("Dr. Mundo", "Cho'Gath", "top"),
        ("Lee Sin", "Elise", "jungle"),
        ("Miss Fortune", "Ezreal", "adc"), # Test "adc" as lane
        ("Aatrox", "Fiora", None), # Test without lane
        ("Malphite", "Yasuo", "top"),
        ("Annie", "Veigar", "middle"),
        ("Tahm Kench", "Senna", "support"),
        ("Kog'Maw", "Lulu", "bottom") # Test Kog'Maw normalization
    ]

    for c1, c2, lane_val in test_matchups:
        print(f"\n--- Testing: {c1} vs {c2} (Lane: {lane_val}) ---")
        data = scraper.get_matchup_win_rate(c1, c2, lane_val)
        if data:
            print(json.dumps(data, indent=2))
        else:
            print(f"Failed to retrieve data for {c1} vs {c2} (Lane: {lane_val}).")
        print("-------------------------------------------------")