import re
import json
import os
from typing import Dict, Any, List, Optional
from minementor.backend.core.interfaces import PlayerModel

class PersistentPlayerModel(PlayerModel):
    """
    Tracks and updates the player state across conversation turns,
    extracts entity states from player text, and maintains player profile.
    """

    DEFAULT_STATE = {
        "edition": "Java",
        "version": "1.20",
        "dimension": "overworld",
        "day": 1,
        "health": 20,
        "food": 20,
        "inventory": {},
        "equipment": {
            "armor": "none",
            "helmet": None,
            "chestplate": None,
            "leggings": None,
            "boots": None,
            "main_hand": None,
            "off_hand": None
        },
        "resources": {
            "wood": 0,
            "cobblestone": 0,
            "coal": 0,
            "iron": 0,
            "gold": 0,
            "diamond": 0,
            "emerald": 0,
            "netherite": 0
        },
        "progression_stage": "Early Game",
        "current_goal": "Survive the first night and craft basic tools",
        "completed_milestones": []
    }

    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file or os.path.join("data", "processed", "player_profile.json")
        self.state = dict(self.DEFAULT_STATE)
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.state.update(saved)
            except Exception as e:
                print(f"Warning: could not load player profile from {self.state_file}: {e}")

    def save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def get_state(self) -> Dict[str, Any]:
        return self.state

    def update_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in updates.items():
            if isinstance(v, dict) and k in self.state and isinstance(self.state[k], dict):
                self.state[k].update(v)
            else:
                self.state[k] = v
        self.save_state()
        return self.state

    def extract_state_from_query(self, query: str) -> Dict[str, Any]:
        """
        Extracts structured player information from user query.
        Example: "I have 5 diamonds, iron armor, 32 bread and I'm on day 8."
        Returns: {'diamonds': 5, 'armor': 'iron', 'food': 32, 'day': 8}
        """
        extracted = {}
        q_lower = query.lower()

        # Extract numerical resource counts (e.g., "5 diamonds", "32 bread", "6 iron")
        resource_patterns = [
            (r"\b(\d+)\s+diamonds?\b", "diamond"),
            (r"\b(\d+)\s+iron(?:\s+ingots?)?\b", "iron"),
            (r"\b(\d+)\s+gold(?:\s+ingots?)?\b", "gold"),
            (r"\b(\d+)\s+coals?\b", "coal"),
            (r"\b(\d+)\s+emeralds?\b", "emerald"),
            (r"\b(\d+)\s+netherite(?:\s+ingots?)?\b", "netherite"),
            (r"\b(\d+)\s+(?:bread|cooked beef|steak|porkchop|food)\b", "food_items"),
            (r"\b(\d+)\s+bookshelves?\b", "bookshelves"),
            (r"\b(\d+)\s+obsidian\b", "obsidian")
        ]

        for pat, key in resource_patterns:
            m = re.search(pat, q_lower)
            if m:
                count = int(m.group(1))
                extracted[key] = count
                if key in self.state["resources"]:
                    self.state["resources"][key] = count
                self.state["inventory"][key] = count

        # Extract armor mention (e.g. "iron armor", "diamond armor", "leather armor")
        armor_match = re.search(r"\b(leather|iron|gold|golden|diamond|netherite)\s+armor\b", q_lower)
        if armor_match:
            armor_type = armor_match.group(1).replace("golden", "gold")
            extracted["armor"] = armor_type
            self.state["equipment"]["armor"] = armor_type

        # Extract day number (e.g. "on day 8", "day 12")
        day_match = re.search(r"\bday\s+(\d+)\b", q_lower)
        if day_match:
            day_num = int(day_match.group(1))
            extracted["day"] = day_num
            self.state["day"] = day_num

        # Extract dimension (overworld, nether, end)
        if "in the nether" in q_lower or "nether dimension" in q_lower:
            extracted["dimension"] = "nether"
            self.state["dimension"] = "nether"
        elif "in the end" in q_lower or "end dimension" in q_lower:
            extracted["dimension"] = "end"
            self.state["dimension"] = "end"

        # Extract edition/version if user specifies it
        if "bedrock" in q_lower:
            extracted["edition"] = "Bedrock"
            self.state["edition"] = "Bedrock"
        elif "java" in q_lower:
            extracted["edition"] = "Java"
            self.state["edition"] = "Java"

        if extracted:
            self.save_state()

        return extracted
