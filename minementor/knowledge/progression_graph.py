import json
import os
from typing import List, Dict, Any, Optional

DEFAULT_PROGRESSION_DATA = {
    "nodes": {
        "punch_wood": {
            "name": "Punch Wood",
            "tier": 0,
            "category": "gathering",
            "prerequisites": [],
            "required_items": {},
            "rewards": ["oak_log", "birch_log", "spruce_log"],
            "description": "Gather wood logs by punching tree trunks."
        },
        "crafting_table": {
            "name": "Make Crafting Table",
            "tier": 0,
            "category": "crafting",
            "prerequisites": ["punch_wood"],
            "required_items": {"wood_planks": 4},
            "rewards": ["crafting_table"],
            "description": "Craft a 3x3 crafting table using four wooden planks."
        },
        "wooden_pickaxe": {
            "name": "Craft Wooden Pickaxe",
            "tier": 0,
            "category": "crafting",
            "prerequisites": ["crafting_table"],
            "required_items": {"wood_planks": 3, "stick": 2},
            "rewards": ["wooden_pickaxe"],
            "description": "Craft a wooden pickaxe to begin mining stone."
        },
        "stone_age": {
            "name": "Mine Cobblestone",
            "tier": 1,
            "category": "mining",
            "prerequisites": ["wooden_pickaxe"],
            "required_items": {"wooden_pickaxe": 1},
            "rewards": ["cobblestone"],
            "description": "Mine cobblestone with a wooden pickaxe."
        },
        "stone_tools_and_furnace": {
            "name": "Craft Stone Tools & Furnace",
            "tier": 1,
            "category": "crafting",
            "prerequisites": ["stone_age"],
            "required_items": {"cobblestone": 11, "stick": 4},
            "rewards": ["stone_pickaxe", "stone_sword", "furnace"],
            "description": "Upgrade to stone equipment and build a furnace for smelting."
        },
        "iron_age": {
            "name": "Mine & Smelt Iron",
            "tier": 2,
            "category": "mining",
            "prerequisites": ["stone_tools_and_furnace"],
            "required_items": {"stone_pickaxe": 1, "furnace": 1, "fuel": 1},
            "rewards": ["iron_ingot"],
            "description": "Mine raw iron from caves or mountains and smelt it in a furnace."
        },
        "iron_gear_and_shield": {
            "name": "Craft Iron Armor, Pickaxe & Shield",
            "tier": 2,
            "category": "crafting",
            "prerequisites": ["iron_age"],
            "required_items": {"iron_ingot": 28, "wood_planks": 6},
            "rewards": ["iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots", "iron_pickaxe", "shield", "water_bucket"],
            "description": "Equip full iron armor, craft an iron pickaxe, shield, and water bucket."
        },
        "diamonds": {
            "name": "Mine Diamonds",
            "tier": 3,
            "category": "mining",
            "prerequisites": ["iron_gear_and_shield"],
            "required_items": {"iron_pickaxe": 1},
            "rewards": ["diamond"],
            "description": "Mine deep underground (Y=-58) with an iron pickaxe to uncover diamonds."
        },
        "diamond_pickaxe": {
            "name": "Craft Diamond Pickaxe",
            "tier": 3,
            "category": "crafting",
            "prerequisites": ["diamonds"],
            "required_items": {"diamond": 3, "stick": 2},
            "rewards": ["diamond_pickaxe"],
            "description": "Craft a diamond pickaxe, the only tool capable of harvesting obsidian."
        },
        "obsidian_and_nether_portal": {
            "name": "Harvest Obsidian & Light Nether Portal",
            "tier": 4,
            "category": "exploration",
            "prerequisites": ["diamond_pickaxe"],
            "required_items": {"diamond_pickaxe": 1, "flint_and_steel": 1},
            "rewards": ["obsidian", "nether_portal"],
            "description": "Mine 10-14 obsidian blocks and build a 4x5 vertical portal frame, then ignite with flint and steel."
        },
        "enchanting_setup": {
            "name": "Build Enchanting Table & Bookshelves",
            "tier": 4,
            "category": "crafting",
            "prerequisites": ["diamond_pickaxe"],
            "required_items": {"diamond": 2, "obsidian": 4, "book": 1},
            "rewards": ["enchanting_table"],
            "description": "Craft an enchanting table and surround it with 15 bookshelves for level 30 enchantments."
        },
        "nether_fortress": {
            "name": "Explore Nether Fortress & Gather Blaze Rods",
            "tier": 5,
            "category": "combat",
            "prerequisites": ["obsidian_and_nether_portal"],
            "required_items": {"nether_portal": 1, "gold_armor_piece": 1},
            "rewards": ["blaze_rod", "nether_wart"],
            "description": "Navigate the Nether to find a Fortress, defeat Blazes for rods, and collect nether wart."
        },
        "eyes_of_ender": {
            "name": "Brewing & Craft Eyes of Ender",
            "tier": 5,
            "category": "crafting",
            "prerequisites": ["nether_fortress"],
            "required_items": {"blaze_rod": 1, "ender_pearl": 1},
            "rewards": ["blaze_powder", "brewing_stand", "eye_of_ender"],
            "description": "Craft blaze powder and combine with ender pearls to produce Eyes of Ender."
        },
        "stronghold": {
            "name": "Locate & Activate Stronghold End Portal",
            "tier": 6,
            "category": "exploration",
            "prerequisites": ["eyes_of_ender"],
            "required_items": {"eye_of_ender": 12},
            "rewards": ["end_portal_activated"],
            "description": "Throw Eyes of Ender to track down the Stronghold portal room and fill empty frame sockets."
        },
        "ender_dragon": {
            "name": "Defeat the Ender Dragon",
            "tier": 7,
            "category": "combat",
            "prerequisites": ["stronghold"],
            "required_items": {"bow": 1, "arrows": 64, "water_bucket": 1, "solid_blocks": 64},
            "rewards": ["dragon_egg", "dragon_breath", "end_gateway"],
            "description": "Destroy the End Crystals on obsidian towers, slay the Ender Dragon, and unlock the End Gateway."
        },
        "end_cities_and_elytra": {
            "name": "Loot End Cities & Obtain Elytra",
            "tier": 8,
            "category": "exploration",
            "prerequisites": ["ender_dragon"],
            "required_items": {"ender_pearl": 16, "solid_blocks": 128},
            "rewards": ["elytra", "shulker_shell", "shulker_box"],
            "description": "Explore outer End islands, loot End Cities, defeat Shulkers for boxes, and claim the Elytra from an End Ship."
        }
    }
}

class MinecraftProgressionGraph:
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or os.path.join(os.path.dirname(__file__), "progression_graph.json")
        self.nodes = {}
        self.load_graph()

    def load_graph(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                self.nodes = content.get("nodes", {})
        else:
            self.nodes = DEFAULT_PROGRESSION_DATA["nodes"]
            self.save_graph()

    def save_graph(self):
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump({"nodes": self.nodes}, f, indent=2)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.nodes.get(node_id)

    def get_prerequisites(self, node_id: str) -> List[str]:
        node = self.nodes.get(node_id)
        return node.get("prerequisites", []) if node else []

    def get_available_next_steps(self, completed_nodes: List[str]) -> List[Dict[str, Any]]:
        """Returns all nodes whose prerequisites are satisfied, sorted by tier."""
        completed_set = set(completed_nodes)
        available = []
        for node_id, data in self.nodes.items():
            if node_id not in completed_set:
                prereqs = data.get("prerequisites", [])
                if all(p in completed_set for p in prereqs):
                    node_copy = dict(data)
                    node_copy["id"] = node_id
                    available.append(node_copy)
        return sorted(available, key=lambda x: x.get("tier", 0))

    def evaluate_player_tier(self, inventory_items: List[str], achievements: List[str]) -> int:
        """Determines the current tier based on player inventory and completed achievements."""
        tier = 0
        inv_set = set(item.lower() for item in inventory_items)
        ach_set = set(ach.lower() for ach in achievements)

        if "elytra" in inv_set or "shulker_box" in inv_set:
            return 8
        if "free the end" in ach_set or "dragon_egg" in inv_set:
            return 7
        if "eye_of_ender" in inv_set or "stronghold" in ach_set:
            return 6
        if "blaze_rod" in inv_set or "nether_wart" in inv_set:
            return 5
        if "obsidian" in inv_set or "diamond_pickaxe" in inv_set:
            return 4
        if "diamond" in inv_set:
            return 3
        if "iron_ingot" in inv_set or "iron_pickaxe" in inv_set:
            return 2
        if "cobblestone" in inv_set or "stone_pickaxe" in inv_set:
            return 1
        return tier
