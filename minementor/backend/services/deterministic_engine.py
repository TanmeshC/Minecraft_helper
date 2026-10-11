import json
import os
import re
from typing import Dict, Any, Optional, List
from collections import Counter

class DeterministicCalculationEngine:
    """
    Deterministic calculation engine for Minecraft crafting requirements,
    base resource trees, and crafting recipe lookups.
    """

    # Base decompositions for common composite items
    # Format: target_item -> {sub_component: count}
    BASE_DECOMPOSITIONS = {
        "bookshelf": {"book": 3, "wood_planks": 6},
        "book": {"paper": 3, "leather": 1},
        "paper": {"sugar_cane": 3},  # yields 3 paper for 3 sugar cane -> 1 paper = 1 sugar_cane
        "enchanting_table": {"obsidian": 4, "diamond": 2, "book": 1},
        "anvil": {"iron_block": 3, "iron_ingot": 4},
        "iron_block": {"iron_ingot": 9},
        "diamond_block": {"diamond": 9},
        "gold_block": {"gold_ingot": 9},
        "chest": {"wood_planks": 8},
        "hopper": {"iron_ingot": 5, "chest": 1},
        "piston": {"wood_planks": 3, "cobblestone": 4, "iron_ingot": 1, "redstone": 1},
        "sticky_piston": {"piston": 1, "slime_ball": 1},
        "observer": {"cobblestone": 6, "redstone": 2, "nether_quartz": 1},
        "repeater": {"redstone_torch": 2, "redstone": 1, "stone": 3},
        "comparator": {"redstone_torch": 3, "nether_quartz": 1, "stone": 3},
        "redstone_torch": {"stick": 1, "redstone": 1},
        "torch": {"stick": 1, "coal": 1},
        "stick": {"wood_planks": 2},  # yields 4 sticks -> handled in crafting ratios
        "wood_planks": {"wood_log": 1},  # yields 4 planks
        "iron_pickaxe": {"iron_ingot": 3, "stick": 2},
        "diamond_pickaxe": {"diamond": 3, "stick": 2},
        "iron_sword": {"iron_ingot": 2, "stick": 1},
        "diamond_sword": {"diamond": 2, "stick": 1},
        "shield": {"wood_planks": 6, "iron_ingot": 1},
        "bucket": {"iron_ingot": 3},
        "nether_portal_minimum": {"obsidian": 10},
        "nether_portal_full": {"obsidian": 14},
        "eye_of_ender": {"ender_pearl": 1, "blaze_powder": 1},
        "blaze_powder": {"blaze_rod": 1},  # 1 blaze rod yields 2 blaze powder
    }

    def __init__(self, recipes_path: Optional[str] = None):
        self.recipes_path = recipes_path or os.path.join("data", "cleaned", "recipes.json")
        self.recipes_by_result = defaultdict_recipes = {}
        self.load_recipes()

    def load_recipes(self):
        if os.path.exists(self.recipes_path):
            try:
                with open(self.recipes_path, "r", encoding="utf-8") as f:
                    recipes = json.load(f)
                    for r in recipes:
                        res_name = r.get("result_item", "").lower()
                        if res_name:
                            self.recipes_by_result[res_name] = r
            except Exception as e:
                print(f"Warning: could not load recipes from {self.recipes_path}: {e}")

    def get_recipe(self, item_name: str) -> Optional[Dict[str, Any]]:
        norm_item = item_name.lower().replace(" ", "_").replace("minecraft:", "")
        return self.recipes_by_result.get(norm_item)

    def calculate_resource_needs(self, target_item: str, quantity: int = 1) -> Dict[str, Any]:
        """
        Calculates exact raw and intermediate resources required for a given target quantity.
        """
        norm_item = target_item.lower().replace(" ", "_").replace("minecraft:", "")
        
        # Specific known complex calculations
        if norm_item in ["bookshelf", "bookshelves"]:
            # 1 bookshelf = 3 books + 6 planks.
            # 1 book = 3 paper + 1 leather.
            # 1 bookshelf needs: 9 paper, 3 leather, 6 wood planks.
            paper_needed = quantity * 9
            leather_needed = quantity * 3
            planks_needed = quantity * 6
            sugar_cane_needed = paper_needed  # 3 cane -> 3 paper
            wood_logs = (planks_needed + 3) // 4

            return {
                "target_item": "bookshelf",
                "target_quantity": quantity,
                "intermediate_requirements": {
                    "book": quantity * 3,
                    "wood_planks": planks_needed
                },
                "raw_materials": {
                    "paper": paper_needed,
                    "sugar_cane": sugar_cane_needed,
                    "leather": leather_needed,
                    "wood_planks": planks_needed,
                    "wood_logs": wood_logs
                },
                "breakdown_explanation": (
                    f"To craft {quantity} bookshelf{'s' if quantity > 1 else ''}, you need:\n"
                    f"• {quantity * 3} Books ({paper_needed} Paper + {leather_needed} Leather)\n"
                    f"• {planks_needed} Wooden Planks (approx {wood_logs} Wood Logs)\n"
                    f"Total Raw Materials: {paper_needed} Paper (from {sugar_cane_needed} Sugar Cane), "
                    f"{leather_needed} Leather, and {planks_needed} Planks."
                )
            }

        elif norm_item in ["enchanting_table", "enchantment_table"]:
            # 1 enchanting table = 4 obsidian, 2 diamonds, 1 book (3 paper, 1 leather)
            return {
                "target_item": "enchanting_table",
                "target_quantity": quantity,
                "intermediate_requirements": {"book": quantity},
                "raw_materials": {
                    "obsidian": 4 * quantity,
                    "diamond": 2 * quantity,
                    "paper": 3 * quantity,
                    "leather": 1 * quantity
                },
                "breakdown_explanation": (
                    f"To craft {quantity} Enchanting Table{'s' if quantity > 1 else ''}, you need:\n"
                    f"• {4 * quantity} Obsidian\n"
                    f"• {2 * quantity} Diamonds\n"
                    f"• {quantity} Book ({3 * quantity} Paper + {1 * quantity} Leather)"
                )
            }

        elif norm_item in ["anvil"]:
            # 1 anvil = 3 iron blocks + 4 iron ingots = 31 iron ingots
            total_ingots = quantity * 31
            return {
                "target_item": "anvil",
                "target_quantity": quantity,
                "intermediate_requirements": {"iron_block": 3 * quantity, "iron_ingot": 4 * quantity},
                "raw_materials": {"iron_ingot": total_ingots},
                "breakdown_explanation": (
                    f"To craft {quantity} Anvil{'s' if quantity > 1 else ''}, you need:\n"
                    f"• 3 Iron Blocks (27 Iron Ingots) + 4 Iron Ingots\n"
                    f"Total: {total_ingots} Iron Ingots."
                )
            }

        # Check in loaded recipes
        recipe = self.get_recipe(norm_item)
        if recipe:
            ingredients = recipe.get("ingredients_summary", {})
            yield_count = recipe.get("result_count", 1)
            batches_needed = (quantity + yield_count - 1) // yield_count
            calculated_reqs = {ing: count * batches_needed for ing, count in ingredients.items()}
            return {
                "target_item": norm_item,
                "target_quantity": quantity,
                "yield_per_craft": yield_count,
                "crafting_batches": batches_needed,
                "raw_materials": calculated_reqs,
                "recipe_type": recipe.get("type", "crafting_shaped"),
                "pattern": recipe.get("pattern", []),
                "key": recipe.get("key", {}),
                "breakdown_explanation": (
                    f"Crafting {quantity} {norm_item} requires {batches_needed} craft(s) producing {batches_needed * yield_count} item(s).\n"
                    f"Ingredients required: " + ", ".join(f"{cnt}x {ing}" for ing, cnt in calculated_reqs.items())
                )
            }

        return {
            "target_item": norm_item,
            "target_quantity": quantity,
            "raw_materials": {},
            "breakdown_explanation": f"Recipe details for '{target_item}' not found in standard recipe catalog."
        }

    def parse_calculation_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Detects if query asks for resource calculation and extracts quantity and target item.
        Example: "I need 4 bookshelves. How much paper and leather?"
        """
        patterns = [
            r"(?:need|make|craft|build)\s+(\d+)\s+([a-zA-Z_\s]+?)(?:\.|\?|,|$)",
            r"(?:how (?:much|many))\s+([a-zA-Z_\s]+?)\s+(?:for|to make|to craft)\s+(\d+)\s+([a-zA-Z_\s]+)",
        ]

        # Pattern 1: need 4 bookshelves
        m1 = re.search(r"\b(?:need|make|craft|build|for)\s+(\d+)\s+([a-zA-Z]+(?:_[a-zA-Z]+|\s+[a-zA-Z]+)*)\b", query, re.IGNORECASE)
        if m1:
            qty = int(m1.group(1))
            item = m1.group(2).strip()
            # clean trailing words
            item = re.sub(r"\b(how|what|and|do|i|much|many)\b.*", "", item).strip()
            if item:
                return self.calculate_resource_needs(item, qty)

        return None
