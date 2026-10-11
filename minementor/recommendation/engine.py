from typing import Dict, Any, List, Optional
from minementor.backend.core.interfaces import RecommendationEngine
from minementor.knowledge.progression_graph import MinecraftProgressionGraph

class ContextualRecommendationEngine(RecommendationEngine):
    """
    Ranks and recommends next goals using player state and the structured progression graph.
    """

    def __init__(self, progression_graph: Optional[MinecraftProgressionGraph] = None):
        self.progression_graph = progression_graph or MinecraftProgressionGraph()

    def recommend_next_steps(self, player_state: Dict[str, Any], query: Optional[str] = None) -> List[Dict[str, Any]]:
        resources = player_state.get("resources", {})
        inventory = player_state.get("inventory", {})
        equipment = player_state.get("equipment", {})
        dimension = player_state.get("dimension", "overworld")
        day = player_state.get("day", 1)
        completed = player_state.get("completed_milestones", [])

        # Auto-infer completed nodes if not explicitly logged
        completed_set = set(completed)
        if resources.get("diamond", 0) > 0 or inventory.get("diamond", 0) > 0:
            completed_set.update(["punch_wood", "crafting_table", "wooden_pickaxe", "stone_age", "stone_tools_and_furnace", "iron_age", "diamonds"])
        elif resources.get("iron", 0) > 0 or equipment.get("armor") == "iron":
            completed_set.update(["punch_wood", "crafting_table", "wooden_pickaxe", "stone_age", "stone_tools_and_furnace", "iron_age"])

        if equipment.get("armor") in ["diamond", "netherite"]:
            completed_set.update(["diamond_pickaxe", "diamonds"])

        if dimension == "nether":
            completed_set.update(["obsidian_and_nether_portal", "diamond_pickaxe"])
        elif dimension == "end":
            completed_set.update(["obsidian_and_nether_portal", "nether_fortress", "eyes_of_ender", "stronghold"])

        available_steps = self.progression_graph.get_available_next_steps(list(completed_set))

        # Score and rank candidates based on state context
        scored_recommendations = []
        for step in available_steps:
            score = 100 - (step.get("tier", 0) * 5)
            rationale = []

            # Prioritization heuristics
            if step["id"] == "iron_gear_and_shield":
                if resources.get("iron", 0) >= 3:
                    score += 25
                    rationale.append("You already have iron ingots ready to craft equipment.")
                if equipment.get("armor") == "none":
                    score += 30
                    rationale.append("You currently have no armor, leaving you vulnerable to hostile mobs.")

            elif step["id"] == "diamonds":
                if equipment.get("armor") == "iron":
                    score += 20
                    rationale.append("You have iron armor, making it safe to venture down to Y=-58 for deep diamond mining.")

            elif step["id"] == "diamond_pickaxe":
                if resources.get("diamond", 0) >= 3 or inventory.get("diamond", 0) >= 3:
                    score += 40
                    rationale.append("You have 3+ diamonds — crafting a diamond pickaxe is the #1 priority to unlock obsidian.")

            elif step["id"] == "obsidian_and_nether_portal":
                if inventory.get("diamond_pickaxe", 0) > 0 or "diamond_pickaxe" in completed_set:
                    score += 25
                    rationale.append("You have a diamond pickaxe and can harvest obsidian for your Nether Portal.")

            elif step["id"] == "enchanting_setup":
                if resources.get("diamond", 0) >= 2 and inventory.get("obsidian", 0) >= 4:
                    score += 30
                    rationale.append("You have the materials required for an Enchanting Table.")

            elif step["id"] == "nether_fortress":
                if dimension == "nether":
                    score += 35
                    rationale.append("You are currently inside the Nether — search for a Fortress to acquire Blaze Rods.")

            scored_recommendations.append({
                "goal_id": step["id"],
                "title": step["name"],
                "tier": step["tier"],
                "category": step["category"],
                "priority_score": score,
                "description": step["description"],
                "rationale": " ".join(rationale) if rationale else step["description"]
            })

        # Sort by priority score descending
        scored_recommendations.sort(key=lambda x: x["priority_score"], reverse=True)
        return scored_recommendations[:3]
