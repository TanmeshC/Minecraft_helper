import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest

from minementor.knowledge.progression_graph import MinecraftProgressionGraph
from minementor.backend.services.deterministic_engine import DeterministicCalculationEngine
from minementor.player.player_state import PersistentPlayerModel
from minementor.recommendation.engine import ContextualRecommendationEngine
from minementor.rag.retrieval.hybrid_retriever import HybridRetriever
from minementor.rag.reranking.ranker import PassageRanker
from minementor.backend.services.mentor_orchestrator import MineMentorOrchestrator

class TestMineMentor(unittest.TestCase):

    def setUp(self):
        self.progression_graph = MinecraftProgressionGraph()
        self.calc_engine = DeterministicCalculationEngine()
        self.player_model = PersistentPlayerModel()
        self.recommendation_engine = ContextualRecommendationEngine(self.progression_graph)
        self.retriever = HybridRetriever()
        self.ranker = PassageRanker()
        self.orchestrator = MineMentorOrchestrator()

    def test_progression_graph_nodes(self):
        self.assertIn("punch_wood", self.progression_graph.nodes)
        self.assertIn("ender_dragon", self.progression_graph.nodes)
        prereqs = self.progression_graph.get_prerequisites("wooden_pickaxe")
        self.assertEqual(prereqs, ["crafting_table"])

    def test_deterministic_bookshelf_calculation(self):
        calc = self.calc_engine.calculate_resource_needs("bookshelf", 4)
        raw = calc["raw_materials"]
        self.assertEqual(raw["paper"], 36)
        self.assertEqual(raw["leather"], 12)
        self.assertEqual(raw["wood_planks"], 24)

    def test_player_context_extraction(self):
        query = "I have 5 diamonds, iron armor, 32 bread and I'm on day 8."
        extracted = self.player_model.extract_state_from_query(query)
        self.assertEqual(extracted.get("diamond"), 5)
        self.assertEqual(extracted.get("armor"), "iron")
        self.assertEqual(extracted.get("day"), 8)

    def test_recommendation_prioritization(self):
        state = {
            "resources": {"diamond": 3},
            "inventory": {"diamond": 3},
            "equipment": {"armor": "iron"},
            "dimension": "overworld",
            "completed_milestones": ["iron_age"]
        }
        recommendations = self.recommendation_engine.recommend_next_steps(state)
        self.assertGreater(len(recommendations), 0)
        # Goal 1 should be crafting diamond pickaxe
        goal_titles = [g["title"] for g in recommendations]
        self.assertIn("Craft Diamond Pickaxe", goal_titles)

    def test_retriever_bm25(self):
        passages = self.retriever.retrieve("crafting table", top_k=3)
        self.assertGreater(len(passages), 0)
        self.assertTrue(any("crafting" in p.text.lower() for p in passages))

    def test_orchestrator_hallucination_trap(self):
        res = self.orchestrator.process_query("How do I make an Aether portal with glowstone and water in vanilla?")
        self.assertIn("not exist in vanilla", res["response"].lower())

    def test_orchestrator_calculation_query(self):
        res = self.orchestrator.process_query("I need 4 bookshelves. How much paper and leather?")
        self.assertEqual(res["intent"], "Resource calculation")
        self.assertIsNotNone(res["deterministic_calculation"])

if __name__ == "__main__":
    unittest.main()
