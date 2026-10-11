import os
import joblib
from typing import Dict, Any, List, Optional, Generator

from minementor.player.player_state import PersistentPlayerModel
from minementor.recommendation.engine import ContextualRecommendationEngine
from minementor.backend.services.deterministic_engine import DeterministicCalculationEngine
from minementor.rag.retrieval.hybrid_retriever import HybridRetriever
from minementor.rag.reranking.ranker import PassageRanker
from minementor.backend.services.llm_provider import GroundedRuleLLMProvider, OllamaLLMProvider

class MineMentorOrchestrator:
    """
    Main End-to-End Intelligence Pipeline implementing Section 9 Core Architecture.
    Integrates ML Classifiers, Deterministic Math, Player State,
    Hybrid RAG Knowledge Retrieval, and Local Ollama / Grounded LLM providers.
    """

    def __init__(
        self,
        intent_model_path: Optional[str] = None,
        difficulty_model_path: Optional[str] = None,
        ollama_model: str = "llama3.2:3b"
    ):
        # 1. Load ML Classifiers
        self.intent_model_path = intent_model_path or os.path.join("minementor", "ml", "classification", "intent_model.joblib")
        self.difficulty_model_path = difficulty_model_path or os.path.join("minementor", "ml", "classification", "difficulty_model.joblib")
        
        self.intent_pipeline = None
        self.difficulty_pipeline = None
        self._load_classifiers()

        # 2. Player Model & Recommendation Engine
        self.player_model = PersistentPlayerModel()
        self.recommendation_engine = ContextualRecommendationEngine()
        self.progression_graph = self.recommendation_engine.progression_graph

        # 3. Deterministic Calculation Engine
        self.calc_engine = DeterministicCalculationEngine()

        # 4. Hybrid Knowledge Retriever & Ranker
        self.retriever = HybridRetriever()
        self.ranker = PassageRanker()

        # 5. LLM Providers
        self.ollama_provider = OllamaLLMProvider(default_model=ollama_model)
        self.rule_provider = GroundedRuleLLMProvider()
        self.active_provider_name = "ollama" if self.ollama_provider.is_available() else "rules"

    def _load_classifiers(self):
        if os.path.exists(self.intent_model_path):
            try:
                self.intent_pipeline = joblib.load(self.intent_model_path)
            except Exception as e:
                print(f"Warning: could not load intent model: {e}")
        if os.path.exists(self.difficulty_model_path):
            try:
                self.difficulty_pipeline = joblib.load(self.difficulty_model_path)
            except Exception as e:
                print(f"Warning: could not load difficulty model: {e}")

    def prepare_query_context(self, user_query: str) -> Dict[str, Any]:
        """
        Runs Stage 1 to 3: Query understanding, entity extraction, calculations, and RAG retrieval.
        """
        # Step 1: Context Extraction & Player State Update
        extracted_entities = self.player_model.extract_state_from_query(user_query)
        current_state = self.player_model.get_state()

        # Step 2: Intent Classification
        predicted_intent = "General"
        intent_conf = 1.0
        if self.intent_pipeline:
            predicted_intent = str(self.intent_pipeline.predict([user_query])[0])
            if hasattr(self.intent_pipeline, "predict_proba"):
                probs = self.intent_pipeline.predict_proba([user_query])[0]
                intent_conf = float(max(probs))

        # Step 3: Difficulty Classification
        predicted_difficulty = "Beginner"
        if self.difficulty_pipeline:
            predicted_difficulty = str(self.difficulty_pipeline.predict([user_query])[0])

        # Step 4: Progression Recommendations
        suggested_steps = []
        q_low = user_query.lower()
        if predicted_intent == "Progression" or any(phrase in q_low for phrase in ["what should i do", "what now", "where do i go", "what next"]):
            suggested_steps = self.recommendation_engine.recommend_next_steps(current_state, user_query)

        # Step 5: Deterministic Calculation Engine
        calc_result = self.calc_engine.parse_calculation_query(user_query)

        # Step 6: Hybrid Knowledge Retrieval
        retrieved_passages = []
        reranked_passages = []
        if not calc_result:
            retrieved_passages = self.retriever.retrieve(user_query, top_k=6)
            if retrieved_passages:
                reranked_passages = self.ranker.rank(user_query, retrieved_passages, top_k=3)

        return {
            "user_query": user_query,
            "intent": predicted_intent,
            "intent_confidence": round(intent_conf, 4),
            "difficulty": predicted_difficulty,
            "extracted_state": extracted_entities,
            "player_state": current_state,
            "recommended_next_goals": suggested_steps,
            "deterministic_calculation": calc_result,
            "reranked_passages": reranked_passages,
            "sources": [
                {
                    "id": p.id,
                    "title": p.metadata.get("title", ""),
                    "score": round(p.score, 4),
                    "source": p.source,
                    "text_snippet": p.text[:200] + "..."
                } for p in reranked_passages
            ]
        }

    def process_query(
        self,
        user_query: str,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete pipeline synchronously and returns synthesized response.
        """
        ctx_bundle = self.prepare_query_context(user_query)
        reranked_passages = ctx_bundle["reranked_passages"]
        context_snippets = [p.text for p in reranked_passages] if reranked_passages else []
        primary_source = reranked_passages[0].source if reranked_passages else None
        calc_result = ctx_bundle["deterministic_calculation"]
        suggested_steps = ctx_bundle["recommended_next_goals"]
        current_state = ctx_bundle["player_state"]

        use_provider = provider or self.active_provider_name

        # If user explicitly asked for progression recommendations and no calc
        if suggested_steps and not calc_result and any(phrase in user_query.lower() for phrase in ["what should i do next", "what now", "where do i go"]):
            rec_texts = [f"**Goal {i+1}: {step['title']}**\n{step['rationale']}" for i, step in enumerate(suggested_steps)]
            response_text = (
                f"Based on your current progression (Day {current_state.get('day', 1)}, "
                f"{current_state.get('equipment', {}).get('armor', 'no')} armor in the {current_state.get('dimension', 'overworld')}):\n\n"
                + "\n\n".join(rec_texts)
            )
        elif use_provider == "ollama" and self.ollama_provider.is_available():
            response_text = self.ollama_provider.generate_response(
                prompt=user_query,
                context=context_snippets,
                difficulty=ctx_bundle["difficulty"],
                player_context=current_state,
                deterministic_info=calc_result,
                source_title=primary_source,
                model=model
            )
        else:
            response_text = self.rule_provider.generate_response(
                prompt=user_query,
                context=context_snippets,
                difficulty=ctx_bundle["difficulty"],
                player_context=current_state,
                deterministic_info=calc_result,
                source_title=primary_source
            )

        ctx_bundle["response"] = response_text
        ctx_bundle["provider_used"] = use_provider
        return ctx_bundle

    def stream_query(
        self,
        user_query: str,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Streams generated response tokens in real-time.
        """
        ctx_bundle = self.prepare_query_context(user_query)
        reranked_passages = ctx_bundle["reranked_passages"]
        context_snippets = [p.text for p in reranked_passages] if reranked_passages else []
        primary_source = reranked_passages[0].source if reranked_passages else None
        calc_result = ctx_bundle["deterministic_calculation"]
        current_state = ctx_bundle["player_state"]

        use_provider = provider or self.active_provider_name

        if use_provider == "ollama" and self.ollama_provider.is_available() and not calc_result:
            for chunk in self.ollama_provider.stream_response(
                prompt=user_query,
                context=context_snippets,
                difficulty=ctx_bundle["difficulty"],
                player_context=current_state,
                deterministic_info=calc_result,
                source_title=primary_source,
                model=model
            ):
                yield chunk
        else:
            full_res = self.process_query(user_query, provider=provider, model=model)["response"]
            yield full_res
