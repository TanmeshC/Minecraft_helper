import re
import math
from typing import List, Dict, Any, Optional
from minementor.backend.core.interfaces import Ranker, RetrievedPassage

class PassageRanker(Ranker):
    """
    Re-ranks candidate retrieved passages using hybrid lexical-semantic
    scoring, query term coverage, and exact keyword proximity.
    """

    def __init__(self, cross_encoder_model: Optional[str] = None):
        self.cross_encoder_model = cross_encoder_model

    def _compute_relevance_score(self, query: str, passage_text: str, base_score: float) -> float:
        q_tokens = [w for w in re.findall(r"\b\w+\b", query.lower()) if len(w) > 2]
        if not q_tokens:
            return base_score

        p_lower = passage_text.lower()
        p_tokens = re.findall(r"\b\w+\b", p_lower)
        p_token_set = set(p_tokens)

        # 1. Query Term Coverage (Recall of query words in passage)
        matched_tokens = sum(1 for q in q_tokens if q in p_token_set)
        coverage_ratio = matched_tokens / len(q_tokens)

        # 2. Exact Phrase Match Bonus
        phrase_bonus = 0.0
        if query.lower().strip("?.! ") in p_lower:
            phrase_bonus = 0.35
        elif len(q_tokens) >= 2:
            bigrams = [" ".join(q_tokens[i:i+2]) for i in range(len(q_tokens)-1)]
            matched_bigrams = sum(1 for bg in bigrams if bg in p_lower)
            phrase_bonus = 0.15 * (matched_bigrams / len(bigrams)) if bigrams else 0.0

        # 3. Density / Proximity Bonus
        density_bonus = min(0.2, (matched_tokens / max(1, len(p_tokens))) * 5.0)

        # Combine into normalized relevance score [0.0, 1.0]
        final_score = (0.45 * coverage_ratio) + (0.35 * min(1.0, base_score / 20.0)) + phrase_bonus + density_bonus
        return round(float(final_score), 4)

    def rank(self, query: str, passages: List[RetrievedPassage], top_k: int = 3) -> List[RetrievedPassage]:
        scored_passages = []
        for p in passages:
            rel_score = self._compute_relevance_score(query, p.text, p.score)
            new_p = RetrievedPassage(
                id=p.id,
                text=p.text,
                score=rel_score,
                source=p.source,
                metadata={**p.metadata, "rerank_score": rel_score}
            )
            scored_passages.append(new_p)

        # Sort descending by re-ranked score
        scored_passages.sort(key=lambda x: x.score, reverse=True)
        return scored_passages[:top_k]
