from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RetrievedPassage:
    id: str
    text: str
    score: float
    source: str
    metadata: Dict[str, Any]

@dataclass
class ClassificationResult:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

class LLMProvider(ABC):
    """Abstract interface for LLM backends (OpenAI, Gemini, Anthropic, local/mock)."""

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512
    ) -> str:
        pass

class EmbeddingProvider(ABC):
    """Abstract interface for dense embedding generation."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

class Classifier(ABC):
    """Abstract interface for text classification (Intent, Difficulty)."""

    @abstractmethod
    def predict(self, text: str) -> ClassificationResult:
        pass

    @abstractmethod
    def batch_predict(self, texts: List[str]) -> List[ClassificationResult]:
        pass

class Ranker(ABC):
    """Abstract interface for passage re-ranking."""

    @abstractmethod
    def rank(self, query: str, passages: List[RetrievedPassage], top_k: int = 3) -> List[RetrievedPassage]:
        pass

class Retriever(ABC):
    """Abstract interface for document/knowledge retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        pass

class VectorStore(ABC):
    """Abstract interface for vector similarity search."""

    @abstractmethod
    def add_documents(self, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievedPassage]:
        pass

class KnowledgeSource(ABC):
    """Abstract interface for knowledge ingestion sources."""

    @abstractmethod
    def load_documents(self) -> List[Dict[str, Any]]:
        pass

class PlayerModel(ABC):
    """Abstract interface for player state tracking."""

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def extract_state_from_query(self, query: str) -> Dict[str, Any]:
        pass

class RecommendationEngine(ABC):
    """Abstract interface for next objective recommendations."""

    @abstractmethod
    def recommend_next_steps(self, player_state: Dict[str, Any], query: Optional[str] = None) -> List[Dict[str, Any]]:
        pass
