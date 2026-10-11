import re
import os
import json
import requests
from typing import List, Dict, Any, Optional, Generator
from minementor.backend.core.interfaces import LLMProvider, RetrievedPassage

class OllamaLLMProvider(LLMProvider):
    """
    Local Ollama LLM provider interfacing with models like llama3.2:3b,
    mistral, or llama3 running locally via Ollama HTTP API.
    """

    def __init__(self, host: str = "http://localhost:11434", default_model: str = "llama3.2:3b"):
        self.host = host.rstrip("/")
        self.default_model = default_model

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            if r.status_code == 200:
                models = r.json().get("models", [])
                return [m["name"] for m in models if "embed" not in m["name"].lower()]
        except Exception:
            pass
        return [self.default_model]

    def _build_system_prompt(
        self,
        difficulty: str = "Beginner",
        player_context: Optional[Dict[str, Any]] = None,
        context: Optional[List[str]] = None,
        deterministic_info: Optional[Dict[str, Any]] = None,
        edition: str = "Java"
    ) -> str:
        sys_parts = [
            "You are MineMentor, an expert, encouraging, and highly knowledgeable AI mentor for Minecraft players, especially beginners.",
            f"The player is at skill level: {difficulty}. The game edition is Minecraft {edition}.",
        ]

        if player_context:
            day = player_context.get("day", 1)
            dim = player_context.get("dimension", "overworld")
            armor = player_context.get("equipment", {}).get("armor", "none")
            resources = player_context.get("resources", {})
            res_str = ", ".join(f"{k}: {v}" for k, v in resources.items() if v > 0) or "minimal"
            sys_parts.append(
                f"PLAYER CONTEXT: Currently on Day {day}, in the {dim.capitalize()}, wearing {armor} armor, with key resources: [{res_str}]."
            )

        if deterministic_info and deterministic_info.get("breakdown_explanation"):
            sys_parts.append(
                f"DETERMINISTIC CRAFTING DATA (Must be used accurately):\n{deterministic_info['breakdown_explanation']}"
            )

        if context and len(context) > 0:
            joined_ctx = "\n---\n".join(context[:3])
            sys_parts.append(
                f"VERIFIED MINECRAFT WIKI KNOWLEDGE:\n{joined_ctx}\n\n"
                "Ground your explanations in the above verified facts where relevant."
            )

        sys_parts.append(
            "CORE GUIDELINES:\n"
            "1. Give structured, beginner-friendly, actionable answers in clean Markdown (bullet points, bold highlights, tips).\n"
            "2. NEVER invent features from community mods as vanilla (e.g. Aether Portal is a mod, not vanilla; chainmail cannot be crafted from iron ingots in vanilla).\n"
            "3. If an answer differs between Java and Bedrock editions, point that out clearly.\n"
            "4. Keep your tone encouraging, direct, and practical."
        )

        return "\n\n".join(sys_parts)

    def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 600,
        difficulty: str = "Beginner",
        player_context: Optional[Dict[str, Any]] = None,
        deterministic_info: Optional[Dict[str, Any]] = None,
        source_title: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        selected_model = model or self.default_model
        sys_prompt = system_instruction or self._build_system_prompt(
            difficulty=difficulty,
            player_context=player_context,
            context=context,
            deterministic_info=deterministic_info,
            edition=player_context.get("edition", "Java") if player_context else "Java"
        )

        payload = {
            "model": selected_model,
            "prompt": prompt,
            "system": sys_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=60)
            if r.status_code == 200:
                resp = r.json().get("response", "").strip()
                if source_title:
                    resp += f"\n\n*Source: {source_title}*"
                return resp
            else:
                return f"Error from Ollama ({r.status_code}): {r.text}"
        except Exception as e:
            return f"Ollama connection error: {e}. Falling back to rule grounded generator."

    def stream_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 600,
        difficulty: str = "Beginner",
        player_context: Optional[Dict[str, Any]] = None,
        deterministic_info: Optional[Dict[str, Any]] = None,
        source_title: Optional[str] = None,
        model: Optional[str] = None
    ) -> Generator[str, None, None]:
        selected_model = model or self.default_model
        sys_prompt = system_instruction or self._build_system_prompt(
            difficulty=difficulty,
            player_context=player_context,
            context=context,
            deterministic_info=deterministic_info,
            edition=player_context.get("edition", "Java") if player_context else "Java"
        )

        payload = {
            "model": selected_model,
            "prompt": prompt,
            "system": sys_prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            r = requests.post(f"{self.host}/api/generate", json=payload, stream=True, timeout=60)
            for line in r.iter_lines():
                if line:
                    chunk_json = json.loads(line)
                    chunk_text = chunk_json.get("response", "")
                    if chunk_text:
                        yield chunk_text
                    if chunk_json.get("done", False):
                        if source_title:
                            yield f"\n\n*Source: {source_title}*"
                        break
        except Exception as e:
            yield f"\n[Streaming error from Ollama: {e}]"


class GroundedRuleLLMProvider(LLMProvider):
    """
    Local zero-cost grounded fallback LLM provider.
    Synthesizes facts directly from retrieved passages, deterministic calculations,
    and structured mechanics while strictly preventing hallucinations.
    """

    def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        difficulty: str = "Beginner",
        player_context: Optional[Dict[str, Any]] = None,
        deterministic_info: Optional[Dict[str, Any]] = None,
        source_title: Optional[str] = None
    ) -> str:
        q_low = prompt.lower()

        # 1. Deterministic Calculation Output
        if deterministic_info and deterministic_info.get("raw_materials"):
            return deterministic_info.get("breakdown_explanation", "")

        # 2. Trick Question & Community Mod Trap Detection
        if "aether portal" in q_low and "glowstone" in q_low:
            return (
                "⚠️ **Vanilla Minecraft Clarification:**\n\n"
                "The **Aether Portal** (built with glowstone and lit with a water bucket) is from a famous community mod (*The Aether*), "
                "and does **not exist in vanilla Minecraft**. If you place water on a glowstone frame in vanilla Minecraft, the water will simply flow out and no portal will open."
            )
        if "chainmail" in q_low and ("craft" in q_low or "make" in q_low):
            return (
                "🛡️ **Crafting Clarification:**\n\n"
                "Chainmail armor **cannot be crafted using iron ingots** in vanilla Minecraft. "
                "Historically, it required the unobtainable 'Fire' item. You can obtain chainmail armor by:\n"
                "1. Trading with Armorer villagers.\n"
                "2. Killing mobs wearing chainmail (rare drop).\n"
                "3. Looting chests in Dungeons, Buried Treasure, or Nether Bastions."
            )

        # 3. Grounding from Retrieved Knowledge Passages
        if context and len(context) > 0:
            relevant_snippets = []
            for ctx in context:
                lines = [l.strip() for l in ctx.split("\n") if l.strip() and not l.startswith("Title:")]
                for l in lines:
                    if len(l) > 40:
                        relevant_snippets.append(l)

            if relevant_snippets:
                lead_snippet = relevant_snippets[0]
                if difficulty == "Beginner":
                    explanation = (
                        f"{lead_snippet}\n\n"
                        f"💡 **Beginner Tip:** Take your time, make sure your area is well-lit with torches, "
                        f"and keep a bed nearby to set your spawn point."
                    )
                else:
                    explanation = (
                        f"{lead_snippet}\n\n"
                        f"⚙️ **Advanced Mechanics Note:** Ensure proper chunk alignment and mob-cap management when scaling this mechanic."
                    )

                if source_title:
                    explanation += f"\n\n*Source: {source_title}*"
                return explanation

        # 4. Fallback
        if "version" in q_low or "difference" in q_low or "cooldown" in q_low:
            return (
                "Which Minecraft edition and version are you playing? "
                "This mechanic differs significantly between **Java Edition** (e.g. weapon attack cooldowns and axe damage) "
                "and **Bedrock Edition** (instant attack speed and shield delay)."
            )

        return (
            "I don't have enough reliable, grounded information in my Minecraft knowledge base to answer that confidently. "
            "Could you specify which Minecraft version or item you are referring to?"
        )
