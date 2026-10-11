import os
import json
import re
from typing import List, Dict, Any

WIKI_RAW_PATH = os.path.join("data", "raw", "wiki_articles.jsonl")
CHUNKS_OUTPUT_PATH = os.path.join("data", "cleaned", "knowledge_chunks.jsonl")

def clean_wiki_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove mediawiki templates like {{...}}
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    # Remove mediawiki file/image links like [[File:...]]
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    # Clean mediawiki links [[Target|Label]] -> Label
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    # Remove external links [http... label]
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    # Remove table markup
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.DOTALL)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def chunk_text(text: str, title: str, chunk_size: int = 300, overlap: int = 50) -> List[Dict[str, Any]]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        # Prepend title context for stronger retrieval grounding
        grounded_text = f"Title: {title}\nContent: {chunk_str}"

        chunks.append({
            "chunk_id": f"{re.sub(r'[^a-zA-Z0-9_]', '_', title)[:30]}_{chunk_index:03d}",
            "title": title,
            "text": grounded_text,
            "raw_content": chunk_str,
            "word_count": len(chunk_words),
            "source": f"Minecraft Wiki: {title}"
        })

        chunk_index += 1
        if end >= len(words):
            break
        start += (chunk_size - overlap)

    return chunks

def ingest_wiki_knowledge(max_articles: int = 2500):
    print(f"Ingesting Minecraft Wiki knowledge from {WIKI_RAW_PATH}...")
    if not os.path.exists(WIKI_RAW_PATH):
        print(f"Error: {WIKI_RAW_PATH} not found.")
        return 0

    total_chunks = 0
    os.makedirs(os.path.dirname(CHUNKS_OUTPUT_PATH), exist_ok=True)

    with open(WIKI_RAW_PATH, "r", encoding="utf-8") as in_f, open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        article_count = 0
        for line in in_f:
            if not line.strip():
                continue
            try:
                article = json.loads(line)
            except Exception:
                continue

            title = article.get("title", "").strip()
            raw_text = article.get("text", "").strip()

            # Skip redirect pages, user pages, talk pages, server hosting or developer bios
            if not title or not raw_text or len(raw_text) < 120:
                continue
            if any(title.startswith(prefix) for prefix in ["User:", "Talk:", "Template:", "Module:", "File:", "MediaWiki:", "Category:"]):
                continue
            # Blacklist server hosting, IT administration, and non-gameplay trivia
            low_title = title.lower()
            if any(term in low_title for term in [
                "setting up a server", "server software", "server list", "hosting", "port forward", 
                "markus persson", "jens bergensten", "daniel rosenfeld", "c418", "notch", 
                "spigot", "bukkit", "paper", "bungeecord", "velocity", "rcon", "dvd", "merchandise"
            ]):
                continue

            cleaned = clean_wiki_text(raw_text)
            if len(cleaned.split()) < 25:
                continue

            chunks = chunk_text(cleaned, title)
            for c in chunks:
                out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
                total_chunks += 1

            article_count += 1
            if article_count >= max_articles:
                break

    print(f"Processed {article_count:,} articles into {total_chunks:,} semantic knowledge chunks.")
    print(f"Chunks saved to {CHUNKS_OUTPUT_PATH}")
    return total_chunks

if __name__ == "__main__":
    ingest_wiki_knowledge()
