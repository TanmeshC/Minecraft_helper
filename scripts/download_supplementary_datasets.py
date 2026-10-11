import os
import requests
from tqdm import tqdm

RECIPES_URL = "https://huggingface.co/datasets/ProfQu/MinecraftRecipes/raw/main/recipes.csv"
RECIPES_TARGET = os.path.join("data", "raw", "recipes.csv")

WIKI_URL = "https://huggingface.co/datasets/KoalaBrainResearcher/Minecraft-Wiki-2023/resolve/main/articles.jsonl"
WIKI_TARGET = os.path.join("data", "raw", "wiki_articles.jsonl")

def download(url, target_path, desc):
    if os.path.exists(target_path):
        print(f"{desc} already exists at {target_path} ({os.path.getsize(target_path):,} bytes)")
        return
    print(f"Downloading {desc} from {url}...")
    headers = {"User-Agent": "MineMentor-Pipeline/1.0"}
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    r.raise_for_status()
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024 * 1024
    with open(target_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
        for chunk in r.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Completed {desc}: {os.path.getsize(target_path):,} bytes")

if __name__ == "__main__":
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    download(RECIPES_URL, RECIPES_TARGET, "ProfQu Recipes")
    download(WIKI_URL, WIKI_TARGET, "KoalaBrain Wiki Articles")
