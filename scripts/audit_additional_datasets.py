import requests
import json
import csv
import io

def audit_recipes():
    print("=== Auditing ProfQu/MinecraftRecipes ===")
    url = "https://huggingface.co/datasets/ProfQu/MinecraftRecipes/raw/main/recipes.csv"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        lines = r.text.strip().splitlines()
        reader = list(csv.DictReader(lines))
        print(f"Total recipe rows: {len(reader)}")
        types = set(row['type'] for row in reader)
        print(f"Recipe types: {types}")
        print("Sample row:")
        print(reader[0])
    else:
        print(f"Failed to fetch recipes: {r.status_code}")

def audit_wiki_sample():
    print("\n=== Auditing KoalaBrainResearcher/Minecraft-Wiki-2023 (Streaming sample) ===")
    url = "https://huggingface.co/datasets/KoalaBrainResearcher/Minecraft-Wiki-2023/resolve/main/articles.jsonl"
    r = requests.get(url, headers={"Range": "bytes=0-100000"}, timeout=30)
    if r.status_code in [200, 206]:
        content = r.content.decode("utf-8", errors="ignore")
        lines = [l for l in content.split("\n") if l.strip()]
        valid_json = []
        for l in lines[:-1]: # exclude potential partial line
            try:
                valid_json.append(json.loads(l))
            except Exception:
                pass
        print(f"Successfully decoded {len(valid_json)} sample wiki articles from byte stream")
        if valid_json:
            first = valid_json[0]
            print(f"Keys: {list(first.keys())}")
            print(f"Title: {first.get('title')}")
            print(f"Text length: {len(first.get('text', ''))} chars")
            print(f"Sample snippet: {first.get('text', '')[:300]}...")
    else:
        print(f"Failed to fetch wiki sample: {r.status_code}")

def audit_naklecha_sample():
    print("\n=== Auditing naklecha/minecraft-question-answer-700k (Streaming sample) ===")
    url = "https://huggingface.co/datasets/naklecha/minecraft-question-answer-700k/resolve/main/train.json"
    r = requests.get(url, headers={"Range": "bytes=0-50000"}, timeout=30)
    if r.status_code in [200, 206]:
        content = r.content.decode("utf-8", errors="ignore")
        print("Raw byte chunk sample (first 400 chars):")
        print(content[:400])
    else:
        print(f"Failed to fetch naklecha sample: {r.status_code}")

if __name__ == "__main__":
    audit_recipes()
    audit_wiki_sample()
    audit_naklecha_sample()
