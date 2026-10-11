import requests
import json

def search_datasets():
    print("Searching Hugging Face for Minecraft datasets...")
    url = "https://huggingface.co/api/datasets?search=minecraft&limit=100"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error: {r.status_code}")
        return
    datasets = r.json()
    print(f"Total datasets found with query 'minecraft': {len(datasets)}")
    
    interesting_candidates = [
        "Aiwensile2/Minecraft_QA-pairs_Instruction_Dataset",
        "naklecha/minecraft-question-answer-700k",
        "PJMixers/naklecha_minecraft-question-answer-700k-ShareGPT",
        "KoalaBrainResearcher/Minecraft-Wiki-2023",
        "ProfQu/MinecraftRecipes",
        "OVAWARE/Minecrafter",
        "SDuttagupta/dataset_minecraft_small",
        "declip/Minecraft-Server-Chat",
        "CraftJarvis/minecraft-vla-sft",
        "CraftJarvis/minecraft-vlp"
    ]
    
    results = []
    for d in datasets:
        ds_id = d.get("id")
        if ds_id in interesting_candidates or any(k in ds_id.lower() for k in ["wiki", "recipe", "craft", "qa", "question", "instruction", "chat"]):
            results.append({
                "id": ds_id,
                "downloads": d.get("downloads", 0),
                "likes": d.get("likes", 0),
                "tags": d.get("tags", []),
                "description": d.get("description", "")
            })
            
    print(f"\nFiltered {len(results)} relevant candidate datasets:")
    for res in sorted(results, key=lambda x: x["downloads"], reverse=True):
        print(f"ID: {res['id']}")
        print(f"  Downloads: {res['downloads']}, Likes: {res['likes']}")
        print(f"  Tags: {res['tags'][:6]}")
        print("-" * 60)

    # Let's also inspect specific repos
    for cand in ["naklecha/minecraft-question-answer-700k", "KoalaBrainResearcher/Minecraft-Wiki-2023", "ProfQu/MinecraftRecipes", "OVAWARE/Minecrafter"]:
        print(f"\nInspecting details for: {cand}")
        det_url = f"https://huggingface.co/api/datasets/{cand}"
        det_r = requests.get(det_url)
        if det_r.status_code == 200:
            det = det_r.json()
            siblings = [s['rfilename'] for s in det.get('siblings', [])]
            card_data = det.get('cardData', {})
            print(f"  Siblings: {siblings}")
            print(f"  CardData: {card_data}")
        else:
            print(f"  Failed to fetch: {det_r.status_code}")

if __name__ == "__main__":
    search_datasets()
