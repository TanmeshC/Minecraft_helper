import os
import re
import json
import ast
import csv
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PRIMARY_PATH = os.path.join("data", "raw", "minecraft_instruction_dataset.json")
RAW_RECIPES_PATH = os.path.join("data", "raw", "recipes.csv")
RAW_WIKI_PATH = os.path.join("data", "raw", "wiki_articles.jsonl")

CLEANED_DIR = os.path.join("data", "cleaned")
PROCESSED_DIR = os.path.join("data", "processed")
EVAL_DIR = os.path.join("data", "evaluation")

INTENT_RULES = [
    ("Resource calculation", r"\b(how much|how many|need|calculate|ratio|total)\b.*\b(wood|planks|ingot|iron|diamond|bookshelf|paper|leather|obsidian|quartz)\b"),
    ("Crafting", r"\b(craft|crafting|recipe|make|recipe for|crafting table|furnace|blast furnace|smoker|anvil|smithing table)\b"),
    ("Troubleshooting", r"\b(why (can't|can not|is not|isn't|won't|does not|doesn't)|not working|not spawning|despawned|glitch|fix|unable to)\b"),
    ("Progression", r"\b(what (should|to) do next|next step|where do i go|progression|game stage|early game|mid game|late game|prepare for)\b"),
    ("Nether", r"\b(nether|fortress|blaze|ghast|piglin|hoglin|bastion|strider|wither skeleton|ancient debris|netherite|crimson|warped)\b"),
    ("End", r"\b(end|ender dragon|elytra|shulker|end city|end ship|end portal|eyes? of ender|stronghold|chorus)\b"),
    ("Villagers", r"\b(villager|iron golem|trade|trading|emerald|profession|job site|lectern|zombie villager|cure|raid|hero of the village)\b"),
    ("Enchanting", r"\b(enchant|enchantment|anvil|grindstone|potion|brewing|mending|unbreaking|fortune|silk touch|sharpness|protection|infinity)\b"),
    ("Farming", r"\b(farm|wheat|carrot|potato|beetroot|seed|crop|bone meal|breed|cow|sheep|pig|chicken|bee|honey|automated farm)\b"),
    ("Redstone", r"\b(redstone|piston|sticky piston|repeater|comparator|observer|hopper|dispenser|dropper|circuit|logic gate|clock)\b"),
    ("Combat", r"\b(combat|sword|shield|bow|crossbow|trident|arrow|damage|armor|creeper|skeleton|zombie|warden|wither|mob attack)\b"),
    ("Mining", r"\b(mine|mining|ore|y level|y=-|branch mine|caving|deepslate|diamond ore|iron ore|coal|gold ore|copper ore|pickaxe)\b"),
    ("Exploration", r"\b(biome|ocean|monument|mansion|desert temple|jungle pyramid|shipwreck|treasure|buried treasure|map|cartographer|structure)\b"),
    ("Survival", r"\b(hunger|food|eat|health|heart|night|sleep|bed|shelter|first day|spawn point|respawn|die|lost my stuff|torch)\b"),
    ("Building", r"\b(build|building|block|wood|stone|slab|stair|wall|fence|glass|decorate|palette|roof|interior)\b"),
]

def classify_intent(question, answer):
    text = (question + " " + answer).lower()
    for intent, pattern in INTENT_RULES:
        if re.search(pattern, text):
            return intent
    return "General"

def classify_difficulty(question, answer, intent):
    text = (question + " " + answer).lower()
    # Expert indicators
    if re.search(r"\b(update suppression|rng manipulation|chunk loader|perimeter|sub-chunk|light suppression|spawning algorithm|bedrock breaking)\b", text):
        return "Expert"
    # Advanced indicators
    if re.search(r"\b(wither|ender dragon|elytra|shulker|bastion|ancient debris|netherite|comparator|logic gate|villager breeder|iron farm|raid farm)\b", text):
        return "Advanced"
    # Intermediate indicators
    if re.search(r"\b(enchant|anvil|brewing|potion|diamond|nether portal|blaze rod|fortress|trading|lectern|piston|repeater|y level)\b", text):
        return "Intermediate"
    # Beginner indicators
    return "Beginner"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Normalize encoding issues
    text = text.replace("\ufffd", "x")
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove leading context prompt leakage
    text = re.sub(r"^(according to the (provided )?text,?\s*|based on the (passage|text),?\s*)", "", text, flags=re.IGNORECASE)
    # Collapse multiple whitespaces
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text

def is_changelog_or_trivia(text):
    pattern = r"\b(snapshot|pre-release|release candidate|1\.\d{1,2}-pre\d?|\b\d{2}w\d{2}[a-z]\b|datafixerupper|optics code|renamed (entity|entities|items?|blocks?)|upgrading path|in the \d{2}w|changelog)\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def is_boilerplate_junk(question, answer):
    q_low = question.lower()
    a_low = answer.lower()
    
    # Generic placeholders
    if q_low in ["prompt", "response", "nan", "test", "question", ""]:
        return True
    if len(question.split()) < 3:
        return True
    
    # Single word or raw hex colors
    if len(answer.split()) <= 2:
        return True
    if re.match(r"^#[0-9a-f]{6}$", a_low):
        return True
    if "------|" in answer or "|---" in answer:
        return True
    if answer.lower() in ["nan", "none", "null", "n/a"]:
        return True
    
    return False

def clean_primary_dataset():
    print(f"Loading raw dataset from {RAW_PRIMARY_PATH}...")
    with open(RAW_PRIMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total raw items: {len(data):,}")

    cleaned_records = []
    dropped_reasons = Counter()
    seen_questions = {}

    for idx, item in enumerate(data):
        raw_q = item.get("instruction")
        raw_a = item.get("output")

        if not isinstance(raw_q, str) or not isinstance(raw_a, str):
            dropped_reasons["non_string_or_nan"] += 1
            continue

        q = clean_text(raw_q)
        a = clean_text(raw_a)

        if is_boilerplate_junk(q, a):
            dropped_reasons["boilerplate_or_too_short"] += 1
            continue

        if is_changelog_or_trivia(q) or is_changelog_or_trivia(a):
            dropped_reasons["changelog_or_snapshot_trivia"] += 1
            continue

        # Canonical key for deduplication
        norm_key = re.sub(r"[^\w\s]", "", q.lower()).strip()
        if norm_key in seen_questions:
            dropped_reasons["duplicate_question"] += 1
            # Keep the longer, more informative answer if the new one is better
            prev_idx = seen_questions[norm_key]
            if len(a.split()) > len(cleaned_records[prev_idx]["output"].split()):
                intent = classify_intent(q, a)
                difficulty = classify_difficulty(q, a, intent)
                cleaned_records[prev_idx] = {
                    "id": f"minementor_qa_{prev_idx:06d}",
                    "instruction": q,
                    "output": a,
                    "intent": intent,
                    "difficulty": difficulty,
                    "source": "Aiwensile2_Filtered_Cleaned"
                }
            continue

        intent = classify_intent(q, a)
        difficulty = classify_difficulty(q, a, intent)

        record_id = f"minementor_qa_{len(cleaned_records):06d}"
        seen_questions[norm_key] = len(cleaned_records)
        cleaned_records.append({
            "id": record_id,
            "instruction": q,
            "output": a,
            "intent": intent,
            "difficulty": difficulty,
            "source": "Aiwensile2_Filtered_Cleaned"
        })

    print("\n--- Data Cleaning Drop Statistics ---")
    for reason, cnt in dropped_reasons.most_common():
        print(f"  Dropped {cnt:,} ({reason})")
    print(f"\nRetained High-Quality Clean Records: {len(cleaned_records):,} ({len(cleaned_records)/len(data)*100:.2f}%)")

    df = pd.DataFrame(cleaned_records)
    print("\nIntent Distribution in Clean Dataset:")
    print(df["intent"].value_counts())
    print("\nDifficulty Distribution in Clean Dataset:")
    print(df["difficulty"].value_counts())

    # Create Train / Val / Test splits (80 / 10 / 10)
    train_df, test_val_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["intent"])
    val_df, test_df = train_test_split(test_val_df, test_size=0.50, random_state=42, stratify=test_val_df["intent"])

    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_path = os.path.join(CLEANED_DIR, "train.parquet")
    val_path = os.path.join(CLEANED_DIR, "val.parquet")
    test_path = os.path.join(CLEANED_DIR, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # Also save JSONL for convenient human inspection
    train_df.head(1000).to_json(os.path.join(CLEANED_DIR, "train_sample_1k.jsonl"), orient="records", lines=True)

    print(f"\nSaved Train split: {len(train_df):,} to {train_path}")
    print(f"Saved Val split:   {len(val_df):,} to {val_path}")
    print(f"Saved Test split:  {len(test_df):,} to {test_path}")

    return {
        "raw_total": len(data),
        "cleaned_total": len(cleaned_records),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "dropped_reasons": dict(dropped_reasons),
        "intent_distribution": df["intent"].value_counts().to_dict(),
        "difficulty_distribution": df["difficulty"].value_counts().to_dict()
    }

def process_recipes():
    print(f"\nProcessing Recipes from {RAW_RECIPES_PATH}...")
    if not os.path.exists(RAW_RECIPES_PATH):
        print("Recipes file not yet available, skipping recipes.")
        return 0

    parsed_recipes = []
    with open(RAW_RECIPES_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            r_type = row.get("type", "")
            raw_result = row.get("result", "")
            raw_key = row.get("key", "")
            raw_pattern = row.get("pattern", "")
            raw_ingredients = row.get("ingredients", "")

            # Parse python string literals to structured dicts safely
            result_item = "unknown"
            result_count = 1
            try:
                if raw_result:
                    r_dict = ast.literal_eval(raw_result)
                    result_item = r_dict.get("item", "").replace("minecraft:", "")
                    result_count = r_dict.get("count", 1)
            except Exception:
                result_item = raw_result

            # Key map
            key_map = {}
            try:
                if raw_key:
                    k_dict = ast.literal_eval(raw_key)
                    for k, v in k_dict.items():
                        if isinstance(v, dict):
                            key_map[k] = v.get("item", "").replace("minecraft:", "")
                        elif isinstance(v, list) and len(v) > 0:
                            key_map[k] = v[0].get("item", "").replace("minecraft:", "")
            except Exception:
                pass

            # Pattern list
            pattern_list = []
            try:
                if raw_pattern:
                    pattern_list = ast.literal_eval(raw_pattern)
            except Exception:
                pattern_list = [raw_pattern]

            # Ingredients list
            ingredients_list = []
            try:
                if raw_ingredients:
                    ing_parsed = ast.literal_eval(raw_ingredients)
                    for ing in ing_parsed:
                        if isinstance(ing, dict):
                            ingredients_list.append(ing.get("item", "").replace("minecraft:", ""))
                        elif isinstance(ing, list) and len(ing) > 0:
                            ingredients_list.append(ing[0].get("item", "").replace("minecraft:", ""))
            except Exception:
                pass

            # Calculate total ingredients required
            ingredient_counts = Counter()
            if pattern_list and key_map:
                for row_str in pattern_list:
                    for char in row_str:
                        if char in key_map:
                            ingredient_counts[key_map[char]] += 1
            elif ingredients_list:
                for ing in ingredients_list:
                    ingredient_counts[ing] += 1

            parsed_recipes.append({
                "recipe_id": f"recipe_{idx:04d}",
                "type": r_type.replace("minecraft:", ""),
                "result_item": result_item,
                "result_count": result_count,
                "pattern": pattern_list,
                "key": key_map,
                "ingredients_summary": dict(ingredient_counts)
            })

    target_recipes = os.path.join(CLEANED_DIR, "recipes.json")
    with open(target_recipes, "w", encoding="utf-8") as f:
        json.dump(parsed_recipes, f, indent=2)
    print(f"Processed and structured {len(parsed_recipes)} crafting recipes saved to {target_recipes}")
    return len(parsed_recipes)

def build_expert_benchmark():
    print("\nBuilding Curated Expert Benchmark Test Dataset (Section 27)...")
    benchmark = [
        # Beginner Questions
        {
            "id": "eval_001",
            "type": "beginner",
            "question": "I just spawned into a new Minecraft world. What should I do first?",
            "ideal_concepts": ["punch tree for wood", "craft wooden planks", "craft crafting table", "wooden pickaxe", "mine stone", "shelter before night", "kill sheep for bed"],
            "difficulty": "Beginner",
            "intent": "Progression",
            "version_dependent": False
        },
        {
            "id": "eval_002",
            "type": "beginner",
            "question": "How do I make a crafting table?",
            "ideal_concepts": ["punch tree for log", "convert 1 wood log to 4 planks", "fill 2x2 grid with 4 planks"],
            "difficulty": "Beginner",
            "intent": "Crafting",
            "version_dependent": False
        },
        # Calculation Questions (Section 24)
        {
            "id": "eval_003",
            "type": "calculation",
            "question": "I want to craft 4 bookshelves. Exactly how much paper and leather do I need in total?",
            "ideal_concepts": ["1 bookshelf = 3 books + 6 planks", "4 bookshelves = 12 books + 24 planks", "1 book = 3 paper + 1 leather", "Total: 36 paper and 12 leather"],
            "calculation_target": {"paper": 36, "leather": 12, "planks": 24},
            "difficulty": "Beginner",
            "intent": "Resource calculation",
            "version_dependent": False
        },
        {
            "id": "eval_004",
            "type": "calculation",
            "question": "I found 6 iron ingots on my first mining trip. What should I make?",
            "ideal_concepts": ["iron pickaxe (3 ingots)", "shield (1 ingot + 6 planks)", "bucket (3 ingots)", "prioritize pickaxe + shield"],
            "difficulty": "Beginner",
            "intent": "Progression",
            "version_dependent": False
        },
        {
            "id": "eval_005",
            "type": "calculation",
            "question": "I have 3 diamonds. What should I craft first?",
            "ideal_concepts": ["diamond pickaxe (requires exactly 3 diamonds)", "allows mining obsidian for nether portal and enchanting table", "do not craft diamond sword or axe first"],
            "difficulty": "Beginner",
            "intent": "Progression",
            "version_dependent": False
        },
        # Version-Specific Questions (Section 17)
        {
            "id": "eval_006",
            "type": "version_specific",
            "question": "How does combat cooldown work in Minecraft?",
            "ideal_concepts": ["Java Edition has attack indicator cooldown since 1.9", "Bedrock Edition does not have attack cooldown (spam clicking works)", "weapon damage differs between Java and Bedrock (axes do high single hit damage in Java)"],
            "difficulty": "Intermediate",
            "intent": "Combat",
            "version_dependent": True,
            "requires_version_clarification": True
        },
        {
            "id": "eval_007",
            "type": "version_specific",
            "question": "At what Y coordinate is diamond ore most common?",
            "ideal_concepts": ["Since 1.18 Caves & Cliffs: Y=-58 or Y=-59", "Pre-1.18: Y=11 or Y=12", "Ore distribution changed drastically in 1.18"],
            "difficulty": "Beginner",
            "intent": "Mining",
            "version_dependent": True
        },
        # Troubleshooting Questions (Section 25)
        {
            "id": "eval_008",
            "type": "troubleshooting",
            "question": "Why won't my villager take a profession?",
            "ideal_concepts": ["check if villager is a nitwit (green coat)", "check if workstation is already claimed by another villager", "check if villager can physically pathfind to workstation", "check trading time of day (working hours)", "check if already traded with (locks profession)"],
            "difficulty": "Intermediate",
            "intent": "Troubleshooting",
            "version_dependent": False
        },
        {
            "id": "eval_009",
            "type": "troubleshooting",
            "question": "Why aren't hostile mobs spawning in my dark mob farm?",
            "ideal_concepts": ["check mob cap (hostile mob cap 70)", "unlit caves within 128 block radius need lighting/slabbing", "player must be between 24 and 128 blocks away", "difficulty set to Peaceful", "spawning blocks must be solid (no glass, half-slabs)"],
            "difficulty": "Advanced",
            "intent": "Troubleshooting",
            "version_dependent": False
        },
        # Multi-Step & Preparation Questions
        {
            "id": "eval_010",
            "type": "multi_step",
            "question": "How do I prepare for the Nether?",
            "ideal_concepts": ["full iron armor or better", "at least 1 piece of gold armor (prevents piglin hostility)", "shield", "bow/crossbow and arrows", "flint and steel or fire charge", "food (steak, bread)", "building blocks (cobblestone won't blow up from ghast fireballs)", "water bucket (for extinguishing before portal, won't place in nether)"],
            "difficulty": "Intermediate",
            "intent": "Nether",
            "version_dependent": False
        },
        # Player-State & Recommendation Questions (Section 12, 15)
        {
            "id": "eval_011",
            "type": "player_state",
            "question": "I have full iron armor, an iron sword, 32 cooked beef, and a diamond pickaxe. What should I do next?",
            "ideal_concepts": ["mine obsidian", "build nether portal", "craft enchanting table (need 2 diamonds, 4 obsidian, 1 book)", "explore Nether for Nether Fortress to get blaze rods"],
            "difficulty": "Intermediate",
            "intent": "Progression",
            "version_dependent": False
        },
        # Trick / Hallucination-Trap Questions
        {
            "id": "eval_012",
            "type": "trick_hallucination_trap",
            "question": "How do I craft an Aether portal with glowstone and a water bucket in vanilla Minecraft?",
            "ideal_concepts": ["Aether portal is not in vanilla Minecraft", "it is a famous community mod (The Aether)", "in vanilla Minecraft, lighting glowstone with water does not create a portal"],
            "difficulty": "Beginner",
            "intent": "Troubleshooting",
            "version_dependent": False
        },
        {
            "id": "eval_013",
            "type": "trick_hallucination_trap",
            "question": "Can I craft chainmail armor using iron ingots in a crafting table?",
            "ideal_concepts": ["chainmail armor cannot be crafted using iron ingots in vanilla Minecraft", "formerly crafted using fire item in old versions", "obtained by trading with armorer villagers, mob drops, or chest loot in dungeons/structures"],
            "difficulty": "Intermediate",
            "intent": "Crafting",
            "version_dependent": False
        }
    ]

    os.makedirs(EVAL_DIR, exist_ok=True)
    benchmark_path = os.path.join(EVAL_DIR, "expert_benchmark.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Created {len(benchmark)} expert-reviewed benchmark questions in {benchmark_path}")
    return len(benchmark)

if __name__ == "__main__":
    summary = clean_primary_dataset()
    recipe_count = process_recipes()
    bench_count = build_expert_benchmark()

    summary["recipes_count"] = recipe_count
    summary["expert_benchmark_count"] = bench_count

    with open(os.path.join(PROCESSED_DIR, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPhase 1 Data Pipeline finished successfully!")
