import os
import json
import re
from collections import Counter, defaultdict
import numpy as np

DATA_PATH = os.path.join("data", "raw", "minecraft_instruction_dataset.json")
REPORT_DATA_PATH = os.path.join("data", "processed", "audit_summary.json")

def analyze_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    print("Loading primary dataset into memory...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_records = len(data)
    print(f"Total raw records loaded: {total_records:,}")

    # Inspect schema
    field_counts = Counter()
    for record in data:
        keys = tuple(sorted(record.keys()))
        field_counts[keys] += 1

    print("\nSchema variants found:")
    for schema, count in field_counts.items():
        print(f"  Fields {schema}: {count:,} records ({count/total_records*100:.2f}%)")

    # Type & missing value analysis
    non_string_types = Counter()
    missing_fields = Counter()
    empty_instructions = 0
    empty_outputs = 0
    non_empty_inputs = 0

    instructions = []
    inputs = []
    outputs = []

    inst_len_chars = []
    inst_len_words = []
    out_len_chars = []
    out_len_words = []

    # Category heuristic keywords
    categories = Counter()
    category_patterns = {
        "Crafting & Recipes": r"\b(craft|recipe|make|create|table|furnace|forge|brewing)\b",
        "Mining & Resources": r"\b(mine|ore|diamond|iron|gold|coal|ancient debris|netherite|copper|redstone|lapis|pickaxe)\b",
        "Combat & Mobs": r"\b(mob|zombie|skeleton|creeper|enderman|warden|wither|dragon|attack|damage|sword|shield|bow|arrow|hostile|passive)\b",
        "Survival & Basics": r"\b(hunger|food|health|eat|night|sleep|bed|shelter|torch|die|spawn|respawn|hardcore)\b",
        "Exploration & Biomes": r"\b(biome|village|desert|jungle|ocean|structure|temple|mansion|stronghold|portal|ruins|monument|map)\b",
        "Nether & End": r"\b(nether|fortress|blaze|ghast|piglin|bastion|end|elytra|shulker|ender dragon|end city)\b",
        "Farming & Breeding": r"\b(farm|wheat|crop|seed|breed|cow|sheep|pig|chicken|villager|iron golem|trade|emerald)\b",
        "Redstone & Mechanics": r"\b(redstone|piston|repeater|comparator|observer|hopper|dispenser|dropper|circuit|contraption|tick)\b",
        "Enchanting & Potions": r"\b(enchant|anvil|grindstone|potion|brewing|splash|lingering|mending|unbreaking|fortune|sharpness)\b",
        "Building & Blocks": r"\b(block|wood|stone|slab|stair|fence|glass|decorate|build|house|wall|roof)\b",
        "Commands & Technical": r"\b(command|gamerule|datapack|mod|plugin|seed|coordinate|f3|hitbox|chunk)\b"
    }

    # Version patterns
    version_patterns = {
        "Java Edition": r"\b(java|java edition)\b",
        "Bedrock Edition": r"\b(bedrock|bedrock edition|pocket edition|mcpe|windows 10)\b",
        "Specific Version Number": r"\b(1\.\d{1,2}(\.\d{1,2})?)\b",
        "Recent 1.20+": r"\b(1\.20|1\.21|sniffer|camel|crafter|armadillo|breeze|trial chamber)\b",
        "Old (<1.13)": r"\b(1\.8|1\.9|1\.10|1\.11|1\.12)\b",
        "Combat Update 1.9": r"\b(1\.9|combat update|attack cooldown|offhand|shield cooldown)\b"
    }
    version_counts = Counter()
    answer_types = Counter()

    invalid_records = []

    for idx, r in enumerate(data):
        raw_inst = r.get("instruction")
        raw_inp = r.get("input")
        raw_out = r.get("output")

        # Check types
        if not isinstance(raw_inst, str):
            non_string_types[f"instruction_{type(raw_inst).__name__}"] += 1
            invalid_records.append((idx, "instruction", raw_inst))
            inst = str(raw_inst) if raw_inst is not None else ""
        else:
            inst = raw_inst.strip()

        if not isinstance(raw_inp, str):
            non_string_types[f"input_{type(raw_inp).__name__}"] += 1
            inp = str(raw_inp) if raw_inp is not None else ""
        else:
            inp = raw_inp.strip()

        if not isinstance(raw_out, str):
            non_string_types[f"output_{type(raw_out).__name__}"] += 1
            invalid_records.append((idx, "output", raw_out))
            out = str(raw_out) if raw_out is not None else ""
        else:
            out = raw_out.strip()

        if not inst or inst.lower() == "nan":
            empty_instructions += 1
        if not out or out.lower() == "nan":
            empty_outputs += 1
        if inp and inp.lower() != "nan":
            non_empty_inputs += 1

        instructions.append(inst)
        inputs.append(inp)
        outputs.append(out)

        inst_words = len(inst.split())
        out_words = len(out.split())
        inst_len_chars.append(len(inst))
        inst_len_words.append(inst_words)
        out_len_chars.append(len(out))
        out_len_words.append(out_words)

        # Answer style classification
        out_lower = out.lower().strip()
        if out_lower in ["true", "false", "true.", "false.", "yes", "no", "yes.", "no."]:
            answer_types["Boolean / Yes-No"] += 1
        elif out_words <= 5:
            answer_types["Short Answer (<=5 words)"] += 1
        elif out_words <= 30:
            answer_types["Medium Answer (6-30 words)"] += 1
        else:
            answer_types["Long Answer (>30 words)"] += 1

        # Check category
        text_lower = (inst + " " + out).lower()
        matched_cat = False
        for cat_name, pattern in category_patterns.items():
            if re.search(pattern, text_lower):
                categories[cat_name] += 1
                matched_cat = True
        if not matched_cat:
            categories["Other / Uncategorized"] += 1

        # Check version mentions
        for ver_name, pattern in version_patterns.items():
            if re.search(pattern, text_lower):
                version_counts[ver_name] += 1

    print("\n--- Data Integrity & Types ---")
    print(f"Total entries: {total_records:,}")
    print(f"Non-string field occurrences: {dict(non_string_types)}")
    print(f"Sample invalid records (first 5): {invalid_records[:5]}")
    print(f"Empty or 'NaN' instructions: {empty_instructions:,} ({empty_instructions/total_records*100:.4f}%)")
    print(f"Empty or 'NaN' outputs: {empty_outputs:,} ({empty_outputs/total_records*100:.4f}%)")
    print(f"Non-empty 'input' field entries: {non_empty_inputs:,} ({non_empty_inputs/total_records*100:.4f}%)")

    print("\n--- Answer Types ---")
    for atype, cnt in answer_types.most_common():
        print(f"  {atype}: {cnt:,} ({cnt/total_records*100:.2f}%)")

    print("\n--- Duplicate Analysis ---")
    unique_inst = len(set(instructions))
    unique_out = len(set(outputs))
    unique_pairs = len(set(zip(instructions, outputs)))

    print(f"Unique instructions: {unique_inst:,} ({unique_inst/total_records*100:.2f}% unique, {(total_records - unique_inst)/total_records*100:.2f}% duplicate)")
    print(f"Unique outputs: {unique_out:,} ({unique_out/total_records*100:.2f}% unique, {(total_records - unique_out)/total_records*100:.2f}% duplicate)")
    print(f"Unique (instruction, output) pairs: {unique_pairs:,} ({unique_pairs/total_records*100:.2f}% unique, {(total_records - unique_pairs)/total_records*100:.2f}% duplicate)")

    # Top repeated instructions
    inst_counter = Counter(instructions)
    print("\nTop 10 most repeated instructions / questions:")
    for inst, count in inst_counter.most_common(10):
        print(f"  [{count:,}x] {inst[:90]}")

    # Top repeated outputs
    out_counter = Counter(outputs)
    print("\nTop 10 most repeated outputs / answers:")
    for out, count in out_counter.most_common(10):
        print(f"  [{count:,}x] {out[:90]}")

    # Length stats
    print("\n--- Length Distributions ---")
    print(f"Instruction word count: Mean={np.mean(inst_len_words):.1f}, Median={np.median(inst_len_words):.1f}, Min={np.min(inst_len_words)}, Max={np.max(inst_len_words)}, 95th%={np.percentile(inst_len_words, 95):.1f}")
    print(f"Output word count: Mean={np.mean(out_len_words):.1f}, Median={np.median(out_len_words):.1f}, Min={np.min(out_len_words)}, Max={np.max(out_len_words)}, 95th%={np.percentile(out_len_words, 95):.1f}")

    # Extremes
    short_inst = sum(1 for w in inst_len_words if w <= 3)
    long_inst = sum(1 for w in inst_len_words if w > 50)
    short_out = sum(1 for w in out_len_words if w <= 2)
    long_out = sum(1 for w in out_len_words if w > 150)
    print(f"Very short instructions (<=3 words): {short_inst:,} ({short_inst/total_records*100:.2f}%)")
    print(f"Very long instructions (>50 words): {long_inst:,} ({long_inst/total_records*100:.2f}%)")
    print(f"Very short outputs (<=2 words): {short_out:,} ({short_out/total_records*100:.2f}%)")
    print(f"Very long outputs (>150 words): {long_out:,} ({long_out/total_records*100:.2f}%)")

    # Categories
    print("\n--- Category Breakdown (multi-label) ---")
    for cat, cnt in categories.most_common():
        print(f"  {cat}: {cnt:,} ({cnt/total_records*100:.2f}%)")

    # Version analysis
    print("\n--- Version / Edition Mentions ---")
    for ver, cnt in version_counts.most_common():
        print(f"  {ver}: {cnt:,} ({cnt/total_records*100:.2f}%)")

    # Contradiction / ambiguity check
    print("\n--- Contradictory Answers to Identical Questions ---")
    multi_ans_questions = defaultdict(set)
    for inst, out in zip(instructions, outputs):
        if inst_counter[inst] > 1:
            multi_ans_questions[inst].add(out.strip())

    contradiction_sample = []
    num_conflicting_questions = 0
    for q, answers in multi_ans_questions.items():
        if len(answers) > 1:
            num_conflicting_questions += 1
            if len(q.split()) > 4 and len(contradiction_sample) < 6:
                contradiction_sample.append((q, list(answers)[:3]))

    print(f"Questions with conflicting/divergent outputs: {num_conflicting_questions:,}")
    for q, ans_list in contradiction_sample:
        print(f"\nQuestion: {q}")
        for i, ans in enumerate(ans_list, 1):
            print(f"  Ans {i}: {ans[:100]}")

    # Look for hallucination or boilerplate signatures
    print("\n--- Hallucination / Boilerplate Signatures ---")
    boilerplate_patterns = [
        r"as of my knowledge cutoff",
        r"as an ai",
        r"in the latest update",
        r"cannot be answered",
        r"not mentioned in the text",
        r"according to the text",
        r"the provided text",
        r"based on the text",
        r"context provided",
        r"the passage"
    ]
    for bp in boilerplate_patterns:
        matches = sum(1 for out in outputs if re.search(r"\b" + bp + r"\b", out.lower()))
        if matches > 0:
            print(f"  Found '{bp}': {matches:,} occurrences")

    # Save summary dictionary
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    summary = {
        "total_records": total_records,
        "unique_instructions": unique_inst,
        "unique_outputs": unique_out,
        "unique_pairs": unique_pairs,
        "empty_instructions": empty_instructions,
        "empty_outputs": empty_outputs,
        "non_empty_inputs": non_empty_inputs,
        "non_string_types": dict(non_string_types),
        "answer_types": dict(answer_types),
        "categories": dict(categories),
        "version_counts": dict(version_counts),
        "conflicting_questions_count": num_conflicting_questions,
        "word_counts": {
            "inst_mean": float(np.mean(inst_len_words)),
            "inst_median": float(np.median(inst_len_words)),
            "inst_min": int(np.min(inst_len_words)),
            "inst_max": int(np.max(inst_len_words)),
            "inst_p95": float(np.percentile(inst_len_words, 95)),
            "out_mean": float(np.mean(out_len_words)),
            "out_median": float(np.median(out_len_words)),
            "out_min": int(np.min(out_len_words)),
            "out_max": int(np.max(out_len_words)),
            "out_p95": float(np.percentile(out_len_words, 95)),
        }
    }
    with open(REPORT_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved audit summary to {REPORT_DATA_PATH}")

if __name__ == "__main__":
    analyze_dataset()
