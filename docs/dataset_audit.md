# MineMentor Dataset Audit & Suitability Report

**Date**: 2026-10-11  
**Project**: MineMentor — AI Minecraft Beginner Assistant  
**Evaluator**: Antigravity Core Research  
**Primary Candidate Dataset**: [`Aiwensile2/Minecraft_QA-pairs_Instruction_Dataset`](https://huggingface.co/datasets/Aiwensile2/Minecraft_QA-pairs_Instruction_Dataset)  
**Decision**: **PARTIALLY SUITABLE** (Strict Filtering & Hybrid Knowledge Architecture Required)

---

## 1. Primary Dataset Overview

The primary dataset under investigation is `Aiwensile2/Minecraft_QA-pairs_Instruction_Dataset`, hosted on Hugging Face.
- **Source**: Scraped from the *Minecraft Wiki* (`minecraft.wiki`) and *Minecraft Fandom Wiki* (`minecraft.fandom.com`).
- **Generation Method**: Synthetically generated Q&A pairs created using OpenAI's `gpt-3.5-turbo` prompted across raw extracted wiki text and tables.
- **Claimed Intent**: Providing instruction-tuning data for LLMs handling diverse Minecraft inquiries across four formats: Short Answer, Long Answer, Boolean, and Normal Q&A.
- **File Format**: Single JSON file `minecraft_instruction_dataset.json` (uncompressed size: 126,364,710 bytes / 120.5 MB).

---

## 2. Exact Number of Examples

An exhaustive programmatic parsing of `minecraft_instruction_dataset.json` reveals:
- **Total Raw Records**: `390,317`
- **Corrupt / Non-String Rows**: `200` (51 records where `instruction` is float `NaN`, 149 records where `output` is float `NaN`).
- **Completely Empty String / Whitespace Rows**: `0`
- **Total Valid String Pairs**: `390,117`

---

## 3. Schema & Fields

Every record in the raw dataset contains exactly 3 keys:
```json
{
  "instruction": "<Question or prompt text>",
  "input": "",
  "output": "<Answer or response text>"
}
```

### Schema Audit Findings:
1. **100% Redundant Field**: The `input` field is completely empty (`""`) across **all 390,317 records** (0 non-empty occurrences).
2. **Missing Essential Metadata**: The dataset completely lacks structured fields for:
   - `edition` (Java vs. Bedrock)
   - `version` (e.g., 1.20, 1.21)
   - `category` / `intent` (Crafting, Combat, Survival, etc.)
   - `difficulty` (Beginner, Intermediate, Advanced)
   - `source_url` or wiki page provenance
   - `confidence` or ground-truth verification flags

---

## 4. Example Records (Audited Directly from Disk)

### High-Quality Beginner / Gameplay Q&A
```json
{
  "instruction": "How can I craft iron leggings in Minecraft?",
  "input": "",
  "output": "Iron leggings can be crafted using seven iron ingots. Arrange them in the crafting table in an upside-down 'U' shape: place three ingots in the top row, two on the sides of the middle row, and none in the bottom row."
}
```

### High-Quality Edition-Specific Q&A
```json
{
  "instruction": "What are the differences in valid spawn blocks between Minecraft's Java Edition and Bedrock Edition during raid waves?",
  "input": "",
  "output": "In Minecraft's Java Edition, valid spawn blocks during raid waves include full blocks, upside-down slabs/stairs, soul sand, and packed/blue ice, but not glass, leaves, or certain other blocks. In contrast, Minecraft's Bedrock Edition allows raid waves to spawn on any block except leaves and scaffolding."
}
```

### Irrelevant Historical Snapshot Trivia (Useless for Beginners)
```json
{
  "instruction": "What modification was made to the spawn eggs of phantoms in the 18w19a snapshot?",
  "input": "",
  "output": "In the 18w19a snapshot, the spawn eggs of phantoms were modified to better reflect the updated texture of the mob."
}
```

### Obsolete Release-Notes Trivia
```json
{
  "instruction": "What are the changes in 1.13-pre5 for Java Edition?",
  "input": "",
  "output": "In Java Edition 1.13-pre5, some of the changes include upgrading path optics code to a separate library named datafixerupper, renaming entities such as Eye of Ender Signal to Eye of Ender..."
}
```

### Corrupted / HTML Scraping Residue (Instruction = NaN)
```json
{
  "instruction": NaN,
  "input": "",
  "output": "------|\n| Raw Codor<br/>Raw Salmon+<br/>Anyfuel |                 |"
}
```

### Corrupted Output (Output = NaN)
```json
{
  "instruction": "How do you differentiate between cobblestone and andesite in Minecraft?",
  "input": "",
  "output": NaN
}
```

---

## 5. Dataset Quality

- **Text Length Distributions**:
  - `instruction` word count: **Mean = 11.7 words**, Median = 11.0, Min = 1, Max = 362, 95th Percentile = 18.0.
  - `output` word count: **Mean = 30.1 words**, Median = 21.0, Min = 1, Max = 958, 95th Percentile = 87.0.
- **Short Answer & Boolean Skew**:
  - `Long Answer (>30 words)`: 157,124 (40.26%)
  - `Medium Answer (6-30 words)`: 132,599 (33.97%)
  - `Boolean / Yes-No`: 56,145 (14.38%)
  - `Short Answer (<=5 words)`: 44,449 (11.39%)
  - `Extremely short outputs (<=2 words)`: **87,538 (22.43%)** — e.g. single-word outputs like "True" (35,039x), "TRUE" (10,012x), "False" (7,601x), "4" (575x), "#3F76E4" (349x).
- **Synthetic Scraping Artifacts**:
  - Hex color values scraped from wiki stylesheets (e.g. `#3F76E4` appears 349 times as an answer!).
  - Unrendered HTML formatting: `<br/>`, `&nbsp;`, Wiki table pipes (`|`).
  - Text encoding artifacts: Unicode replacement characters `22` instead of `2x2` or `2×2`.
  - Placeholder instructions: `response` (169x), `prompt` (137x).

---

## 6. Duplicate & Near-Duplicate Analysis

- **Unique Instructions**: `346,411` (**88.75% unique**, 11.25% duplicate queries).
- **Unique Outputs**: `302,141` (**77.41% unique**, 22.59% duplicate answers).
- **Unique (Instruction, Output) Exact Pairs**: `381,199` (**97.66% unique**, 2.34% duplicate pairs).
- **Top Repeated Instructions**:
  - `What are block states in Minecraft?` (225x)
  - `response` (169x)
  - `prompt` (137x)
  - `What is the purpose of block states in Minecraft?` (64x)
  - `What are advancements in Minecraft?` (56x)
  - `What is a spawn egg in Minecraft?` (54x)
  - `What structures can be found in the Jungle Hills biome?` (51x)
- **Contradictory / Divergent Answers on Identical Queries**:
  - **19,656 distinct questions** have multiple diverging answers. While some represent phrasing variations, others present conflicting mechanical facts because GPT-3.5 generated responses from different wiki sections or snapshot versions without timestamps.

---

## 7. Category & Topic Breakdown

Programmatic multi-label classification across the 390,317 entries reveals significant topic imbalance:

| Category | Record Count | % of Dataset | Suitability for Beginner Assistant |
|---|---|---|---|
| **Building & Blocks** | 82,650 | 21.18% | Medium (lots of obscure decorative blocks) |
| **Exploration & Biomes** | 61,195 | 15.68% | High |
| **Combat & Mobs** | 49,935 | 12.79% | High |
| **Mining & Resources** | 43,842 | 11.23% | High |
| **Survival & Basics** | 42,665 | 10.93% | Very High |
| **Crafting & Recipes** | 38,284 | 9.81% | Very High (requires deterministic verification) |
| **Nether & End** | 31,399 | 8.04% | High |
| **Farming & Breeding** | 23,010 | 5.90% | High |
| **Redstone & Mechanics** | 17,199 | 4.41% | Medium |
| **Enchanting & Potions** | 12,817 | 3.28% | Very High (underrepresented) |
| **Commands & Technical** | 9,971 | 2.55% | Low (out of scope for survival beginners) |
| **Other / Historical / Changelog / Trivia** | 132,465 | 33.94% | Very Low (snapshot trivia, pre-releases, code internals) |

**Key Finding**: Over a third of the dataset consists of historical patch trivia (e.g. "What happened in snapshot 14w02a?"), which actively hurts a beginner assistant by polluting the retrieval space.

---

## 8. Version & Edition Issues

Minecraft mechanics differ dramatically between editions and historical versions.
- **Edition Mentions**:
  - `Java Edition` explicitly referenced: `58,085` (14.88%)
  - `Bedrock Edition` explicitly referenced: `55,151` (14.13%)
  - **Unspecified Edition**: **~71% of the dataset** does not clarify whether the answer applies to Java or Bedrock!
- **Version Awareness Stripped by Authors**:
  - The dataset README explicitly acknowledges: *"we performed extensive data cleaning to remove irrelevant sections, such as game version information, achievements, advancements, and history."*
  - This decision was counterproductive: in Minecraft, version information is critical (e.g. 1.9 Combat Update offhand/shield mechanics, 1.14 Villager trading overhaul, 1.16 Nether overhaul, 1.18 World generation height change from Y=0 to Y=-64).
  - Merging mechanics without version tags risks answering a modern player with obsolete 1.12 mechanics.

---

## 9. Potential Hallucinations & Synthetic Artifacts

Programmatic string scanning reveals LLM hallucination and prompt leakage artifacts:
- `"the passage"` / `"the provided text"` / `"based on the text"` / `"not mentioned in the text"`: **123 occurrences** in outputs. This occurs when GPT-3.5 was asked to answer from a wiki excerpt and leaked the context prompt.
- `"as an ai"`: **5 occurrences**.
- `"in the latest update"`: **30 occurrences** (vague, time-bound claims that will be obsolete as new updates release).

---

## 10. License Analysis: CC BY-NC-SA 3.0

The dataset is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)**.

### Permitted Uses:
1. **Academic Projects & Research Papers**: Fully permitted.
2. **Open-Source Non-Commercial Portfolios / GitHub Projects**: Fully permitted, provided attribution is given.
3. **Sharing & Adaptation**: Permitted for non-commercial educational purposes.

### Restrictions:
1. **Commercial Use Strictly Prohibited**: The dataset, fine-tuned models trained exclusively on it, and direct commercial deployments **cannot be monetized**.
2. **ShareAlike (Viral Copyleft)**: Any derivative dataset or adapted model weights must be distributed under the same CC BY-NC-SA 3.0 license.
3. **Attribution Required**: You must give appropriate credit to the authors (Aiwensile2 / Zenodo DOI: `10.5281/zenodo.11568799`) and link to the license.

### Commercial Guidance:
> [!WARNING]
> If MineMentor is ever distributed commercially or hosted as a paid SaaS, model weights fine-tuned on this dataset cannot be used. A dual pipeline using MIT/Apache-2.0 datasets (or a RAG retrieval system referencing live Wiki content under CC BY-SA 3.0 with proper attribution) must be maintained.

---

## 11. Other Useful Minecraft Datasets Discovered

A comprehensive audit of Hugging Face repositories identified several high-value complementary datasets:

1. **`ProfQu/MinecraftRecipes`**
   - **Records**: 836 exact recipes.
   - **Source**: Extracted directly from official Minecraft source code.
   - **License**: **MIT** (Permissive, commercial-friendly).
   - **Utility**: Perfect ground truth for our **Deterministic Calculation Engine** (Section 24). Contains exact grid recipes (`pattern`, `ingredients`, `result`).

2. **`naklecha/minecraft-question-answer-700k`**
   - **Records**: 694,814 Q&A pairs (glaive.ai synthetic pipeline).
   - **Source**: Extracted from 18,000+ Minecraft Wiki pages.
   - **License**: CC BY-NC-SA 3.0.
   - **Unique Advantage**: **Includes exact source URLs** (e.g., `https://minecraft.wiki/w/Food#Nourishment_value`) for every single Q&A entry. Excellent for source citation and verification.

3. **`KoalaBrainResearcher/Minecraft-Wiki-2023`**
   - **Size**: 43.1 MB (`articles.jsonl`, 10K+ articles).
   - **Source**: Full raw text of Minecraft Wiki pages (`title`, `text`).
   - **Utility**: Unsurpassed source text for our **Vector Database & RAG Retrieval** layer. Allows chunking directly from original wiki documentation rather than relying solely on generated Q&A pairs.

4. **`CraftJarvis/minecraft-contractor-crafting` & `CraftJarvis/minecraft-vla-sft`**
   - **License**: MIT / Academic.
   - **Utility**: Embodied task planning and sequential progression traces.

5. **`declip/Minecraft-Server-Chat`**
   - **Size**: 1M+ chat entries.
   - **License**: **CC0 1.0 (Public Domain)**.
   - **Utility**: Provides natural player query phrasing and slang for intent classification and query normalization.

---

## 12. Comparison Table

| Dataset | Size / Records | License | Source | Primary Modality | Strengths | Weaknesses |
|---|---|---|---|---|---|---|
| **Aiwensile2 (Primary)** | 390,317 rows | CC BY-NC-SA 3.0 | Wiki + GPT-3.5 | Instruction Q&A | Large scale; diverse questions | 34% changelog noise; 22% 1-word answers; stripped version info; 200 corrupt rows |
| **naklecha-700k** | 694,814 rows | CC BY-NC-SA 3.0 | Wiki + Glaive.ai | Q&A with URLs | **Contains exact Wiki source URLs**; higher consistency | Large size; CC BY-NC-SA; synthetic |
| **ProfQu Recipes** | 836 recipes | **MIT** | Minecraft Source Code | Structured tabular (CSV) | **100% deterministic & accurate**; official recipe patterns; MIT | Crafting only |
| **KoalaBrain Wiki** | ~10,000+ articles (43 MB) | CC BY-NC-SA / CC BY-SA | Official Minecraft Wiki | Unstructured Text (JSONL) | **Pure factual ground truth**; ideal for chunking and embeddings | Requires parsing & chunking |
| **declip Server Chat**| 1M+ messages | **CC0 (Public Domain)** | Multiplayer server logs | Dialogue / Chat | Real-world player language and slang | High noise; informal |

---

## 13. Recommended Dataset Combination

Rather than treating the primary dataset as a monolithic knowledge base, we recommend a **4-pillar hybrid dataset architecture**:

```
                               ┌───────────────────────────────────────────────┐
                               │           MINEMENTOR HYBRID DATA              │
                               └──────────────────────┬────────────────────────┘
                                                      │
         ┌─────────────────────┬──────────────────────┴───────────────┬────────────────────────┐
         │                     │                                      │                        │
         ▼                     ▼                                      ▼                        ▼
┌─────────────────┐   ┌─────────────────┐                   ┌──────────────────┐     ┌───────────────────┐
│ ProfQu Recipes  │   │ KoalaBrain Wiki │                   │ Cleaned Aiwensile│     │ naklecha-700k     │
│ (836 Recipes)   │   │ (43 MB Text)    │                   │ (~120K Filtered) │     │ (Citation Index)  │
└────────┬────────┘   └────────┬────────┘                   └────────┬─────────┘     └─────────┬─────────┘
         │                     │                                     │                         │
         ▼                     ▼                                     ▼                         ▼
┌─────────────────┐   ┌─────────────────┐                   ┌──────────────────┐     ┌───────────────────┐
│  Deterministic  │   │  RAG Knowledge  │                   │ Intent/Difficulty│     │ Provenance & URL  │
│  Crafting Calc  │   │  Base (FAISS /  │                   │ Classifier       │     │ Citation Engine   │
│     Engine      │   │     BM25)       │                   │ Training Set     │     │                   │
└─────────────────┘   └─────────────────┘                   └──────────────────┘     └───────────────────┘
```

1. **Deterministic Layer (`ProfQu/MinecraftRecipes`)**: 100% authoritative crafting logic and resource calculations.
2. **Knowledge Retrieval Base (`KoalaBrainResearcher/Minecraft-Wiki-2023`)**: Raw factual chunked text for hybrid vector/keyword search.
3. **ML Training & Benchmark Subset (`Aiwensile2 Cleaned`)**: Filtered down to ~120,000 clean survival/mechanic pairs for intent classification, difficulty labeling, and ranking.
4. **Citation Layer (`naklecha-700k`)**: Supplies verifiable source URLs to display in the UI.

---

## 14. Recommended Training Strategy

For the ML models specified in the project roadmap (Intent Classifier, Difficulty Classifier, Ranking Model):

1. **Step 1: Data Cleaning & Normalization**:
   - Drop all 200 corrupt (`NaN`) records.
   - Drop all snapshot/changelog questions (`1.13-pre5`, `18w19a`, `snapshot`, `pre-release`).
   - Remove boilerplate single-word responses (`True`, `False`, `#3F76E4`).
   - Deduplicate queries to avoid train/test data leakage.
2. **Step 2: Model 1 — Intent Classification**:
   - Train a fast baseline: **TF-IDF + Logistic Regression / LinearSVC**.
   - Evaluate against a compact transformer: **MiniLM-L6-v2 / DistilBERT**.
   - Target classes: `Crafting`, `Mining`, `Survival`, `Combat`, `Exploration`, `Farming`, `Building`, `Redstone`, `Enchanting`, `Nether`, `End`, `Villagers`, `Troubleshooting`, `Progression`, `Resource Calculation`, `General`.
3. **Step 3: Model 2 — Difficulty Classification**:
   - Label training set into `Beginner`, `Intermediate`, `Advanced`, `Expert`.
   - Use cross-entropy loss with class weights to counteract distribution skew.
4. **Step 4: Strict Evaluation Separation**:
   - Create a clean 80/10/10 Train/Validation/Test split.
   - Maintain a completely held-out, expert-curated evaluation benchmark (Section 27) that contains zero scraped wiki questions.

---

## 15. Recommended RAG Strategy

Do NOT embed raw question-answer pairs as the sole RAG database. Answering questions by matching them against pre-generated answers causes compounding hallucinations.

### The Correct Pipeline:
1. **Chunk Authoritative Wiki Documents**: Use sliding-window chunking (300-500 tokens with 50-token overlap) over raw wiki documentation.
2. **Hybrid Retrieval**:
   - **Dense Vector Search**: Open-source local sentence transformer (`all-MiniLM-L6-v2` or `bge-small-en-v1.5`) indexed via **FAISS**.
   - **Sparse Keyword Search**: **BM25** (vital for Minecraft's exact item IDs like `diamond_pickaxe`, `blaze_rod`, `tripwire_hook`).
   - **Reciprocal Rank Fusion (RRF)**: Combine dense and sparse scores.
3. **Cross-Encoder Re-Ranking**: Use a lightweight cross-encoder (`ms-marco-MiniLM-L-6-v2`) to re-score the top 20 candidates down to top 5.
4. **Deterministic Augmentation**: If the user query is identified as a crafting or calculation intent, bypass vector retrieval and feed exact recipe data directly from the deterministic engine.

---

## 16. Whether Fine-Tuning is Worthwhile

**Verdict**: **Fine-tuning should NOT be the primary path for Minecraft factual knowledge.**

### Empirical Reasons:
1. **Factual Grounding**: Minecraft updates frequently. A fine-tuned model memorizes fixed facts in its weights that cannot be updated without retraining.
2. **Hallucination Risk**: When fine-tuned on synthetic datasets with 14% boolean answers and 22% 2-word answers, small LLMs tend to generate clipped, overconfident answers.
3. **RAG Dominance**: RAG provides 100% updateability, explicit citations, zero retraining cost, and run-time version toggling (Java vs. Bedrock).

### Where Fine-Tuning IS Worthwhile (Phase 8):
- Fine-tuning a small model (e.g., Llama-3-8B / Mistral-7B via LoRA) **strictly for tone and style** (mentor persona, beginner-friendly explanations, structured progression breakdowns) after grounding with RAG.

---

## 17. Final Recommendation

### Suitability Decision: **PARTIALLY SUITABLE**

| Status | Recommendation |
|---|---|
| **Raw Dataset as Monolithic AI Knowledge Base** | ❌ **REJECT** (Too noisy, 34% changelog trivia, stripped versions, corrupt rows, ungrounded) |
| **Filtered Subset for ML Intent & Difficulty Training** | ✅ **ACCEPT** (Cleaned down to ~120K relevant survival/crafting pairs) |
| **Authoritative RAG Grounding** | ✅ **USE WIKI CHUNKS + RECIPE ENGINE** (`ProfQu` + `KoalaBrain`) |
| **Licensing Compliance** | ✅ **CC BY-NC-SA 3.0 Documented** (Free/open-source non-commercial research) |

We are now ready to establish Phase 1 (Data Pipeline & Cleaning) with clear scientific justification.
