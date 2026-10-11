import os
import sys
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
from minementor.backend.services.mentor_orchestrator import MineMentorOrchestrator

def test_pipeline():
    print("Initializing MineMentor Orchestrator...")
    orchestrator = MineMentorOrchestrator()

    test_queries = [
        "I just spawned. What should I do?",
        "How do I make a crafting table?",
        "I need 4 bookshelves. How much paper and leather?",
        "I found 6 iron. What should I make?",
        "I have 3 diamonds. What should I do with them?",
        "I have 5 diamonds, iron armor, 32 bread and I'm on day 8.",
        "What should I do next?",
        "How do I prepare for the Nether?",
        "Why won't my villager take a profession?",
        "How do I craft an Aether portal with glowstone and water?",
        "Can I craft chainmail armor with iron ingots?"
    ]

    print("\n" + "="*80)
    print("RUNNING END-TO-END PIPELINE VALIDATION")
    print("="*80)

    for i, q in enumerate(test_queries, 1):
        print(f"\n[{i}] USER: {q}")
        res = orchestrator.process_query(q)
        print(f"  • Predicted Intent:     {res['intent']} (conf: {res['intent_confidence']})")
        print(f"  • Predicted Difficulty: {res['difficulty']}")
        if res['extracted_state']:
            print(f"  • Extracted Entities:   {res['extracted_state']}")
        if res['recommended_next_goals']:
            print(f"  • Suggested Next Goals: {[g['title'] for g in res['recommended_next_goals']]}")
        if res['sources']:
            print(f"  • Retrieved Sources:    {[s['source'] for s in res['sources']]}")
        print(f"  • MINEMENTOR:\n{res['response']}\n")
        print("-" * 80)

if __name__ == "__main__":
    test_pipeline()
