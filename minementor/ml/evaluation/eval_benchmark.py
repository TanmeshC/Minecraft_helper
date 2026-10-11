import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
from minementor.backend.services.mentor_orchestrator import MineMentorOrchestrator

BENCHMARK_PATH = os.path.join("data", "evaluation", "expert_benchmark.json")
RESULTS_PATH = os.path.join("data", "processed", "benchmark_evaluation_results.json")

def evaluate_expert_benchmark():
    print(f"Loading Expert Benchmark from {BENCHMARK_PATH}...")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    print(f"Loaded {len(benchmarks)} expert test cases.")
    orchestrator = MineMentorOrchestrator()

    results = []
    total_time = 0.0
    exact_calc_matches = 0
    total_calc_questions = 0
    concept_coverage_scores = []
    hallucination_tests_passed = 0
    total_hallucination_tests = 0

    for item in benchmarks:
        q = item["question"]
        q_id = item["id"]
        q_type = item["type"]
        expected_intent = item.get("intent")
        ideal_concepts = item.get("ideal_concepts", [])

        t0 = time.time()
        output = orchestrator.process_query(q)
        lat = time.time() - t0
        total_time += lat

        response_text = output["response"].lower()

        # Measure concept coverage
        matched_concepts = 0
        for concept in ideal_concepts:
            c_tokens = concept.lower().split()
            # If at least half the key words in concept are found
            if sum(1 for w in c_tokens if len(w) > 3 and w in response_text) >= max(1, len(c_tokens) // 2):
                matched_concepts += 1

        coverage = matched_concepts / len(ideal_concepts) if ideal_concepts else 1.0
        concept_coverage_scores.append(coverage)

        # Calculation match check
        calc_match = None
        expected_calc = item.get("calculation_target")
        if expected_calc:
            total_calc_questions += 1
            calc_data = output.get("deterministic_calculation")
            if calc_data:
                raw_mats = calc_data.get("raw_materials", {})
                all_match = all(raw_mats.get(k) == v or (k == "wood_planks" and raw_mats.get("planks") == v) for k, v in expected_calc.items())
                if all_match:
                    exact_calc_matches += 1
                    calc_match = True
                else:
                    calc_match = False
            else:
                calc_match = False

        # Trick / hallucination trap check
        trap_passed = None
        if q_type == "trick_hallucination_trap":
            total_hallucination_tests += 1
            if "not exist in vanilla" in response_text or "cannot be crafted using iron" in response_text or "community mod" in response_text:
                hallucination_tests_passed += 1
                trap_passed = True
            else:
                trap_passed = False

        results.append({
            "id": q_id,
            "type": q_type,
            "question": q,
            "predicted_intent": output["intent"],
            "expected_intent": expected_intent,
            "predicted_difficulty": output["difficulty"],
            "expected_difficulty": item.get("difficulty"),
            "concept_coverage": round(coverage, 3),
            "calculation_match": calc_match,
            "hallucination_trap_passed": trap_passed,
            "latency_ms": round(lat * 1000, 2),
            "sources_count": len(output["sources"])
        })

    avg_latency_ms = (total_time / len(benchmarks)) * 1000
    avg_coverage = sum(concept_coverage_scores) / len(concept_coverage_scores)
    calc_accuracy = (exact_calc_matches / total_calc_questions) if total_calc_questions > 0 else 1.0
    trap_pass_rate = (hallucination_tests_passed / total_hallucination_tests) if total_hallucination_tests > 0 else 1.0

    summary = {
        "benchmark_sample_count": len(benchmarks),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "mean_concept_coverage": round(avg_coverage, 4),
        "deterministic_calculation_accuracy": round(calc_accuracy, 4),
        "hallucination_resistance_rate": round(trap_pass_rate, 4),
        "detailed_results": results
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("EXPERT BENCHMARK EVALUATION RESULTS (Section 27 & 28)")
    print("="*60)
    print(f"Total Test Cases:                    {len(benchmarks)}")
    print(f"Average Pipeline Latency:            {avg_latency_ms:.2f} ms")
    print(f"Mean Concept Coverage:               {avg_coverage * 100:.2f}%")
    print(f"Deterministic Calculation Accuracy:  {calc_accuracy * 100:.2f}%")
    print(f"Hallucination Resistance Rate:       {trap_pass_rate * 100:.2f}%")
    print(f"Results saved to: {RESULTS_PATH}")
    return summary

if __name__ == "__main__":
    evaluate_expert_benchmark()
