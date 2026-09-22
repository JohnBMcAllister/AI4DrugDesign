import glob
import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Standard reference pricing per 1M tokens
INPUT_PRICE_PER_M = 10.00
OUTPUT_PRICE_PER_M = 50.00


def format_pipeline_prompt(p: dict, c: dict) -> str:
  """Replicates the prompt template from tabs/pipeline/detail.py."""
  return (
      "You are an expert medicinal chemist. Explain how this compound "
      "works in the context of its protein target.\n\n"
      f"PROTEIN:\n"
      f"  Name: {p.get('title', 'N/A')}\n"
      f"  PDB ID: {p.get('pdb_id', 'N/A')}\n"
      f"  Classification: {p.get('classification', 'N/A')}\n"
      f"  Organism: {p.get('organism', 'N/A')}\n\n"
      f"COMPOUND:\n"
      f"  Name: {c.get('name', 'Unknown')}\n"
      f"  SMILES: {c.get('smiles', 'N/A')}\n"
      f"  MW: {c.get('mw', 'N/A')}, LogP: {c.get('logp', 'N/A')}\n"
      f"  Binding: {c.get('activity_type', 'N/A')} = "
      f"{c.get('activity_value', 'N/A')} {c.get('activity_units', '')}\n\n"
      "Explain:\n"
      "1. How this compound likely binds to the protein\n"
      "2. Key functional groups and their roles\n"
      "3. Mechanism of action (inhibitor, agonist, etc.)\n"
      "4. Strengths and weaknesses\n"
      "5. Potential optimization strategies\n\n"
      "Be concise and scientifically accurate."
  )


def benchmark_model(
    model_name: str,
    dataset: list,
    out_path: str = "benchmarks/results/astra-eval.json",
):
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  runs = []
  total_in_tokens = 0
  total_out_tokens = 0
  total_latency = 0.0

  for idx, sample in enumerate(dataset, start=1):
    protein = sample["protein"]
    compound = sample["compound"]
    prompt = format_pipeline_prompt(protein, compound)

    start = time.perf_counter()
    res = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - start

    usage = res.usage
    in_tokens = usage.prompt_tokens
    out_tokens = usage.completion_tokens

    cost = ((in_tokens / 1_000_000) * INPUT_PRICE_PER_M) + (
        (out_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
    )

    total_in_tokens += in_tokens
    total_out_tokens += out_tokens
    total_latency += elapsed

    record = {
        "index": idx,
        "pdb_id": protein.get("pdb_id"),
        "compound_name": compound.get("name"),
        "latency_sec": round(elapsed, 3),
        "prompt_tokens": in_tokens,
        "completion_tokens": out_tokens,
        "total_tokens": usage.total_tokens,
        "cost_usd": round(cost, 6),
        "output": res.choices[0].message.content,
    }
    runs.append(record)
    print(
        f"Completed [{idx}/{len(dataset)}] {record['compound_name']} |"
        f" {record['latency_sec']}s | {usage.total_tokens} tokens"
    )

  n = len(dataset)
  summary = {
      "model": model_name,
      "sample_count": n,
      "avg_latency_sec": round(total_latency / n, 3) if n else 0,
      "avg_prompt_tokens": round(total_in_tokens / n, 1) if n else 0,
      "avg_completion_tokens": round(total_out_tokens / n, 1) if n else 0,
      "projected_cost_per_1k_runs_usd": (
          round(
              (
                  ((total_in_tokens / n) * 1000 / 1_000_000 * INPUT_PRICE_PER_M)
                  + (
                      (total_out_tokens / n)
                      * 1000
                      / 1_000_000
                      * OUTPUT_PRICE_PER_M
                  )
              ),
              4,
          )
          if n
          else 0
      ),
      "runs": runs,
  }

  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

  print(f"\nSaved control benchmark results to {out_path}")


if __name__ == "__main__":
  candidate_files = sorted(glob.glob("benchmarks/*_candidates.json"))

  combined_dataset = []
  for file_path in candidate_files:
    with open(file_path, "r") as f:
      items = json.load(f)
      combined_dataset.extend(items)
      print(f"Loaded {len(items)} samples from {file_path}")

  # Fallback to TEST_SAMPLES if no generated candidate files were found
  if not combined_dataset:
    print("No *_candidates.json files found. Falling back to TEST_SAMPLES...")
    combined_dataset = TEST_SAMPLES

  print(f"\nTotal test samples ready for evaluation: {len(combined_dataset)}")

  # Execute baseline control benchmark
  benchmark_model(
      model_name="gpt-6-astra",
      dataset=combined_dataset,
      out_path="benchmarks/results/astra-eval.json",
  )