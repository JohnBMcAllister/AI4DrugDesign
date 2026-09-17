import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Standard reference pricing per 1M tokens (adjust according to your provider tier)
INPUT_PRICE_PER_M = 0.05
OUTPUT_PRICE_PER_M = 0.40

# Representative test samples matching the exact state dictionaries passed in detail.py
TEST_SAMPLES = [
    {
        "protein": {
            "title": "SARS-CoV-2 main protease",
            "pdb_id": "6LU7",
            "classification": "VIRAL PROTEIN",
            "organism": "Severe acute respiratory syndrome coronavirus 2",
        },
        "compound": {
            "name": "N3 inhibitor",
            "smiles": "CC(C)C[C@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)OCC2=CC=CC=C2)C(=O)N[C@@H](CC(=O)N3CC[C@@H]3)C(=O)C=C",
            "mw": 680.8,
            "logp": 2.1,
            "activity_type": "IC50",
            "activity_value": "16.7",
            "activity_units": "uM",
        },
    },
    {
        "protein": {
            "title": "Tyrosine-protein kinase ABL1",
            "pdb_id": "1IEP",
            "classification": "TRANSFERASE",
            "organism": "Homo sapiens",
        },
        "compound": {
            "name": "Imatinib",
            "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            "mw": 493.6,
            "logp": 3.5,
            "activity_type": "Ki",
            "activity_value": "0.1",
            "activity_units": "uM",
        },
    },
    # Add 15–20 curated test pairs here
]


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
    out_path: str = "benchmarks/results/baseline_control.json",
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
  benchmark_model(
      model_name="gpt-5-nano-2025-08-07",
      dataset=TEST_SAMPLES,
      out_path="benchmarks/results/baseline_control.json",
  )