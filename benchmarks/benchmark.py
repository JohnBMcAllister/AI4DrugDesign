import argparse
import glob
import json
import os
import re
import time

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Reference pricing per 1M tokens, per model. Sourced from each provider's
# public pricing page; update here rather than at the call site.
MODEL_PRICING = {
    "gpt-5-nano-2025-08-07": {"input": 0.05, "output": 0.40},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
}

# Anthropic requires max_tokens. Set well above the ~1.5k tokens these
# answers run to, so nothing is truncated; billing is on tokens actually
# generated, so a high ceiling costs nothing.
ANTHROPIC_MAX_TOKENS = 16000


def provider_for(model_name: str) -> str:
  """Infer the provider from the model id."""
  return "anthropic" if model_name.startswith("claude") else "openai"


def call_openai(model_name: str, prompt: str) -> dict:
  """One OpenAI completion, normalised to the shared record shape."""
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  res = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt}],
  )
  return {
      "text": res.choices[0].message.content,
      "in_tokens": res.usage.prompt_tokens,
      "out_tokens": res.usage.completion_tokens,
      "stop_reason": res.choices[0].finish_reason,
  }


def call_anthropic(model_name: str, prompt: str) -> dict:
  """One Claude message, normalised to the shared record shape.

  Anthropic names the usage fields input_tokens/output_tokens; they are
  mapped onto prompt_tokens/completion_tokens so the output JSON stays
  comparable with the OpenAI control.
  """
  client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
  res = client.messages.create(
      model=model_name,
      max_tokens=ANTHROPIC_MAX_TOKENS,
      messages=[{"role": "user", "content": prompt}],
  )
  return {
      "text": "".join(b.text for b in res.content if b.type == "text"),
      "in_tokens": res.usage.input_tokens,
      "out_tokens": res.usage.output_tokens,
      "stop_reason": res.stop_reason,
  }


CALLERS = {"openai": call_openai, "anthropic": call_anthropic}


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
    text_dir: str = None,
    label: str = None,
):
  """Run every sample through one model and write results to out_path.

  If text_dir is given, each answer is also written as its own .txt file so
  the raw text can be read without digging through JSON. If label is given
  (A/B/C), those files are named by label only, with no model name in the
  path, so they can be handed out for blind review.
  """
  if provider_for(model_name) == "anthropic" and not os.getenv(
      "ANTHROPIC_API_KEY"
  ):
    raise SystemExit("ANTHROPIC_API_KEY not set - check .env")

  pricing = MODEL_PRICING.get(model_name)
  if pricing is None:
    raise SystemExit(
        f"No pricing entry for {model_name!r}. Add one to MODEL_PRICING so "
        "cost_usd is not silently wrong."
    )

  call = CALLERS[provider_for(model_name)]
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  if text_dir:
    os.makedirs(text_dir, exist_ok=True)

  runs = []
  total_in_tokens = 0
  total_out_tokens = 0
  total_latency = 0.0

  for idx, sample in enumerate(dataset, start=1):
    protein = sample["protein"]
    compound = sample["compound"]
    prompt = format_pipeline_prompt(protein, compound)

    start = time.perf_counter()
    res = call(model_name, prompt)
    elapsed = time.perf_counter() - start

    in_tokens = res["in_tokens"]
    out_tokens = res["out_tokens"]

    cost = ((in_tokens / 1_000_000) * pricing["input"]) + (
        (out_tokens / 1_000_000) * pricing["output"]
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
        "total_tokens": in_tokens + out_tokens,
        "cost_usd": round(cost, 6),
        "stop_reason": res["stop_reason"],
        "output": res["text"],
    }

    if text_dir:
      if label:
        stem = f"{label}_{idx:02d}"
      else:
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", record["compound_name"] or "")
        stem = f"{idx:02d}_{safe_name}"
      text_path = os.path.join(text_dir, f"{stem}.txt")
      with open(text_path, "w") as f:
        f.write(res["text"] or "")
      record["output_file"] = text_path

    runs.append(record)
    truncated = " [TRUNCATED]" if res["stop_reason"] == "max_tokens" else ""
    print(
        f"Completed [{idx}/{len(dataset)}] {record['compound_name']} |"
        f" {record['latency_sec']}s | {record['total_tokens']} tokens"
        f" | ${record['cost_usd']:.6f}{truncated}"
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
                  ((total_in_tokens / n) * 1000 / 1_000_000 * pricing["input"])
                  + (
                      (total_out_tokens / n)
                      * 1000
                      / 1_000_000
                      * pricing["output"]
                  )
              ),
              4,
          )
          if n
          else 0
      ),
      "runs": runs,
  }
  if label:
    summary["label"] = label

  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

  print(f"\nSaved benchmark results to {out_path}")
  if text_dir:
    print(f"Saved {len(runs)} raw outputs to {text_dir}/")


def load_dataset() -> list:
  """Load every *_candidates.json fixture, in sorted filename order."""
  combined = []
  for file_path in sorted(glob.glob("benchmarks/*_candidates.json")):
    with open(file_path, "r") as f:
      items = json.load(f)
      combined.extend(items)
      print(f"Loaded {len(items)} samples from {file_path}")

  if not combined:
    raise SystemExit(
        "No benchmarks/*_candidates.json files found. Run from the repo root."
    )
  return combined


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Benchmark one model against the fixed compound set."
  )
  parser.add_argument(
      "--model",
      default="claude-haiku-4-5",
      help="Model id. Must have an entry in MODEL_PRICING.",
  )
  parser.add_argument(
      "--out",
      default=None,
      help="Results JSON path. Defaults to benchmarks/results/<model>.json.",
  )
  parser.add_argument(
      "--text-dir",
      default=None,
      help="Directory for per-run .txt outputs. Defaults to "
      "benchmarks/outputs/<model>/, or benchmarks/outputs/blind/ with --label.",
  )
  parser.add_argument(
      "--label",
      default=None,
      help="Blind-review label (A/B/C). Names output files by label only, "
      "with no model name in the path.",
  )
  parser.add_argument(
      "--limit",
      type=int,
      default=None,
      help="Only run the first N samples. Use while iterating to avoid "
      "spending credit on the full set.",
  )
  args = parser.parse_args()

  slug = re.sub(r"[^A-Za-z0-9]+", "_", args.model)
  out_path = args.out or f"benchmarks/results/{slug}.json"
  text_dir = args.text_dir or (
      "benchmarks/outputs/blind" if args.label
      else f"benchmarks/outputs/{slug}"
  )

  dataset = load_dataset()
  if args.limit:
    dataset = dataset[: args.limit]
    print(f"Limiting to first {len(dataset)} samples.")

  print(f"\nTotal test samples ready for evaluation: {len(dataset)}")
  print(f"Model: {args.model} ({provider_for(args.model)})")

  benchmark_model(
      model_name=args.model,
      dataset=dataset,
      out_path=out_path,
      text_dir=text_dir,
      label=args.label,
  )