# Benchmark Metrics Guide

This benchmark reports two complementary groups of metrics:

1. **Operational metrics**: response latency, token usage, estimated generation cost, and run validity.
2. **Scientific-quality metrics**: structured-rubric score, Medicinal and Chemical Term Density (MCTD), and target-conditioned semantic alignment.

The operational metrics quantify the resources required to generate a response. The scientific-quality metrics evaluate different properties of the response itself. Cost, latency, and quality should be compared separately rather than combined into one score.

## Quantitative operational metrics

### Per-run measurements

Each attempted sample is stored in the `runs` array of the result JSON. A successful generation contains the following quantitative fields:

| Result field | Unit | Description |
|---|---:|---|
| `latency_sec` | seconds | Wall-clock time for the model generation API call, measured with `time.perf_counter()` and rounded to three decimal places |
| `prompt_tokens` | tokens | Input tokens reported by the OpenAI API for the generation call |
| `completion_tokens` | tokens | Output tokens reported by the OpenAI API for the generation call |
| `total_tokens` | tokens | Total usage reported by the API, normally `prompt_tokens + completion_tokens` |
| `cost_usd` | USD | Estimated generation cost based on the hard-coded input and output token prices |
| `finish_reason` | categorical | API finish reason, retained to help identify filtered or otherwise incomplete responses |

The generation cost is estimated as:

```text
cost_usd = (prompt_tokens / 1,000,000 × input_price_per_million)
         + (completion_tokens / 1,000,000 × output_price_per_million)
```

The current constants in `benchmark_openai.py` are `$10.00` per million input tokens and `$50.00` per million output tokens. These are benchmark configuration values, not a live pricing lookup. Update them when evaluating a model with different pricing, and report the values used when publishing results.

### Aggregate measurements

The top level of the result JSON summarizes all attempted samples:

| Result field | Description |
|---|---|
| `total_samples` | Number of samples loaded into the benchmark dataset |
| `completed_samples` | Number of attempts currently saved in `runs`, including flagged responses and API errors |
| `valid_samples` | Number of saved runs included in aggregate averages |
| `flagged_samples` | Number of HTTP-successful responses excluded because the completion was empty or content-filtered |
| `error_samples` | Number of failed API requests stored with an `error` field |
| `avg_latency_sec` | Arithmetic mean of generation latency across valid runs |
| `avg_prompt_tokens` | Arithmetic mean of generation prompt tokens across valid runs |
| `avg_completion_tokens` | Arithmetic mean of generation completion tokens across valid runs |
| `avg_cost_usd` | Arithmetic mean of estimated generation cost across valid runs |
| `projected_cost_per_1k_runs_usd` | `avg_cost_usd × 1,000`, assuming future runs have the same average token usage |

A run is valid for aggregation when it has neither an `error` field nor `flagged: true`. Flagged and error runs remain in the output for auditability but do not contribute to latency, token, cost, or scientific-quality averages. Because the benchmark checkpoints after every sample and can resume from an existing file, `completed_samples` may be smaller than `total_samples` while a run is in progress.

The aggregate latency and cost calculations are:

```text
avg_latency_sec = sum(valid generation latency) / valid_samples
avg_cost_usd = sum(valid estimated generation cost) / valid_samples
projected_cost_per_1k_runs_usd = avg_cost_usd × 1,000
```

The projection is a linear estimate, not a measured batch price or throughput metric. The script does not currently report requests per second, tokens per second, percentiles, variance, or confidence intervals.

### Judge usage is separate

The structured-rubric evaluator makes an additional API call. When usage information is available, each run's `structured_rubric` object records:

- `judge_latency_sec`
- `judge_prompt_tokens`
- `judge_completion_tokens`
- `judge_total_tokens`
- `judge_model`

These judge measurements are **not** included in `latency_sec`, `cost_usd`, `avg_latency_sec`, or `avg_cost_usd`. The benchmark does not currently calculate judge cost. Therefore, `cost_usd` represents generation cost only, and the true end-to-end evaluation cost is higher whenever rubric judging is enabled.

## Scientific-quality metrics

The benchmark uses three scientific-quality metrics:

1. Structured-rubric score
2. Medicinal and Chemical Term Density (MCTD)
3. Target-conditioned semantic alignment

These metrics measure different aspects of a generated response. They should be interpreted together rather than added into a single score.

Their top-level aggregate fields are `avg_structured_rubric`, `avg_medical_chemical_term_density`, and `avg_semantic_embedding_alignment`. Each is an arithmetic mean over valid runs that contain that particular metric. This detail matters when a judge call fails: the generation can still count as a valid operational run and contribute MCTD and alignment scores while being omitted from `avg_structured_rubric`.

## 1. Structured-rubric score

### What it measures

The structured rubric uses the model selected by `BENCHMARK_JUDGE_MODEL` as an automated judge; it defaults to `gpt-4o-mini`. The judge receives the target metadata, ligand information, and generated response. It scores whether the response contains scientifically useful medicinal-chemistry reasoning.

Each response is evaluated in five categories:

| Category | Weight | What the judge evaluates |
|---|---:|---|
| Binding mechanics | 25% | Binding mode, residue interactions, hydrogen bonds, hydrophobic contacts, and pocket interactions |
| Functional groups | 20% | Pharmacophore features and the roles of important chemical groups |
| Mechanism | 20% | Proposed mechanism of action, such as competitive, allosteric, reversible, or covalent inhibition |
| Optimization depth | 20% | Target-specific SAR reasoning and feasible optimization strategies, including bioisosteres and physicochemical-property tuning |
| Scientific conciseness | 15% | Information density, relevance, and avoidance of generic or unsupported claims |

The overall rubric score is a weighted score on a 1--10 scale:

```text
overall =
    0.25 × binding mechanics
  + 0.20 × functional groups
  + 0.20 × mechanism
  + 0.20 × optimization depth
  + 0.15 × scientific conciseness
```

### How to read the score

The following ranges are practical interpretation guidelines, not formally validated scientific thresholds:

| Score | Suggested interpretation |
|---:|---|
| 9.0--10.0 | Exceptional coverage, specificity, and scientific depth |
| 8.0--8.9 | Strong response with detailed and mostly complete reasoning |
| 7.0--7.9 | Useful response with some missing detail or uneven reasoning |
| 6.0--6.9 | Moderate response with noticeable omissions or generic discussion |
| Below 6.0 | Limited, insufficiently specific, or potentially unreliable response |

A score of `8.00` does **not** mean that the response is 80% factually correct. It means that the automated judge rated its coverage and quality at approximately eight on the defined 1--10 rubric.

### Important limitation

This metric is itself an LLM judgment. A technically convincing response may score well even if it contains a factual error. Small score differences should not be treated as decisive, especially without repeated judging or expert review. For example, scores of `8.00` and `8.03` should normally be considered effectively tied.

## 2. Medicinal and Chemical Term Density (MCTD)

### What it measures

MCTD measures how concentrated the response is with recognized medicinal-chemistry terminology. It is calculated as:

```text
MCTD = unique recognized domain terms / total output words × 100
```

The domain terms can include chemical groups, pharmacological concepts, protein features, binding-site terminology, and target-specific residues or structural motifs.

### How to read the score

An MCTD score of `6.49` means that the response contains approximately 6.49 unique recognized domain terms per 100 output words.

In general:

- A higher MCTD indicates denser use of medicinal-chemistry terminology.
- A lower MCTD may indicate more general explanation, repetition, or conversational padding.
- MCTD is most useful for comparing responses produced from the same prompt and evaluated with the same lexicon and tokenizer.

There is no universal value that makes an MCTD score "good." Response length, writing style, target, terminology list, and term-recognition rules all affect the result.

### Important limitation

MCTD measures terminology presence, not correctness. A response can mention many technical terms while using them incorrectly. Conversely, a clear and scientifically correct response can receive a lower score because it uses simpler language. Short responses may also receive an advantage because the denominator is smaller.

## 3. Target-conditioned semantic alignment

### What it measures

Semantic alignment measures how closely the meaning of a generated response matches a target-specific reference profile. The reference profile contains established information about the protein target, binding site, known ligand, and relevant structural biology.

The generated response and reference profile are converted into embedding vectors. Their cosine similarity is then calculated:

```text
alignment = dot(generated_vector, reference_vector)
            / (norm(generated_vector) × norm(reference_vector))
```

### How to read the score

Higher values indicate greater semantic similarity to the corresponding target reference. For example, an alignment of `0.79` represents stronger similarity than `0.68` when both values were produced with the same embedding model and evaluation procedure.

Cosine similarity theoretically ranges from `-1` to `1`, although biomedical text evaluated in this benchmark is normally expected to produce positive values. The score should be used comparatively; it is not an accuracy percentage. An alignment of `0.74` does **not** mean that the response is 74% correct.

Absolute thresholds should not be transferred to another benchmark unless the same embedding model, reference profiles, preprocessing, and text lengths are used.

### Important limitation

Semantic similarity does not guarantee factual correctness. A response can closely resemble the reference topic while making an incorrect claim. A scientifically valid insight can also score lower if it is absent from the reference profile. Results therefore depend on both the embedding model and the quality and completeness of the target reference.

## Reading the metrics together

The metrics answer different questions:

| Metric | Primary question |
|---|---|
| Rubric score | Does the response provide complete and useful medicinal-chemistry reasoning? |
| MCTD | How concentrated is the response with domain-specific terminology? |
| Semantic alignment | How closely does the response match established target-specific information? |

A strong response should ideally have:

- a high rubric score, showing scientific depth and completeness;
- a reasonably high MCTD, showing efficient use of technical terminology; and
- strong semantic alignment, showing relevance to the target-specific reference.

However, no single metric determines the winner. Examples include:

- **High rubric + high alignment + moderate MCTD:** scientifically strong and relevant, but written with more explanatory language.
- **High MCTD + moderate rubric:** terminology-dense, but potentially missing deeper reasoning or logical connections.
- **High rubric + lower alignment:** detailed reasoning that may introduce ideas not represented in the reference or drift from target-specific evidence.
- **High alignment + low rubric:** topically relevant, but incomplete in mechanism, functional-group analysis, or optimization guidance.

## Recommended comparison rules

When comparing models:

1. Compare results generated from the same compounds, prompts, and reference profiles.
2. Report the number of valid samples because averages based on different sample counts may not be directly comparable.
3. Inspect per-target results in addition to the overall average.
4. Treat very small score differences as ties unless repeated runs show a consistent difference.
5. Keep latency and generation cost separate from scientific-quality metrics.
6. Do not add rubric, MCTD, and alignment values directly because they use different scales and measure different properties.
7. Use expert review when factual accuracy or experimental decisions are important.

## General limitations

The benchmark currently relies on automated evaluation and a limited set of target-compound examples. A single generation per compound does not capture output variability, and aggregate averages do not provide confidence intervals. The GPT-4o-mini judge may have its own model biases, MCTD depends on term-recognition coverage, and semantic alignment depends on the selected reference profiles and embedding model.

These metrics are therefore best used to identify broad comparative patterns and cost--quality trade-offs. They should not be interpreted as proof of biological validity, binding affinity, synthetic feasibility, clinical efficacy, or experimental success.

## Suggested future improvements

- Generate multiple responses per compound and report standard deviations or confidence intervals.
- Repeat judge evaluations or use multiple independent judges.
- Add blinded review by medicinal chemists or structural biologists.
- Verify residue-level and mechanism claims against curated structural evidence.
- Expand and version the MCTD terminology resource.
- Version the embedding model and target reference profiles for reproducibility.
- Evaluate factuality and hallucination rates as separate metrics.

