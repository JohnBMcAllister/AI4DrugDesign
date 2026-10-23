# Scientific-Quality Metrics Guide

This document explains the three qualitative metrics used in the medicinal-chemistry benchmark:

1. Structured-rubric score
2. Medicinal and Chemical Term Density (MCTD)
3. Target-conditioned semantic alignment

These metrics measure different aspects of a generated response. They should be interpreted together rather than added into a single score.

## 1. Structured-rubric score

### What it measures

The structured rubric uses GPT-4o-mini as an automated judge. The judge receives the target metadata, ligand information, benchmark prompt, and generated response. It scores whether the response contains scientifically useful medicinal-chemistry reasoning.

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

