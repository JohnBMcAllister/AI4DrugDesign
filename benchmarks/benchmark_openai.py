import argparse
import functools
import glob
import json
import math
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

import tabs.pipeline.docking as docking_module
from tabs.pipeline.detail import (
    _get_pocket_contacts,
    build_compound_context,
    build_explain_prompt,
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# RCSB is hit once per compound by the pocket analysis otherwise. Patching the
# module attribute works because _get_pocket_contacts imports fetch_pdb_file
# inside the function body, so the name resolves at call time.
docking_module.fetch_pdb_file = functools.lru_cache(maxsize=16)(
    docking_module.fetch_pdb_file
)

# Bump this whenever the prompt, the context block or the judge rubric changes.
# Scores from different versions are not comparable; the resume path refuses to
# mix them in one results file.
PROMPT_VERSION = "vina-contacts-v1"

# Reference pricing per 1M tokens. These drive avg_cost_usd and
# projected_cost_per_1k_runs_usd, so set them to the real figures for the model
# under test before quoting those numbers.
INPUT_PRICE_PER_M = float(os.getenv("BENCHMARK_INPUT_PRICE_PER_M", "10.00"))
OUTPUT_PRICE_PER_M = float(os.getenv("BENCHMARK_OUTPUT_PRICE_PER_M", "50.00"))

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
JUDGE_MODEL = os.getenv("BENCHMARK_JUDGE_MODEL", "gpt-4o-mini")

RUBRIC_WEIGHTS = {
    "binding_mechanics_score": 0.25,
    "functional_group_score": 0.20,
    "mechanism_score": 0.20,
    "optimization_depth_score": 0.20,
    "scientific_conciseness_score": 0.15,
}

# overall_richness_score is computed deterministically from RUBRIC_WEIGHTS in
# validate_judge_result, so the judge is not asked for it.
JUDGE_RESPONSE_SCHEMA = {
    "name": "medicinal_chemistry_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "binding_mechanics_score": {"type": "number", "minimum": 1, "maximum": 10},
            "functional_group_score": {"type": "number", "minimum": 1, "maximum": 10},
            "mechanism_score": {"type": "number", "minimum": 1, "maximum": 10},
            "optimization_depth_score": {"type": "number", "minimum": 1, "maximum": 10},
            "scientific_conciseness_score": {"type": "number", "minimum": 1, "maximum": 10},
            "unsupported_residues": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reasoning": {"type": "string"},
        },
        "required": [
            "binding_mechanics_score",
            "functional_group_score",
            "mechanism_score",
            "optimization_depth_score",
            "scientific_conciseness_score",
            "unsupported_residues",
            "reasoning",
        ],
        "additionalProperties": False,
    },
}

# Deterministic medical and chemical term density (MCTD) vocabularies.
GENERAL_MEDCHEM_TERMS = {
    "activity", "agonist", "allosteric", "amide", "amine", "antagonist",
    "aromatic", "assay", "atp competitive", "binding affinity", "binding mode",
    "bioisostere", "bioisosterism", "catalytic", "covalent", "electrophile",
    "functional group", "gatekeeper", "heteroaryl", "hydrogen bond",
    "hydrogen bond acceptor", "hydrogen bond donor", "hydrophobic", "inhibitor",
    "kinase", "ligand", "ligand efficiency", "lipophilic", "logp",
    "mechanism of action", "metabolic stability", "pharmacokinetics",
    "pharmacophore", "polar", "potency", "reversible", "sar", "selectivity",
    "solubility", "steric", "structure activity relationship", "substituent",
    "toxicity", "warhead",
}

TARGET_MEDCHEM_TERMS = {
    "6LU7": {
        "3clpro", "main protease", "mpro", "catalytic dyad", "cys145", "his41",
        "gly143", "ser144", "his163", "glu166", "gln189", "oxyanion hole",
        "s1 subsite", "s2 subsite", "s4 subsite", "thiolate",
    },
    "1IEP": {
        "abl", "bcr abl", "imatinib", "dfg out", "dfg motif", "hinge region",
        "met318", "thr315", "t315i", "asp381", "gatekeeper residue", "p loop",
        "type ii inhibitor",
    },
    "2HYY": {
        "abl", "bcr abl", "imatinib", "dfg out", "dfg motif", "hinge region",
        "met318", "thr315", "t315i", "asp381", "gatekeeper residue", "p loop",
        "type i inhibitor", "type ii inhibitor",
    },
    "3ERT": {
        "estrogen receptor", "er alpha", "tamoxifen", "4 hydroxytamoxifen",
        "glu353", "arg394", "his524", "helix 12", "ligand binding domain",
        "coactivator", "serm",
    },
}

WORD_PATTERN = re.compile(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b")
RESIDUE_PATTERN = re.compile(
    r"\b"
    r"(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)"
    r"\s*-?\s*\d+[A-Za-z]?"
    r"\b",
    flags=re.IGNORECASE,
)

# Concise profiles derived from the deposited structure descriptions and their
# primary literature. They describe known target/ligand biology, not an
# unobserved pose for each benchmark candidate.
TARGET_REFERENCE_PROFILES = {
    "6LU7": (
        "SARS-CoV-2 main protease, also called Mpro or 3CLpro, is a cysteine "
        "protease with an active site formed around catalytic residues His41 and "
        "Cys145. Structure 6LU7 contains the peptidomimetic inhibitor N3 covalently "
        "bound to Cys145. Ligand recognition spans substrate-recognition subsites "
        "and the oxyanion-hole region. Relevant optimization questions include "
        "directed electrophile reactivity, protease selectivity, permeability, "
        "solubility, and productive occupation of the S1, S2, and S4 subsites."
    ),
    "1IEP": (
        "Structure 1IEP contains the mouse c-Abl kinase domain in complex with "
        "imatinib, also called STI-571. Imatinib binds the canonical ATP-binding "
        "site and stabilizes a specific inactive activation-loop conformation. "
        "Relevant medicinal-chemistry features include hinge recognition, the "
        "gatekeeper region, the hydrophobic back pocket, conformational selectivity, "
        "kinase selectivity, physicochemical properties, and resistance mutations "
        "such as T315I."
    ),
    "2HYY": (
        "Structure 2HYY contains the human Abl kinase domain in complex with "
        "imatinib, also called STI571 or Glivec. The structure supports analysis of "
        "ATP-site inhibition, inactive-state recognition, and the molecular basis of "
        "resistance to Abl inhibitors. Relevant optimization questions include hinge "
        "and gatekeeper interactions, type-I versus type-II binding, resistance "
        "mutations, kinase selectivity, potency, solubility, and lipophilicity."
    ),
    "3ERT": (
        "Structure 3ERT contains the human estrogen receptor alpha ligand-binding "
        "domain in complex with the selective antagonist 4-hydroxytamoxifen. "
        "Antagonist binding prevents the receptor from forming the productive "
        "coactivator-binding surface associated with transcriptional activation. "
        "Relevant optimization questions include ligand-pocket recognition, helix-12 "
        "positioning, antagonist or SERM behavior, estrogen-receptor subtype "
        "selectivity, metabolic stability, basicity, lipophilicity, and tissue effects."
    ),
}


# ── Prompt construction ───────────────────────────────────────────────


def resolve_contacts(sample: dict) -> dict | None:
    """Binding-site contacts for a candidate.

    Prefers contacts stored when the candidate file was generated (no network,
    and guaranteed to match what the pipeline computed). Falls back to deriving
    them from a stored docked_sdf.
    """
    if "contacts" in sample:
        return sample["contacts"]

    compound = sample["compound"]
    pdb_id = (sample["protein"].get("pdb_id") or "").strip().upper()
    if not compound.get("docked_sdf") or not pdb_id:
        return None
    return _get_pocket_contacts(compound, pdb_id)


# ── Structured LLM-as-a-judge ─────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """
You are evaluating the scientific quality of a medicinal-chemistry explanation.

The model under evaluation was shown the data block under
"context_given_to_the_model" and asked to explain the compound. Grade its output
against that data.

Score the explanation from 1 to 10 in each category:

1. binding_mechanics_score
   Binding mode, pocket features and residue-level interactions. Reward claims
   anchored to residues that appear in the supplied contact lists. Penalise
   residues the context does not mention, and penalise ignoring the supplied
   contacts in favour of generic description.

2. functional_group_score
   Pharmacophore elements, and how specific functional groups in the given
   SMILES affect binding, polarity, reactivity or solvent exposure.

3. mechanism_score
   Whether the proposed mechanism of action is chemically and biologically
   plausible for this target, and appropriately qualified.

4. optimization_depth_score
   Target-specific SAR reasoning, bioisosteric ideas, LogP or solubility tuning,
   selectivity, resistance, experimental feasibility. Reward proposals tied to a
   specific weakness visible in the supplied data.

5. scientific_conciseness_score
   Reward information density and careful scientific language. Penalise
   repetition, generic filler, invented contacts, unjustified certainty and
   hallucination.

The docking affinity and the contact lists in the context are predictions from
AutoDock Vina and a distance cutoff, not experimental measurements. An answer
that presents a Vina affinity as a measured binding constant, or a distance
contact as an observed interaction, loses points for unjustified certainty.
Where the context says a value is not available, the correct behaviour is to say
so rather than supply a number.

Where the context reports no experimental activity, the compound is a screening
candidate, not the ligand crystallised in the referenced structure. Treat claims
that it is as errors.

List in unsupported_residues every residue the output cites that does not appear
in the supplied contact lists. Use an empty array if there are none.

Do not reward an answer merely because it is long. Give a concise explanation of
the most important strengths and weaknesses.
""".strip()


def build_judge_messages(context_block: str, generated_output: str) -> list[dict]:
    evaluation_input = {
        "context_given_to_the_model": context_block,
        "generated_output": generated_output,
    }
    user_prompt = (
        "Evaluate the following model output using the rubric.\n\n"
        + json.dumps(evaluation_input, indent=2)
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def validate_judge_result(result: dict) -> dict:
    normalized = {}

    for field in RUBRIC_WEIGHTS:
        if field not in result:
            raise ValueError(f"Judge response is missing {field}")

        try:
            score = float(result[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Judge returned a non-numeric value for {field}") from exc

        if not 1 <= score <= 10:
            raise ValueError(f"{field} must be between 1 and 10; received {score}")

        normalized[field] = round(score, 2)

    reasoning = result.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Judge response must contain non-empty reasoning")

    unsupported = result.get("unsupported_residues", [])
    if not isinstance(unsupported, list):
        raise ValueError("unsupported_residues must be a list")

    weighted_score = sum(
        normalized[field] * weight for field, weight in RUBRIC_WEIGHTS.items()
    )

    normalized["overall_richness_score"] = round(weighted_score, 2)
    normalized["unsupported_residues"] = [str(r) for r in unsupported]
    normalized["unsupported_residue_count"] = len(unsupported)
    normalized["reasoning"] = reasoning.strip()

    return normalized


def evaluate_scientific_richness(
    context_block: str,
    generated_output: str,
    judge_model: str = JUDGE_MODEL,
) -> dict:
    messages = build_judge_messages(context_block, generated_output)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=judge_model,
        messages=messages,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": JUDGE_RESPONSE_SCHEMA,
        },
    )
    elapsed = time.perf_counter() - start

    raw_content = response.choices[0].message.content
    if not raw_content:
        raise ValueError("Judge returned an empty response")

    result = validate_judge_result(json.loads(raw_content))
    result["judge_model"] = judge_model
    result["judge_latency_sec"] = round(elapsed, 3)

    if response.usage is not None:
        result["judge_prompt_tokens"] = response.usage.prompt_tokens
        result["judge_completion_tokens"] = response.usage.completion_tokens
        result["judge_total_tokens"] = response.usage.total_tokens

    return result


# ── Deterministic term density ────────────────────────────────────────


def normalize_term_text(text: str) -> str:
    """Normalize punctuation variants before matching domain phrases."""
    normalized = text.lower()
    normalized = re.sub(r"[‐-―]", "-", normalized)
    normalized = re.sub(r"[-_/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def matched_domain_terms(text: str, vocabulary: set[str]) -> set[str]:
    """Unique domain terms and residue labels present in *text*."""
    if not text:
        return set()

    normalized_text = normalize_term_text(text)
    matched = set()

    for term in vocabulary:
        normalized_term = normalize_term_text(term)
        term_pattern = (
            rf"(?<![A-Za-z0-9])"
            rf"{re.escape(normalized_term)}"
            rf"(?![A-Za-z0-9])"
        )
        if re.search(term_pattern, normalized_text):
            matched.add(normalized_term)

    for residue in RESIDUE_PATTERN.findall(text):
        matched.add(re.sub(r"[\s-]+", "", residue.lower()))

    return matched


def calculate_mctd(
    generated_output: str,
    pdb_id: str | None,
    prompt_text: str | None = None,
) -> dict:
    """Unique domain terms per 100 output words.

    Terms are counted once, so repeating technical vocabulary cannot inflate the
    score. The prompt now supplies real residue labels, so a model can raise the
    raw score by echoing them; novel_score subtracts every term that already
    appeared in the prompt and is the figure that discriminates between models.
    """
    empty = {
        "score": 0.0,
        "unique_domain_terms": 0,
        "novel_score": 0.0,
        "novel_domain_terms": 0,
        "total_words": 0,
        "matched_terms": [],
        "novel_terms": [],
        "prompt_adjusted": prompt_text is not None,
    }
    if not generated_output or not generated_output.strip():
        return empty

    words = WORD_PATTERN.findall(generated_output)
    total_words = len(words)
    if not total_words:
        return empty

    normalized_pdb_id = (pdb_id or "").strip().upper()
    vocabulary = (
        GENERAL_MEDCHEM_TERMS | TARGET_MEDCHEM_TERMS.get(normalized_pdb_id, set())
    )

    matched_terms = matched_domain_terms(generated_output, vocabulary)
    given_terms = matched_domain_terms(prompt_text or "", vocabulary)
    novel_terms = matched_terms - given_terms

    return {
        "score": round(len(matched_terms) / total_words * 100, 3),
        "unique_domain_terms": len(matched_terms),
        "novel_score": round(len(novel_terms) / total_words * 100, 3),
        "novel_domain_terms": len(novel_terms),
        "total_words": total_words,
        "matched_terms": sorted(matched_terms),
        "novel_terms": sorted(novel_terms),
        "prompt_adjusted": prompt_text is not None,
    }


def average_mctd(valid_runs: list[dict], key: str = "score") -> float | None:
    scores = [
        record["medical_chemical_term_density"][key]
        for record in valid_runs
        if key in record.get("medical_chemical_term_density", {})
    ]
    return round(sum(scores) / len(scores), 3) if scores else None


# ── Semantic alignment ────────────────────────────────────────────────


def chunk_text(text: str, words_per_chunk: int = 180) -> list[str]:
    """Split long text before embedding to avoid model-context truncation."""
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be greater than zero")
    words = (text or "").split()
    return [
        " ".join(words[start:start + words_per_chunk])
        for start in range(0, len(words), words_per_chunk)
    ]


def mean_embedding(embedding_rows) -> list[float]:
    """Mean-pool a list or array of embedding vectors."""
    rows = embedding_rows.tolist() if hasattr(embedding_rows, "tolist") else embedding_rows
    rows = [list(map(float, row)) for row in rows]
    if not rows:
        raise ValueError("The embedding model returned no vectors")

    vector_size = len(rows[0])
    if vector_size == 0 or any(len(row) != vector_size for row in rows):
        raise ValueError("The embedding model returned inconsistent vector dimensions")

    return [
        sum(row[column] for row in rows) / len(rows)
        for column in range(vector_size)
    ]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have equal, nonzero dimensions")

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cannot compare a zero-length embedding vector")
    return dot_product / (left_norm * right_norm)


def calculate_semantic_alignment(
    generated_output: str,
    pdb_id: str | None,
    embedding_model,
    words_per_chunk: int = 180,
) -> dict:
    """Compare an output with its target-conditioned structural profile."""
    normalized_pdb_id = (pdb_id or "").strip().upper()
    reference_text = TARGET_REFERENCE_PROFILES.get(normalized_pdb_id)
    if reference_text is None:
        raise ValueError(
            f"No semantic reference profile is configured for PDB "
            f"{normalized_pdb_id!r}. Add one to TARGET_REFERENCE_PROFILES."
        )

    output_chunks = chunk_text(generated_output, words_per_chunk)
    if not output_chunks:
        raise ValueError("Cannot calculate semantic alignment for an empty output")
    reference_chunks = chunk_text(reference_text, words_per_chunk)

    output_embeddings = embedding_model.encode(output_chunks, normalize_embeddings=True)
    reference_embeddings = embedding_model.encode(
        reference_chunks, normalize_embeddings=True
    )
    output_vector = mean_embedding(output_embeddings)
    reference_vector = mean_embedding(reference_embeddings)

    return {
        "score": round(cosine_similarity(output_vector, reference_vector), 6),
        "reference_pdb_id": normalized_pdb_id,
        "embedding_model": str(getattr(
            embedding_model, "model_name_or_path", DEFAULT_EMBEDDING_MODEL,
        )),
        "output_chunks": len(output_chunks),
        "words_per_chunk": words_per_chunk,
    }


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Semantic alignment requires the benchmark dependencies. "
            "Run `uv sync --extra benchmark`."
        ) from exc
    return SentenceTransformer(model_name)


def average_semantic_alignment(valid_runs: list[dict]) -> float | None:
    scores = [
        record["semantic_embedding_alignment"]["score"]
        for record in valid_runs
        if "score" in record.get("semantic_embedding_alignment", {})
    ]
    return round(sum(scores) / len(scores), 6) if scores else None


def average_structured_rubric(valid_runs: list[dict]) -> dict | None:
    scored_runs = [
        record["structured_rubric"]
        for record in valid_runs
        if "structured_rubric" in record
    ]
    if not scored_runs:
        return None

    score_fields = [
        *RUBRIC_WEIGHTS,
        "overall_richness_score",
        "unsupported_residue_count",
    ]
    return {
        field: round(
            sum(float(rubric.get(field, 0)) for rubric in scored_runs) / len(scored_runs),
            3,
        )
        for field in score_fields
    }


# ── Benchmark driver ──────────────────────────────────────────────────


def is_excluded(record: dict) -> bool:
    """Runs that shouldn't count toward the averages: hard errors and 'bad'
    responses (empty output or content-filtered) that still came back as a 200
    from the API."""
    return "error" in record or record.get("flagged", False)


def sample_for(record: dict, dataset: list) -> dict | None:
    """Map a saved record back to its dataset entry by 1-based index."""
    sample_index = record.get("index")
    if not isinstance(sample_index, int):
        print("Skipping backfill: record has no valid index.")
        return None

    position = sample_index - 1
    if not 0 <= position < len(dataset):
        print(f"Skipping backfill: index {sample_index} is outside the dataset.")
        return None
    return dataset[position]


def benchmark_model(
    model_name: str,
    dataset: list,
    out_path: str,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    embedding_model = load_embedding_model(embedding_model_name)

    runs: list[dict] = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                saved = json.load(f)
        except Exception as exc:
            raise SystemExit(f"Could not read {out_path}: {exc}")

        saved_version = saved.get("prompt_version")
        if saved_version != PROMPT_VERSION:
            raise SystemExit(
                f"{out_path} was produced under prompt_version "
                f"{saved_version!r}, but this script is {PROMPT_VERSION!r}. "
                "Those scores are not comparable — use a new --out path."
            )
        runs = saved.get("runs", [])
        print(f"Resuming benchmark: loaded {len(runs)} existing runs.")

    completed_indices = {
        r["index"] for r in runs if isinstance(r.get("index"), int)
    }

    total_in_tokens = sum(r.get("prompt_tokens", 0) for r in runs if not is_excluded(r))
    total_out_tokens = sum(r.get("completion_tokens", 0) for r in runs if not is_excluded(r))
    total_latency = sum(r.get("latency_sec", 0.0) for r in runs if not is_excluded(r))
    total_cost = sum(r.get("cost_usd", 0.0) for r in runs if not is_excluded(r))

    def build_summary() -> dict:
        valid_runs = [r for r in runs if not is_excluded(r)]
        n_valid = len(valid_runs)
        avg_cost = round(total_cost / n_valid, 6) if n_valid else 0
        return {
            "model": model_name,
            "prompt_version": PROMPT_VERSION,
            "judge_model": JUDGE_MODEL,
            "input_price_per_m": INPUT_PRICE_PER_M,
            "output_price_per_m": OUTPUT_PRICE_PER_M,
            "total_samples": len(dataset),
            "completed_samples": len(runs),
            "valid_samples": n_valid,
            "flagged_samples": sum(1 for r in runs if r.get("flagged", False)),
            "error_samples": sum(1 for r in runs if "error" in r),
            "avg_latency_sec": round(total_latency / n_valid, 3) if n_valid else 0,
            "avg_prompt_tokens": round(total_in_tokens / n_valid, 1) if n_valid else 0,
            "avg_completion_tokens": round(total_out_tokens / n_valid, 1) if n_valid else 0,
            "avg_cost_usd": avg_cost,
            "projected_cost_per_1k_runs_usd": round(avg_cost * 1000, 4) if n_valid else 0,
            "avg_structured_rubric": average_structured_rubric(valid_runs),
            "avg_medical_chemical_term_density": average_mctd(valid_runs, "score"),
            "avg_novel_term_density": average_mctd(valid_runs, "novel_score"),
            "avg_semantic_embedding_alignment": average_semantic_alignment(valid_runs),
            "runs": runs,
        }

    def checkpoint():
        with open(out_path, "w") as f:
            json.dump(build_summary(), f, indent=2)

    def score_record(record: dict, sample: dict, output: str, context_block: str):
        """Attach every metric that is missing from *record*. Returns True if a
        paid judge call was made, so the caller can checkpoint."""
        pdb_id = sample["protein"].get("pdb_id")

        if "medical_chemical_term_density" not in record:
            record["medical_chemical_term_density"] = calculate_mctd(
                generated_output=output,
                pdb_id=pdb_id,
                prompt_text=record.get("prompt"),
            )

        if "semantic_embedding_alignment" not in record:
            try:
                record["semantic_embedding_alignment"] = calculate_semantic_alignment(
                    generated_output=output,
                    pdb_id=pdb_id,
                    embedding_model=embedding_model,
                )
                record.pop("semantic_error", None)
            except Exception as exc:
                record["semantic_error"] = str(exc)
                print(f"Semantic alignment failed [{record.get('index')}]: {exc}")

        if "structured_rubric" in record:
            return False

        try:
            record["structured_rubric"] = evaluate_scientific_richness(
                context_block=context_block,
                generated_output=output,
            )
            record.pop("judge_error", None)
            rubric = record["structured_rubric"]
            print(
                f"  judge {rubric['overall_richness_score']}/10 "
                f"({rubric['unsupported_residue_count']} unsupported residues)"
            )
        except Exception as judge_error:
            record["judge_error"] = str(judge_error)
            print(
                f"Judge failed [{record.get('index')}] "
                f"{record.get('compound_name')}: {judge_error}"
            )
        return True

    # ── Backfill metrics for runs loaded from disk ─────────────────────
    for record in runs:
        if is_excluded(record):
            continue
        output = record.get("output")
        if not output:
            continue

        sample = sample_for(record, dataset)
        if sample is None:
            continue

        context_block = record.get("context") or build_compound_context(
            sample["compound"], sample["protein"], resolve_contacts(sample)
        )
        if "context" not in record:
            record["context"] = context_block

        if score_record(record, sample, output, context_block):
            checkpoint()

    # ── Generate and score the remaining samples ──────────────────────
    for idx, sample in enumerate(dataset, start=1):
        if idx in completed_indices:
            continue

        protein = sample["protein"]
        compound = sample["compound"]
        contacts = resolve_contacts(sample)
        context_block = build_compound_context(compound, protein, contacts)
        prompt = build_explain_prompt(compound, protein, contacts)

        start = time.perf_counter()
        try:
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

            choice = res.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            is_empty = not content or not content.strip()
            is_bad = is_empty or finish_reason == "content_filter"

            record = {
                "index": idx,
                "prompt_version": PROMPT_VERSION,
                "pdb_id": protein.get("pdb_id"),
                "compound_name": compound.get("name"),
                "has_docked_pose": bool(contacts),
                "latency_sec": round(elapsed, 3),
                "prompt_tokens": in_tokens,
                "completion_tokens": out_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": round(cost, 6),
                "finish_reason": finish_reason,
                "prompt": prompt,
                "context": context_block,
                "output": content,
            }

            if is_bad:
                record["flagged"] = True
                record["flag_reason"] = "empty_completion" if is_empty else "content_filter"
                print(
                    f"Flagged [{idx}/{len(dataset)}] {compound.get('name')} | "
                    f"{record['flag_reason']} (excluded from stats)"
                )
            else:
                total_in_tokens += in_tokens
                total_out_tokens += out_tokens
                total_latency += elapsed
                total_cost += cost

                print(
                    f"Completed [{idx}/{len(dataset)}] {record['compound_name']} | "
                    f"{record['latency_sec']}s | {usage.total_tokens} tokens"
                )
                score_record(record, sample, content, context_block)

        except BadRequestError as e:
            elapsed = time.perf_counter() - start
            err_code = getattr(e, "code", "bad_request")
            print(
                f"Skipped [{idx}/{len(dataset)}] {compound.get('name')} | "
                f"Flagged ({err_code}): {e.message}"
            )
            record = {
                "index": idx,
                "prompt_version": PROMPT_VERSION,
                "pdb_id": protein.get("pdb_id"),
                "compound_name": compound.get("name"),
                "latency_sec": round(elapsed, 3),
                "error": str(e.message),
                "code": err_code,
                "output": None,
            }
        except Exception as e:
            print(f"Error on [{idx}/{len(dataset)}] {compound.get('name')}: {e}")
            break

        runs.append(record)
        checkpoint()

    # Final recompute, so a fully-resumed run still refreshes the summary.
    checkpoint()
    print(f"\nSaved final benchmark results to {out_path}")


def load_dataset(pattern: str = "benchmarks/*_candidates.json") -> list:
    candidate_files = sorted(glob.glob(pattern))
    if not candidate_files:
        raise SystemExit(
            f"No candidate files matched {pattern!r}. Generate them first — the "
            "prompt needs docking results, ADME properties and binding-site "
            "contacts per candidate."
        )

    combined: list = []
    for file_path in candidate_files:
        with open(file_path, "r") as f:
            items = json.load(f)
        combined.extend(items)
        print(f"Loaded {len(items)} samples from {file_path}")

    missing_docking = sum(
        1 for s in combined if s["compound"].get("vina_affinity") is None
    )
    if missing_docking:
        print(
            f"WARNING: {missing_docking}/{len(combined)} candidates have no "
            "vina_affinity. Those prompts will read 'not available' for every "
            "docking field — regenerate the candidate files."
        )

    missing_contacts = sum(
        1 for s in combined
        if "contacts" not in s and not s["compound"].get("docked_sdf")
    )
    if missing_contacts:
        print(
            f"WARNING: {missing_contacts}/{len(combined)} candidates have "
            "neither stored contacts nor a docked_sdf, so the model gets no "
            "binding-site residues."
        )

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument(
        "--out",
        default=f"benchmarks/results/astra-eval-{PROMPT_VERSION}.json",
    )
    parser.add_argument("--candidates", default="benchmarks/*_candidates.json")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    dataset = load_dataset(args.candidates)
    print(f"\nTotal test samples ready for evaluation: {len(dataset)}")

    benchmark_model(
        model_name=args.model,
        dataset=dataset,
        out_path=args.out,
        embedding_model_name=args.embedding_model,
    )