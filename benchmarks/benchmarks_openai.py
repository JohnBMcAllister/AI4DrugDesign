import glob
import json
import math
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Standard reference pricing per 1M tokens
INPUT_PRICE_PER_M = 2.00
OUTPUT_PRICE_PER_M = 12.00
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Structured LLM-as-a-judge
JUDGE_MODEL = os.getenv("BENCHMARK_JUDGE_MODEL", "gpt-4o-mini")

RUBRIC_WEIGHTS = {
    "binding_mechanics_score": 0.25,
    "functional_group_score": 0.20,
    "mechanism_score": 0.20,
    "optimization_depth_score": 0.20,
    "scientific_conciseness_score": 0.15
}

JUDGE_RESPONSE_SCHEMA = {
    "name": "medicinal_chemistry_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "binding_mechanics_score": {
                "type": "number",
                "minimum": 1,
                "maximum": 10,
            },
            "functional_group_score": {
                "type": "number",
                "minimum": 1,
                "maximum": 10,
            },
            "mechanism_score": {
                "type": "number",
                "minimum": 1,
                "maximum": 10,
            },
            "optimization_depth_score": {
                "type": "number",
                "minimum": 1,
                "maximum": 10,
            },
            "scientific_conciseness_score": {
                "type": "number",
                "minimum": 1,
                "maximum": 10,
            },
            "overall_richness_score": {
                "type": "number",
                "minimum": 1,
                "maximum": 10,
            },
            "reasoning": {
                "type": "string"
            },
        },
        "required": [
            "binding_mechanics_score",
            "functional_group_score",
            "mechanism_score",
            "optimization_depth_score",
            "scientific_conciseness_score",
            "overall_richness_score",
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
    "kinase", "ligand", "lipophilic", "logp", "mechanism of action",
    "metabolic stability", "pharmacokinetics", "pharmacophore", "polar", "potency",
    "reversible", "sar", "selectivity", "solubility", "steric",
    "structure activity relationship", "substituent", "toxicity", "warhead",
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

def build_judge_messages(protein: dict, compound: dict, generated_output: str) -> list[dict]:
    system_prompt = """
You are evaluating the scientific quality of a medicinal-chemistry explanation.

Score the explanation from 1 to 10 in each category:

1. binding_mechanics_score
   Evaluate binding mode, binding-site features, residue interactions,
   hydrogen bonds, hydrophobic interactions, and appropriate uncertainty.

2. functional_group_score
   Evaluate identification of pharmacophore elements and explanation of
   how functional groups affect binding, polarity, reactivity, or exposure.

3. mechanism_score
   Evaluate whether the proposed mechanism of action is chemically and
   biologically plausible and appropriately qualified.

4. optimization_depth_score
   Evaluate target-specific SAR reasoning, bioisosteric ideas, LogP or
   solubility tuning, selectivity, resistance, and experimental feasibility.

5. scientific_conciseness_score
   Reward high information density, careful scientific language, and useful
   caveats. Penalize repetition, generic filler, invented residue contacts,
   unjustified certainty, and hallucinations.

Do not reward an answer merely because it is long.

Claims about a candidate's exact binding pose must be treated as hypotheses
unless the supplied metadata establishes that the candidate is the ligand
in the referenced structure.

The overall score should reflect the weighted scientific richness of the
answer. Give a concise explanation of the most important strengths and weaknesses.
""".strip()

    evaluation_input = {
        "protein": {
            "name": protein.get("title"),
            "pdb_id": protein.get("pdb_id"),
            "classification": protein.get("classification"),
            "organism": protein.get("organism"),
        },
        "compound": {
            "name": compound.get("name"),
            "smiles": compound.get("smiles"),
            "molecular_weight": compound.get("mw"),
            "logp": compound.get("logp"),
            "activity_type": compound.get("activity_type"),
            "activity_value": compound.get("activity_value"),
            "activity_units": compound.get("activity_units"),
        },
        "generated_output": generated_output,
    }

    user_prompt = (
        "Evaluate the following model output using the rubric.\n\n"
        + json.dumps(evaluation_input, indent=2)
    )

    return [
        {"role": "system", "content": system_prompt},
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

    weighted_score = sum(
        normalized[field] * weight
        for field, weight in RUBRIC_WEIGHTS.items()
    )

    normalized["overall_richness_score"] = round(weighted_score, 2)
    normalized["reasoning"] = reasoning.strip()

    return normalized

def evaluate_scientific_richness(protein: dict, compound: dict, generated_output: str,
                                 judge_model: str = JUDGE_MODEL) -> dict:

    messages = build_judge_messages(
        protein=protein,
        compound=compound,
        generated_output=generated_output
    )

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

    parsed_result = json.loads(raw_content)
    result = validate_judge_result(parsed_result)

    result["judge_model"] = judge_model
    result["judge_latency_sec"] = round(elapsed, 3)

    if response.usage is not None:
        result["judge_prompt_tokens"] = response.usage.prompt_tokens
        result["judge_completion_tokens"] = response.usage.completion_tokens
        result["judge_total_tokens"] = response.usage.total_tokens

    return result


def normalize_term_text(text: str) -> str:
    """Normalize punctuation variants before matching domain phrases."""
    normalized = text.lower()
    normalized = re.sub(r"[\u2010-\u2015]", "-", normalized)
    normalized = re.sub(r"[-_/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def calculate_mctd(generated_output: str, pdb_id: str | None) -> dict:
    """Calculate unique domain terms per 100 output words.

    Terms are counted once, so repeating technical vocabulary cannot inflate
    the score. Target-specific terms and dynamically detected residue labels
    are included in the numerator.
    """
    if not generated_output or not generated_output.strip():
        return {
            "score": 0.0,
            "unique_domain_terms": 0,
            "total_words": 0,
            "matched_terms": [],
        }

    words = WORD_PATTERN.findall(generated_output)
    total_words = len(words)
    normalized_output = normalize_term_text(generated_output)
    normalized_pdb_id = (pdb_id or "").strip().upper()
    vocabulary = (
        GENERAL_MEDCHEM_TERMS
        | TARGET_MEDCHEM_TERMS.get(normalized_pdb_id, set())
    )
    matched_terms = set()

    for term in vocabulary:
        normalized_term = normalize_term_text(term)
        term_pattern = (
            rf"(?<![A-Za-z0-9])"
            rf"{re.escape(normalized_term)}"
            rf"(?![A-Za-z0-9])"
        )
        if re.search(term_pattern, normalized_output):
            matched_terms.add(normalized_term)

    for residue in RESIDUE_PATTERN.findall(generated_output):
        canonical_residue = re.sub(r"[\s-]+", "", residue.lower())
        matched_terms.add(canonical_residue)

    score = (
        len(matched_terms) / total_words * 100
        if total_words
        else 0.0
    )
    return {
        "score": round(score, 3),
        "unique_domain_terms": len(matched_terms),
        "total_words": total_words,
        "matched_terms": sorted(matched_terms),
    }


def average_mctd(valid_runs: list[dict]) -> float | None:
    scores = [
        record["medical_chemical_term_density"]["score"]
        for record in valid_runs
        if "medical_chemical_term_density" in record
    ]
    return round(sum(scores) / len(scores), 3) if scores else None


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
            f"No semantic reference profile is configured for PDB {normalized_pdb_id!r}"
        )

    output_chunks = chunk_text(generated_output, words_per_chunk)
    if not output_chunks:
        raise ValueError("Cannot calculate semantic alignment for an empty output")
    reference_chunks = chunk_text(reference_text, words_per_chunk)

    output_embeddings = embedding_model.encode(
        output_chunks,
        normalize_embeddings=True,
    )
    reference_embeddings = embedding_model.encode(
        reference_chunks,
        normalize_embeddings=True,
    )
    output_vector = mean_embedding(output_embeddings)
    reference_vector = mean_embedding(reference_embeddings)

    return {
        "score": round(cosine_similarity(output_vector, reference_vector), 6),
        "reference_pdb_id": normalized_pdb_id,
        "embedding_model": str(getattr(
            embedding_model,
            "model_name_or_path",
            DEFAULT_EMBEDDING_MODEL,
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
        if "semantic_embedding_alignment" in record
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

    score_fields = [*RUBRIC_WEIGHTS, "overall_richness_score"]
    return {
        field: round(
            sum(float(rubric[field]) for rubric in scored_runs) / len(scored_runs),
            3,
        )
        for field in score_fields
    }


def benchmark_model(
    model_name: str,
    dataset: list,
    out_path: str = "benchmarks/results/terra-eval.json",
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    embedding_model = load_embedding_model(embedding_model_name)
    
    # Resume from existing progress if file already exists
    runs = []
    saved = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                saved = json.load(f)
                runs = saved.get("runs", [])
                print(f"Resuming benchmark: loaded {len(runs)} existing runs.")
        except Exception:
            runs = []

    completed_indices = {r["index"] for r in runs}

    def is_excluded(r):
        """Runs that shouldn't count toward the averages: hard errors and
        'bad' responses (empty output or content-filtered) that still came
        back as a 200 from the API."""
        return "error" in r or r.get("flagged", False)

    total_in_tokens = sum(r.get("prompt_tokens", 0) for r in runs if not is_excluded(r))
    total_out_tokens = sum(r.get("completion_tokens", 0) for r in runs if not is_excluded(r))
    total_latency = sum(r.get("latency_sec", 0.0) for r in runs if not is_excluded(r))
    total_cost = sum(r.get("cost_usd", 0.0) for r in runs if not is_excluded(r))

    # Backfill this deterministic metric for previously completed runs. This
    # lets resumed result files gain MCTD scores without regenerating outputs.
    for record in runs:
        if is_excluded(record) or "medical_chemical_term_density" in record:
            continue

        sample_index = record.get("index")
        if not isinstance(sample_index, int):
            print("Skipping MCTD backfill: record has no valid index.")
            continue

        dataset_position = sample_index - 1
        if not 0 <= dataset_position < len(dataset):
            print(
                f"Skipping MCTD backfill: index {sample_index} "
                "is outside the dataset."
            )
            continue

        output = record.get("output")
        if not output:
            continue

        sample = dataset[dataset_position]
        record["medical_chemical_term_density"] = calculate_mctd(
            generated_output=output,
            pdb_id=sample["protein"].get("pdb_id"),
        )

    # Backfill semantic alignment for previously completed generations using
    # the same cached local embedding model used for new records.
    for record in runs:
        if is_excluded(record) or "semantic_embedding_alignment" in record:
            continue

        sample_index = record.get("index")
        if not isinstance(sample_index, int):
            print("Skipping semantic backfill: record has no valid index.")
            continue

        dataset_position = sample_index - 1
        if not 0 <= dataset_position < len(dataset):
            print(
                f"Skipping semantic backfill: index {sample_index} "
                "is outside the dataset."
            )
            continue

        output = record.get("output")
        if not output:
            continue

        sample = dataset[dataset_position]
        record["semantic_embedding_alignment"] = calculate_semantic_alignment(
            generated_output=output,
            pdb_id=sample["protein"].get("pdb_id"),
            embedding_model=embedding_model,
        )

    # Backfill the API-based rubric separately from generation. Checkpoint each
    # attempt so a stopped run resumes at the next unscored record rather than
    # paying to judge the same outputs again.
    for record in runs:
        if is_excluded(record) or "structured_rubric" in record:
            continue

        sample_index = record.get("index")
        if not isinstance(sample_index, int):
            print("Skipping judge backfill: record has no valid index.")
            continue

        dataset_position = sample_index - 1
        if not 0 <= dataset_position < len(dataset):
            print(
                f"Skipping judge backfill: index {sample_index} "
                "is outside the dataset."
            )
            continue

        output = record.get("output")
        if not output:
            continue

        sample = dataset[dataset_position]
        try:
            record["structured_rubric"] = evaluate_scientific_richness(
                protein=sample["protein"],
                compound=sample["compound"],
                generated_output=output,
            )
            record.pop("judge_error", None)
            print(
                f"Judge backfill [{sample_index}/{len(dataset)}] "
                f"{record.get('compound_name')} | "
                f"{record['structured_rubric']['overall_richness_score']}/10"
            )
        except Exception as judge_error:
            record["judge_error"] = str(judge_error)
            print(
                f"Judge backfill failed [{sample_index}/{len(dataset)}] "
                f"{record.get('compound_name')}: {judge_error}"
            )

        saved["runs"] = runs
        with open(out_path, "w") as f:
            json.dump(saved, f, indent=2)

    for idx, sample in enumerate(dataset, start=1):
        if idx in completed_indices:
            continue

        protein = sample["protein"]
        compound = sample["compound"]
        prompt = format_pipeline_prompt(protein, compound)

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
                "pdb_id": protein.get("pdb_id"),
                "compound_name": compound.get("name"),
                "latency_sec": round(elapsed, 3),
                "prompt_tokens": in_tokens,
                "completion_tokens": out_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": round(cost, 6),
                "finish_reason": finish_reason,
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

                record["medical_chemical_term_density"] = calculate_mctd(
                    generated_output=content,
                    pdb_id=protein.get("pdb_id"),
                )
                record["semantic_embedding_alignment"] = calculate_semantic_alignment(
                    generated_output=content,
                    pdb_id=protein.get("pdb_id"),
                    embedding_model=embedding_model,
                )

                try:
                    record["structured_rubric"] = evaluate_scientific_richness(
                        protein=protein,
                        compound=compound,
                        generated_output=content,
                    )

                    print(
                        f"judge score: "
                        f"{record['structured_rubric']['overall_richness_score']}/10"
                    )
                except Exception as judge_error:
                    record["judge_error"] = str(judge_error)
                    print(
                        f"Judge failed [{idx}/{len(dataset)}] "
                        f"{compound.get('name')}: {judge_error}"
                    )

                print(
                    f"Completed [{idx}/{len(dataset)}] {record['compound_name']} | "
                    f"{record['latency_sec']}s | {usage.total_tokens} tokens"
                )

        except BadRequestError as e:
            elapsed = time.perf_counter() - start
            err_code = getattr(e, "code", "bad_request")
            print(f"Skipped [{idx}/{len(dataset)}] {compound.get('name')} | Flagged ({err_code}): {e.message}")
            record = {
                "index": idx,
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

        # Checkpoint to disk on every single sample
        valid_runs = [r for r in runs if not is_excluded(r)]
        n_valid = len(valid_runs)
        flagged_count = sum(1 for r in runs if r.get("flagged", False))
        error_count = sum(1 for r in runs if "error" in r)
        avg_cost = round(total_cost / n_valid, 6) if n_valid else 0
        summary = {
            "model": model_name,
            "total_samples": len(dataset),
            "completed_samples": len(runs),
            "valid_samples": n_valid,
            "flagged_samples": flagged_count,
            "error_samples": error_count,
            "avg_latency_sec": round(total_latency / n_valid, 3) if n_valid else 0,
            "avg_prompt_tokens": round(total_in_tokens / n_valid, 1) if n_valid else 0,
            "avg_completion_tokens": round(total_out_tokens / n_valid, 1) if n_valid else 0,
            "avg_cost_usd": avg_cost,
            "projected_cost_per_1k_runs_usd": round(avg_cost * 1000, 4) if n_valid else 0,
            "avg_structured_rubric": average_structured_rubric(valid_runs),
            "avg_medical_chemical_term_density": average_mctd(valid_runs),
            "avg_semantic_embedding_alignment": average_semantic_alignment(valid_runs),
            "runs": runs,
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Final recompute + write, so a fully-resumed run (nothing left to do in
    # the loop above) still gets the summary refreshed in the latest format.
    valid_runs = [r for r in runs if not is_excluded(r)]
    n_valid = len(valid_runs)
    flagged_count = sum(1 for r in runs if r.get("flagged", False))
    error_count = sum(1 for r in runs if "error" in r)
    avg_cost = round(total_cost / n_valid, 6) if n_valid else 0
    summary = {
        "model": model_name,
        "total_samples": len(dataset),
        "completed_samples": len(runs),
        "valid_samples": n_valid,
        "flagged_samples": flagged_count,
        "error_samples": error_count,
        "avg_latency_sec": round(total_latency / n_valid, 3) if n_valid else 0,
        "avg_prompt_tokens": round(total_in_tokens / n_valid, 1) if n_valid else 0,
        "avg_completion_tokens": round(total_out_tokens / n_valid, 1) if n_valid else 0,
        "avg_cost_usd": avg_cost,
        "projected_cost_per_1k_runs_usd": round(avg_cost * 1000, 4) if n_valid else 0,
        "avg_structured_rubric": average_structured_rubric(valid_runs),
        "avg_medical_chemical_term_density": average_mctd(valid_runs),
        "avg_semantic_embedding_alignment": average_semantic_alignment(valid_runs),
        "runs": runs,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved final benchmark results to {out_path}")


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
      model_name="gpt-5.6-terra",
      dataset=combined_dataset,
      out_path="benchmarks/results/terra-eval.json",
  )
