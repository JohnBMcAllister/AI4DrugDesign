"""Shared helpers: logging, AI client, PDB ligand / UniProt lookups."""

import logging
import os

import requests
from anthropic import Anthropic

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drug_discovery_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Anthropic ────────────────────────────────────────────────────────────

AI_MODEL = "claude-sonnet-5"
AI_MAX_TOKENS = 8192
 
 
def get_ai_client():
    """Return an Anthropic client, or None when no API key is configured."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)
 
 
def generate_text(prompt: str, max_tokens: int = AI_MAX_TOKENS) -> str:
    """Send a single-turn prompt to the configured model and return its text.
 
    Raises RuntimeError when no API key is configured, and lets any API
    error propagate so the caller can report it.
    """
    client = get_ai_client()
    if client is None:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — cannot generate explanation"
        )
 
    response = client.messages.create(
        model=AI_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if block.type == "text"
    )


# ── Constants ─────────────────────────────────────────────────────────

SKIP_LIGANDS = frozenset({
    "HOH", "SO4", "PO4", "GOL", "EDO", "ACT", "FMT", "IOD",
    "CL", "MG", "ZN", "CA", "NA", "PEG", "DMS",
})


# ── PDB helpers ───────────────────────────────────────────────────────

def fetch_pdb_ligands(pdb_id: str, entry: dict | None = None) -> list[dict]:
    """Fetch non-polymer ligands from a PDB entry.

    If *entry* is already available (from a prior /core/entry call), pass it
    to avoid a redundant request.
    """
    base = "https://data.rcsb.org/rest/v1/core"
    if entry is None:
        try:
            r = requests.get(f"{base}/entry/{pdb_id}", timeout=15)
            if not r.ok:
                return []
            entry = r.json()
        except Exception as exc:
            logger.warning(f"Error fetching PDB entry for ligands: {exc}")
            return []

    non_poly_ids = (
        entry.get("rcsb_entry_container_identifiers", {})
        .get("non_polymer_entity_ids") or []
    )
    ligands: list[dict] = []
    for eid in non_poly_ids[:10]:
        try:
            er = requests.get(
                f"{base}/nonpolymer_entity/{pdb_id}/{eid}", timeout=10
            )
            if er.ok:
                d = er.json().get("pdbx_entity_nonpoly", {})
                comp_id = d.get("comp_id", "")
                name = d.get("name", comp_id)
                if comp_id and comp_id not in SKIP_LIGANDS:
                    ligands.append({"comp_id": comp_id, "name": name})
        except Exception as exc:
            logger.debug(f"Error fetching ligand entity {eid}: {exc}")
    return ligands


def get_uniprot_from_pdb(pdb_id: str) -> str | None:
    """Map PDB ID to UniProt accession via SIFTS."""
    try:
        r = requests.get(
            f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}",
            timeout=15,
        )
        if r.ok:
            data = r.json().get(pdb_id.lower(), {}).get("UniProt", {})
            if data:
                return list(data.keys())[0]
    except Exception:
        pass
    return None
