"""Shared helpers: logging, OpenAI / Anthropic clients, PDB ligand / UniProt lookups."""

import logging
import os

import anthropic
import requests
from openai import OpenAI

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


# ── OpenAI ────────────────────────────────────────────────────────────

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# ── Anthropic ─────────────────────────────────────────────────────────

def get_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


# ── AI backend ────────────────────────────────────────────────────────

CLAUDE_MAX_TOKENS = 4096


def ai_complete(prompt: str) -> str | None:
    """Send *prompt* to the configured backend and return its text.

    The backend is chosen by AI_PROVIDER in .env ("openai" by default,
    "anthropic" for Claude). Returns None when the relevant API key is
    missing, so callers degrade the same way they did when
    get_openai_client() returned None.

    Env is read on every call, not at import time: app.py imports the tabs
    package before it calls load_dotenv(), so anything read at module level
    would miss the .env values entirely.
    """
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()

    if provider == "anthropic":
        client = get_anthropic_client()
        if client is None:
            return None
        resp = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5"),
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    client = get_openai_client()
    if client is None:
        return None
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07"),
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


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
