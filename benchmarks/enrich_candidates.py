"""Backfill docking, ADME and binding-site data onto the existing candidates.

Docks the SAME compounds already in benchmarks/*_candidates.json, so the
benchmark set does not change and the old-prompt run stays comparable. Nothing
is re-discovered from ChEMBL.

Run from the repo root:

    uv run python benchmarks/enrich_candidates.py

Each file is backed up to <name>.bak before being rewritten.
"""

import functools
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tabs.pipeline.docking as docking_module
from tabs.pipeline.detail import _get_pocket_contacts
from tabs.pipeline.docking import (
    calculate_optimal_workers,
    find_binding_center,
    prepare_receptor_pdbqt,
)
from tabs.pipeline.docking_worker import dock_single_compound
from tabs.pipeline.filters import apply_adme_filter, apply_rule_of_5
from tabs.pipeline.helpers import logger

# _get_pocket_contacts re-fetches the PDB for every compound. Cache it for this
# script only; the import inside that function resolves at call time, so
# patching the module attribute is enough.
docking_module.fetch_pdb_file = functools.lru_cache(maxsize=8)(
    docking_module.fetch_pdb_file
)


def dock_compounds(compounds: list, pdb_id: str) -> None:
    """Dock in place: sets vina_affinity and docked_sdf on each compound."""
    pdb_text = docking_module.fetch_pdb_file(pdb_id)
    center = find_binding_center(pdb_text)
    logger.info(f"{pdb_id}: docking {len(compounds)} compounds at {center}")

    tmpdir = tempfile.mkdtemp()
    try:
        pdb_path = os.path.join(tmpdir, "receptor.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_text)

        receptor_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")
        prepare_receptor_pdbqt(pdb_path, receptor_pdbqt)
        if not os.path.exists(receptor_pdbqt):
            raise FileNotFoundError(f"Receptor PDBQT missing for {pdb_id}")

        dock_args = [
            (i, compounds[i]["smiles"], receptor_pdbqt, center)
            for i in range(len(compounds))
        ]
        workers = min(calculate_optimal_workers(), len(dock_args))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(dock_single_compound, a): a[0] for a in dock_args}
            for future in as_completed(futures):
                idx, affinity, docked_sdf = future.result()
                compounds[idx]["vina_affinity"] = round(affinity, 2)
                compounds[idx]["docked_sdf"] = docked_sdf
                compounds[idx]["docked"] = True
                state = "failed" if affinity == float("inf") else f"{affinity:.2f}"
                logger.info(f"  {compounds[idx].get('name')}: {state}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def enrich_file(path: str) -> None:
    with open(path) as f:
        samples = json.load(f)
    if not samples:
        print(f"{path}: empty, skipped")
        return

    pdb_id = samples[0]["protein"]["pdb_id"]
    print(f"\n=== {path} ({pdb_id}, {len(samples)} compounds) ===")

    # Working copies, keyed back to the originals by SMILES.
    compounds = []
    for s in samples:
        c = dict(s["compound"])
        c.setdefault("chembl_id", "")
        c.setdefault("activity_units", c.get("activity_units") or "")
        compounds.append(c)

    # Step 3 — adds mw, logp, hbd, hba, ro5_violations.
    # With only a handful of compounds the adaptive thresholds keep all of
    # them, so this acts as a property annotator rather than a filter.
    _, after_ro5, _ = apply_rule_of_5(compounds)
    after_ro5 = after_ro5 or compounds

    # Step 4 — adds vina_affinity and docked_sdf.
    dock_compounds(after_ro5, pdb_id)

    # Step 5 — adds tpsa, rot_bonds, adme_score.
    _, after_adme, _ = apply_adme_filter(after_ro5)
    after_adme = after_adme or after_ro5

    # Rank the successful docks by affinity, as the pipeline does.
    docked = [
        c for c in after_adme
        if c.get("vina_affinity") is not None and c["vina_affinity"] != float("inf")
    ]
    docked.sort(key=lambda c: c["vina_affinity"])
    for rank, c in enumerate(docked, start=1):
        c["binding_rank"] = rank

    by_smiles = {c["smiles"]: c for c in after_adme}

    enriched, no_pose = [], 0
    for s in samples:
        smiles = s["compound"]["smiles"]
        c = by_smiles.get(smiles)
        if c is None:
            print(f"  WARNING: {s['compound'].get('name')} dropped by a filter, left as-is")
            enriched.append(s)
            continue

        c.pop("docked", None)
        contacts = _get_pocket_contacts(c, pdb_id) if c.get("docked_sdf") else None
        if not contacts:
            no_pose += 1

        enriched.append({
            "protein": s["protein"],
            "compound": c,
            "contacts": contacts,
        })

    shutil.copy(path, path + ".bak")
    with open(path, "w") as f:
        json.dump(enriched, f, indent=2)

    affinities = [c["vina_affinity"] for c in docked]
    span = f"{min(affinities):.2f} to {max(affinities):.2f}" if affinities else "none"
    print(f"  docked {len(docked)}/{len(samples)}  affinity {span} kcal/mol")
    print(f"  contacts resolved for {len(samples) - no_pose}/{len(samples)}")
    print(f"  wrote {path} (backup at {path}.bak)")


if __name__ == "__main__":
    paths = sorted(glob("benchmarks/*_candidates.json"))
    if not paths:
        raise SystemExit("No benchmarks/*_candidates.json found — run from the repo root.")

    for path in paths:
        try:
            enrich_file(path)
        except Exception as exc:
            logger.exception(f"{path} failed: {exc}")
            print(f"  FAILED: {exc}")

    print("\nDone. Re-run the benchmark to a new --out path.")
