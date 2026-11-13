"""Step 4 — Binding activity ranking via AutoDock Vina."""

import functools
import os
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.error import HTTPError

import numpy as np
import pandas as pd
from rdkit import Chem

from tabs.pipeline.docking_worker import dock_single_compound
from tabs.pipeline.helpers import SKIP_LIGANDS, logger


def calculate_optimal_workers() -> int:
    """Calculate optimal number of parallel workers based on available resources."""
    import multiprocessing

    total_cpus = multiprocessing.cpu_count()

    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / (1024**3)

        ram_based_workers = int(available_ram_gb / 0.5)

        cpu_usage = psutil.cpu_percent(interval=0.1)
        if cpu_usage > 50:
            available_cpu_fraction = (100 - cpu_usage) / 100
            cpu_based_workers = max(1, int(total_cpus * available_cpu_fraction))
        else:
            cpu_based_workers = max(1, total_cpus - 2)

        optimal = min(ram_based_workers, cpu_based_workers)

        logger.info(
            f"Resource detection: {total_cpus} CPUs, {available_ram_gb:.1f}GB RAM available, "
            f"CPU usage {cpu_usage:.0f}% → {optimal} workers"
        )

    except ImportError:
        optimal = max(1, total_cpus - 2)
        logger.info(
            f"psutil not available, using conservative estimate: "
            f"{total_cpus} CPUs → {optimal} workers"
        )

    return max(1, min(optimal, 32))


def ligand_efficiency(affinity: float, smiles: str) -> float:
    """Ligand efficiency = -affinity / heavy atom count. Higher is better.

    Negative for a positive (repulsive) Vina score — a compound that does not
    bind has no efficiency to report. Matches the fallback calculation in
    detail.build_compound_context, so both discovery paths agree.
    """
    mol = Chem.MolFromSmiles(smiles)
    hac = mol.GetNumHeavyAtoms() if mol else 1
    return round(-affinity / max(hac, 1), 3)


# ── Experimental potency normalisation ────────────────────────────────
#
# ChEMBL mixes assay types and units in standard_value: an IC50 in nM sits in
# the same column as a percent inhibition and a unitless relative binding
# affinity. Sorting that column directly compares 0.003 "%" against 0.35 "nM"
# and buries real inhibitors behind whatever happens to have a small number
# attached — which matters because this ordering decides which compounds get
# docked at all.

_POTENCY_TYPES = frozenset({"IC50", "KI", "KD", "EC50", "XC50", "AC50"})

# Multipliers onto nanomolar.
_UNITS_TO_NM = {
    "PM": 1e-3,
    "NM": 1.0,
    "UM": 1e3,
    "MM": 1e6,
    "M": 1e9,
}


def _normalize_units(units) -> str:
    """Upper-case the unit string, folding both micro signs onto 'u'."""
    return (units or "").strip().replace("µ", "u").replace("μ", "u").upper()


def potency_nm(compound: dict) -> float | None:
    """Return the compound's potency in nanomolar, or None if not comparable.

    None means "this activity value cannot be ranked against a concentration" —
    a percent inhibition, a relative binding affinity, a missing unit — not
    "this compound is inactive".
    """
    activity_type = (compound.get("activity_type") or "").strip().upper()
    if activity_type not in _POTENCY_TYPES:
        return None

    factor = _UNITS_TO_NM.get(_normalize_units(compound.get("activity_units")))
    if factor is None:
        return None

    value = compound.get("activity_value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None

    return value * factor


@functools.lru_cache(maxsize=16)
def fetch_pdb_file(pdb_id: str) -> str:
    """Fetch PDB file text from RCSB.

    Cached: a single pipeline run asks for the same structure from the docking
    step, the receptor-interaction viewer and the pocket-contact analysis.
    """
    pdb_id = pdb_id.strip().upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except HTTPError as e:
        if e.code == 404:
            raise Exception(f"PDB ID '{pdb_id}' not found on RCSB.")
        raise Exception(f"Failed to fetch PDB '{pdb_id}': HTTP {e.code}")


def find_binding_center(pdb_text: str) -> tuple[float, float, float]:
    """Find the binding-site centre from ONE copy of the co-crystallised ligand.

    Ligand atoms are grouped by chain, not by residue name and not by residue
    instance:

    * Grouping by residue name averages every copy in the file together. For a
      structure with the ligand bound in several chains that puts the centre
      between the pockets — 31 A off for 1IEP, 34 A off for 2HYY, against a
      20 A box.
    * Grouping by residue instance splits a ligand that was deposited as
      several linked chemical components, so only the largest piece is used.
      6LU7's inhibitor N3 is three components (02J, PJE, 010) and only the
      13-atom middle fragment would be seen.

    Grouping by chain keeps a multi-component ligand whole while still
    selecting a single copy out of a multimer.
    """
    ligand_coords: dict[str, list] = {}
    ligand_names: dict[str, set] = {}
    protein_coords: list = []

    for line in pdb_text.splitlines():
        record = line[:6]
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except (ValueError, IndexError):
            continue

        if record == "ATOM  ":
            protein_coords.append(xyz)
            continue
        if record != "HETATM":
            continue

        res_name = line[17:20].strip()
        if res_name in SKIP_LIGANDS or res_name in ("WAT", "K"):
            continue
        chain = line[21] if len(line) > 21 else " "
        ligand_coords.setdefault(chain, []).append(xyz)
        ligand_names.setdefault(chain, set()).add(res_name)

    if ligand_coords:
        # Chain carrying the most ligand atoms; on a tie, the first in the file.
        best = max(ligand_coords, key=lambda k: len(ligand_coords[k]))
        atoms = np.array(ligand_coords[best])
        center = atoms.mean(axis=0)
        spread = float(np.linalg.norm(atoms.max(axis=0) - atoms.min(axis=0)))
        logger.info(
            f"Binding center from chain {best} "
            f"({len(atoms)} ligand atoms, components {sorted(ligand_names[best])}, "
            f"{spread:.1f} A spread); {len(ligand_coords)} ligand-bearing chain(s)"
        )
        if spread > 25:
            logger.warning(
                f"Ligand atoms in chain {best} span {spread:.1f} A, which is wider "
                "than the 20 A docking box. That chain may hold more than one "
                "ligand, so the box may not cover the intended site."
            )
    elif protein_coords:
        center = np.array(protein_coords).mean(axis=0)
        logger.warning(
            "No ligand found, using protein geometric center "
            "(docking box may miss the pocket)"
        )
    else:
        logger.warning("No coordinates found, using origin")
        return (0.0, 0.0, 0.0)

    return round(float(center[0]), 3), round(float(center[1]), 3), round(float(center[2]), 3)


def prepare_receptor_pdbqt(pdb_path: str, output_path: str) -> None:
    """Prepare receptor PDBQT from PDB file."""
    cleaned_path = pdb_path.replace(".pdb", "_clean.pdb")
    with open(pdb_path) as f:
        lines = f.readlines()
    with open(cleaned_path, "w") as f:
        for line in lines:
            if line.startswith(("ATOM", "TER", "END", "MODEL", "ENDMDL", "REMARK", "CRYST1")):
                f.write(line)

    mk_script = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
    if mk_script is None:
        raise Exception("mk_prepare_receptor not found. Install with: pip install meeko")

    output_basename = output_path.removesuffix(".pdbqt")
    cmd = [mk_script, "-i", cleaned_path, "-o", output_basename, "-p", "--allow_bad_res", "--default_altloc", "A"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise Exception(f"Receptor prep failed: {result.stderr[:200]}")


def rank_by_activity(ro5_state, protein_info_state=None, num_dock=50):
    """Rank compounds using AutoDock Vina docking scores."""
    num_dock = int(num_dock)
    logger.info("=== STEP 4: Ranking by binding activity (AutoDock Vina) ===")
    if not ro5_state:
        logger.warning("No compounds provided for docking")
        return "**Step 4 – Docking:** ❌ No compounds to rank", None, None

    if not protein_info_state:
        logger.error("No protein info for docking")
        return "**Step 4 – Docking:** ❌ No protein info", None, None

    total = len(ro5_state)
    pdb_id = protein_info_state["pdb_id"]

    # Pre-rank by experimental potency to choose which compounds to dock.
    # Only values that normalise to a concentration are comparable; everything
    # else keeps its original order behind them rather than being interleaved
    # on a meaningless cross-unit comparison.
    with_potency: list[dict] = []
    without_potency: list[dict] = []
    for c in ro5_state:
        nm = potency_nm(c)
        if nm is None:
            without_potency.append(c)
        else:
            c["potency_nm"] = round(nm, 4)
            with_potency.append(c)

    with_potency.sort(key=lambda c: c["potency_nm"])
    pre_ranked = with_potency + without_potency

    logger.info(
        f"Pre-ranking {total} compounds: {len(with_potency)} with comparable "
        f"potency, {len(without_potency)} without (kept in discovery order)"
    )
    if with_potency:
        logger.info(
            f"Most potent: {with_potency[0].get('name')} at "
            f"{with_potency[0]['potency_nm']:.3f} nM"
        )
    elif without_potency:
        logger.warning(
            "No compound had a potency value in comparable units — the docking "
            "selection is effectively the ChEMBL discovery order."
        )

    # Dock top N compounds
    num_dock = min(num_dock, len(pre_ranked))
    to_dock = pre_ranked[:num_dock]

    # Calculate optimal workers based on available resources
    num_workers = min(calculate_optimal_workers(), num_dock)
    est_time = (num_dock / num_workers) * 0.5
    logger.info(f"Docking {num_dock} compounds in parallel ({num_workers} workers) - ETA: {est_time:.1f}min")

    try:
        pdb_text = fetch_pdb_file(pdb_id)
        center = find_binding_center(pdb_text)
        logger.info(f"Binding center: {center}")

        tmpdir = tempfile.mkdtemp()
        try:
            pdb_path = os.path.join(tmpdir, "receptor.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb_text)

            receptor_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")
            prepare_receptor_pdbqt(pdb_path, receptor_pdbqt)

            if not os.path.exists(receptor_pdbqt):
                raise FileNotFoundError(
                    f"Receptor PDBQT not found at {receptor_pdbqt} after preparation"
                )
            logger.info("Receptor prepared, starting parallel docking...")

            dock_args = [
                (i, to_dock[i]["smiles"], receptor_pdbqt, center)
                for i in range(num_dock)
            ]

            completed = 0
            failed = 0
            logger.info("Creating ProcessPoolExecutor...")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                logger.info("Submitting docking jobs...")
                futures = {executor.submit(dock_single_compound, arg): arg[0] for arg in dock_args}
                logger.info("All jobs submitted, waiting for results...")

                for future in as_completed(futures):
                    idx, affinity, docked_sdf = future.result()
                    to_dock[idx]["vina_affinity"] = round(affinity, 2)
                    to_dock[idx]["docked_sdf"] = docked_sdf
                    to_dock[idx]["docked"] = True
                    to_dock[idx]["ligand_efficiency"] = ligand_efficiency(
                        affinity, to_dock[idx]["smiles"]
                    )

                    if affinity == float('inf'):
                        failed += 1

                    completed += 1
                    if completed % 5 == 0 or completed == num_dock:
                        pct = (completed / num_dock) * 100
                        logger.info(f"Docking progress: {completed}/{num_dock} ({pct:.0f}%) - {failed} failed")

            logger.info(f"Parallel docking complete: {num_dock} compounds ({failed} failed)")

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        logger.exception(f"Docking failed: {e}")
        return f"**Step 4 – Docking:** ❌ {e}", None, None

    # Keep ONLY successfully docked compounds
    docked = [
        c for c in pre_ranked
        if c.get("docked") and c.get("vina_affinity") != float('inf')
    ]

    docked.sort(key=lambda x: x["vina_affinity"])
    ranked = docked

    for i, c in enumerate(ranked):
        c["binding_rank"] = i + 1

    rows = []
    for c in ranked[:500]:
        rows.append({
            "Rank": c["binding_rank"],
            "Name": c["name"][:25],
            "Vina (kcal/mol)": (
                f"{c['vina_affinity']:.2f}"
                if c.get("vina_affinity") and c["vina_affinity"] != float('inf')
                else "Not docked"
            ),
            "Potency": (
                f"{c['potency_nm']:.3g} nM"
                if c.get("potency_nm") is not None
                else "not comparable"
            ),
            "ChEMBL Activity": (
                f"{c['activity_value']:.0f} {c.get('activity_units') or ''}".strip()
                if c.get("activity_value")
                else "N/A"
            ),
            "MW": c.get("mw", ""),
            "LogP": c.get("logp", ""),
        })

    successful_docks = len(ranked)
    failed_docks = num_dock - successful_docks
    status = f"**Step 4 – Docking:** ✅ {successful_docks}/{num_dock} compounds docked with AutoDock Vina"
    if failed_docks > 0:
        status += f" ({failed_docks} failed and discarded)"

    df = pd.DataFrame(rows) if rows else None
    logger.info(f"=== STEP 4 COMPLETE: {successful_docks} successful, {failed_docks} failed ===")
    return status, ranked, df