"""Step 4 — Binding activity ranking via AutoDock Vina."""

import os
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.error import HTTPError

import numpy as np
import pandas as pd

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


def fetch_pdb_file(pdb_id: str) -> str:
    """Fetch PDB file text from RCSB."""
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
    """Find binding site center from co-crystallized ligand."""
    residue_coords = {}

    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        res_name = line[17:20].strip()
        if res_name in SKIP_LIGANDS or res_name in ("WAT", "K"):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            residue_coords.setdefault(res_name, []).append((x, y, z))
        except (ValueError, IndexError):
            continue

    if not residue_coords:
        logger.warning("No ligand found, using protein geometric center")
        return (0.0, 0.0, 0.0)

    best_res = max(residue_coords, key=lambda r: len(residue_coords[r]))
    coords = np.array(residue_coords[best_res])
    center = coords.mean(axis=0)
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

    # Pre-rank by ChEMBL activity to select top candidates
    with_activity = [c for c in ro5_state if c.get("activity_value") and c["activity_value"] > 0]
    without_activity = [c for c in ro5_state if not c.get("activity_value") or c["activity_value"] <= 0]

    with_activity.sort(key=lambda x: x["activity_value"])
    pre_ranked = with_activity + without_activity

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
            "ChEMBL Activity": (
                f"{c['activity_value']:.0f} {c.get('activity_units', '')}"
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
