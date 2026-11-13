"""GrowMax — fragment-based molecule growing for drug candidate generation."""

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from rdkit.SimDivFilters import rdSimDivPickers

from tabs.pipeline.helpers import logger

# [c;H1:1] matches an aromatic carbon with exactly 1 implicit H — no explicit H needed.
# [N;H1:1] matches N with 1 implicit H (pyrrole-type NH, piperidine NH, etc.).
_REACTION_SMARTS = [
    ("methyl",      "[c;H1:1]>>[c:1]C"),
    ("methoxy",     "[c;H1:1]>>[c:1]OC"),
    ("amino",       "[c;H1:1]>>[c:1]N"),
    ("hydroxyl",    "[c;H1:1]>>[c:1]O"),
    ("fluoro",      "[c;H1:1]>>[c:1]F"),
    ("chloro",      "[c;H1:1]>>[c:1]Cl"),
    ("cyano",       "[c;H1:1]>>[c:1]C#N"),
    ("amide",       "[c;H1:1]>>[c:1]C(=O)N"),
    ("carboxyl",    "[c;H1:1]>>[c:1]C(=O)O"),
    ("CF3",         "[c;H1:1]>>[c:1]C(F)(F)F"),
    ("phenyl",      "[c;H1:1]>>[c:1]c1ccccc1"),
    ("pyridyl",     "[c;H1:1]>>[c:1]c1ccncc1"),
    ("piperidyl",   "[c;H1:1]>>[c:1]C1CCNCC1"),
    ("morpholinyl", "[c;H1:1]>>[c:1]C1CCOCC1"),
    ("sulfonamide", "[c;H1:1]>>[c:1]S(=O)(=O)N"),
    ("acetamide",   "[N;H1:1]>>[N:1]C(=O)C"),
]

_COMPILED_REACTIONS = [
    (label, ReactionFromSmarts(smarts))
    for label, smarts in _REACTION_SMARTS
]


def _is_drug_like(mol) -> bool:
    """Loose drug-likeness gate to prune clearly non-drug variants before docking."""
    return (
        Descriptors.MolWt(mol) <= 550
        and Descriptors.MolLogP(mol) <= 6
        and Descriptors.NumHDonors(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 11
    )


def _grow_one_round(parent_smiles_list, seen, max_variants, round_num):
    """Grow every molecule in parent_smiles_list by one step. Returns new (smiles, label) pairs."""
    new_variants: list[tuple[str, str]] = []
    label_suffix = f"_r{round_num}"

    for parent_smiles in parent_smiles_list:
        parent = Chem.MolFromSmiles(parent_smiles)
        if parent is None:
            continue
        for label, rxn in _COMPILED_REACTIONS:
            try:
                products = rxn.RunReactants((parent,))
            except Exception:
                continue
            for prod_tuple in products:
                if not prod_tuple:
                    continue
                prod = prod_tuple[0]
                try:
                    Chem.SanitizeMol(prod)
                except Exception:
                    continue
                if not _is_drug_like(prod):
                    continue
                canon = Chem.MolToSmiles(prod)
                if canon in seen:
                    continue
                seen.add(canon)
                new_variants.append((canon, label + label_suffix))
                if len(new_variants) >= max_variants:
                    return new_variants

    return new_variants


def generate_variants(
    seed_smiles: str, max_variants: int = 200, rounds: int = 2
) -> list[tuple[str, str]]:
    """
    Iteratively grow the seed fragment for `rounds` steps.

    Each round takes the previous round's new variants as parents and grows them
    one step further. Returns at most max_variants (canonical_smiles, label) tuples.
    """
    seed = Chem.MolFromSmiles(seed_smiles)
    if seed is None:
        raise ValueError(f"Invalid SMILES: {seed_smiles}")

    seen: set[str] = {Chem.MolToSmiles(seed)}
    all_variants: list[tuple[str, str]] = []
    current_gen = [Chem.MolToSmiles(seed)]

    for round_num in range(1, rounds + 1):
        remaining = max_variants - len(all_variants)
        if remaining <= 0:
            break
        new = _grow_one_round(current_gen, seen, remaining, round_num)
        all_variants.extend(new)
        current_gen = [smi for smi, _ in new]
        logger.info(f"GrowMax round {round_num}: {len(new)} new variants (total {len(all_variants)})")
        if not current_gen:
            break

    return all_variants


def fetch_growmax_compounds(seed_smiles: str, max_variants: int = 200):
    """
    Generate round 1 seed variants from a fragment.

    Returns (status, compounds, df) — identical format to fetch_compounds.
    Score-guided growth across subsequent rounds is handled by run_score_guided_growth
    during the docking step.
    """
    logger.info("=== STEP 2 (GrowMax): Generating seed variants ===")

    seed_smiles = (seed_smiles or "").strip()
    if not seed_smiles:
        return "**Step 2 – GrowMax:** ❌ Enter a seed SMILES first", None, None

    if Chem.MolFromSmiles(seed_smiles) is None:
        return "**Step 2 – GrowMax:** ❌ Invalid SMILES string", None, None

    try:
        variants = generate_variants(seed_smiles, max_variants, rounds=1)
    except Exception as exc:
        logger.exception(f"GrowMax variant generation failed: {exc}")
        return f"**Step 2 – GrowMax:** ❌ {exc}", None, None

    if not variants:
        return (
            "**Step 2 – GrowMax:** ⚠️ No valid variants generated — "
            "try a different seed fragment",
            None,
            None,
        )

    compounds = [
        {
            "smiles": smi,
            "name": f"{label}_{i + 1}",
            "chembl_id": "GrowMax",
            "activity_type": "fragment-grown",
            "activity_value": None,
            "activity_units": "",
        }
        for i, (smi, label) in enumerate(variants)
    ]

    rows = [
        {
            "Name": c["name"],
            "SMILES": c["smiles"][:60] + ("..." if len(c["smiles"]) > 60 else ""),
        }
        for c in compounds
    ]

    status = f"**Step 2 – GrowMax:** ✅ Generated {len(compounds)} seed variants (round 1)"
    logger.info(f"=== STEP 2 (GrowMax) COMPLETE: {len(compounds)} seed variants ===")
    return status, compounds, pd.DataFrame(rows)


def _ligand_efficiency(affinity: float, smiles: str) -> float:
    """Ligand efficiency = |affinity| / heavy atom count. Higher is better."""
    mol = Chem.MolFromSmiles(smiles)
    hac = mol.GetNumHeavyAtoms() if mol else 1
    return round(abs(affinity) / max(hac, 1), 3)


def _diverse_top_k(candidates: list, top_k: int) -> list:
    """
    Select top_k diverse parents using MaxMin Tanimoto picking seeded at the
    best-LE compound. Candidates must be pre-sorted by LE descending.

    This prevents all growth parents converging on one chemical series when a
    single scaffold dominates the LE ranking.
    """
    if len(candidates) <= top_k:
        return candidates

    fps = []
    valid = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps.append(fp)
        valid.append(c)

    if len(valid) <= top_k:
        return valid

    picker = rdSimDivPickers.MaxMinPicker()
    # firstPicks=[0] seeds with the best-LE compound so it's always included
    indices = list(picker.LazyBitVectorPick(fps, len(valid), top_k, firstPicks=[0]))
    selected = [valid[i] for i in indices]
    logger.info(
        f"Diversity picker: selected {len(selected)} parents from {len(valid)} candidates"
    )
    return selected


def run_score_guided_growth(
    round1_compounds: list,
    protein_info: dict,
    rounds: int = 3,
    dock_per_round: int = 20,
    top_k: int = 10,
):
    """
    Score-guided GrowMax: dock → rank by LE → grow top K → dock → repeat.

    Generator — yields (status, None, None) progress updates during each round,
    then a final (status, all_docked, df) on completion. Gradio streams each
    yield to the UI so the user sees round-by-round progress.

    Parent selection uses ligand efficiency (LE = |affinity| / HAC) rather than
    raw affinity, so smaller compounds that bind efficiently are preferred over
    large compounds with marginally better scores.
    """
    import os
    import shutil
    import tempfile
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tabs.pipeline.docking import (
        calculate_optimal_workers,
        fetch_pdb_file,
        find_binding_center,
        prepare_receptor_pdbqt,
    )
    from tabs.pipeline.docking_worker import dock_single_compound
    from tabs.pipeline.filters import apply_rule_of_5

    if not round1_compounds:
        yield "**Step 4 – Docking:** ❌ No seed compounds to grow from", None, None
        return
    if not protein_info:
        yield "**Step 4 – Docking:** ❌ No protein info available", None, None
        return

    logger.info(
        f"=== Score-guided GrowMax: {rounds} rounds, "
        f"{dock_per_round}/round, top {top_k} grown by LE ==="
    )

    pdb_id = protein_info["pdb_id"]
    yield f"**Step 4 – Docking:** 🔄 Preparing receptor ({pdb_id})...", None, None

    try:
        pdb_text = fetch_pdb_file(pdb_id)
        center = find_binding_center(pdb_text)
    except Exception as exc:
        yield f"**Step 4 – Docking:** ❌ Failed to fetch receptor: {exc}", None, None
        return

    tmpdir = tempfile.mkdtemp()
    all_docked: list[dict] = []

    try:
        pdb_path = os.path.join(tmpdir, "receptor.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_text)
        receptor_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")
        prepare_receptor_pdbqt(pdb_path, receptor_pdbqt)

        seen_smiles: set[str] = {c["smiles"] for c in round1_compounds}
        current_pool = sorted(
            round1_compounds,
            key=lambda c: QED.qed(Chem.MolFromSmiles(c["smiles"])) if Chem.MolFromSmiles(c["smiles"]) else 0.0,
            reverse=True,
        )

        for round_num in range(1, rounds + 1):
            if not current_pool:
                logger.warning(f"GrowMax round {round_num}: empty pool, stopping early")
                break

            to_dock = current_pool[:dock_per_round]
            yield (
                f"**Step 4 – Docking:** 🔄 Round {round_num}/{rounds}: "
                f"docking {len(to_dock)} compounds...",
                None, None,
            )
            logger.info(f"GrowMax round {round_num}: docking {len(to_dock)} compounds")

            num_workers = min(calculate_optimal_workers(), len(to_dock))
            dock_args = [
                (i, to_dock[i]["smiles"], receptor_pdbqt, center)
                for i in range(len(to_dock))
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(dock_single_compound, arg): arg[0]
                    for arg in dock_args
                }
                for future in as_completed(futures):
                    idx, affinity, docked_sdf = future.result()
                    to_dock[idx]["vina_affinity"] = round(affinity, 2)
                    to_dock[idx]["docked_sdf"] = docked_sdf
                    to_dock[idx]["docked"] = True
                    to_dock[idx]["growth_round"] = round_num
                    to_dock[idx]["ligand_efficiency"] = _ligand_efficiency(
                        affinity, to_dock[idx]["smiles"]
                    )

            round_docked = [
                c for c in to_dock
                if c.get("docked") and c.get("vina_affinity") != float("inf")
            ]
            # Sort by LE descending — efficient binders make better growth parents
            round_docked.sort(key=lambda x: x.get("ligand_efficiency", 0), reverse=True)
            all_docked.extend(round_docked)

            best = round_docked[0] if round_docked else None
            best_str = (
                f"best −{abs(best['vina_affinity']):.1f} kcal/mol "
                f"(LE {best['ligand_efficiency']:.2f})"
                if best else "none docked"
            )
            yield (
                f"**Step 4 – Docking:** 🔄 Round {round_num}/{rounds} complete — "
                f"{len(round_docked)} docked, {best_str}",
                None, None,
            )
            logger.info(f"GrowMax round {round_num}: {len(round_docked)} docked, {best_str}")

            if round_num >= rounds:
                break

            top_parents = _diverse_top_k(round_docked, top_k)
            if not top_parents:
                logger.warning(f"GrowMax round {round_num}: no successful docks, stopping")
                break

            yield (
                f"**Step 4 – Docking:** 🔄 Round {round_num}/{rounds}: "
                f"growing from {len(top_parents)} diverse parents...",
                None, None,
            )

            next_variants: list[dict] = []
            for parent in top_parents:
                for smi, label in generate_variants(parent["smiles"], max_variants=30, rounds=1):
                    if smi not in seen_smiles:
                        seen_smiles.add(smi)
                        next_variants.append({
                            "smiles": smi,
                            "name": f"{label}_r{round_num + 1}",
                            "chembl_id": "GrowMax",
                            "activity_type": "fragment-grown",
                            "activity_value": None,
                            "activity_units": "",
                        })

            if not next_variants:
                logger.warning(f"GrowMax round {round_num}: no new variants generated")
                break

            _, ro5_filtered, _ = apply_rule_of_5(next_variants)
            ro5_filtered = ro5_filtered or []

            # Pre-sort by QED descending so the dock slice gets the most drug-like compounds
            def _qed(c):
                mol = Chem.MolFromSmiles(c["smiles"])
                return QED.qed(mol) if mol else 0.0

            ro5_filtered.sort(key=_qed, reverse=True)
            current_pool = ro5_filtered
            logger.info(
                f"GrowMax: {len(next_variants)} round-{round_num + 1} variants, "
                f"{len(current_pool)} passed Ro5, sorted by QED"
            )

    except Exception as exc:
        logger.exception(f"Score-guided growth failed: {exc}")
        yield f"**Step 4 – Docking:** ❌ {exc}", None, None
        return
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not all_docked:
        yield "**Step 4 – Docking:** ❌ No compounds successfully docked", None, None
        return

    # Final display sort: affinity ascending (LE already used for parent selection)
    all_docked.sort(key=lambda x: x.get("vina_affinity", float("inf")))
    for i, c in enumerate(all_docked):
        c["binding_rank"] = i + 1

    rows = []
    for c in all_docked[:500]:
        rows.append({
            "Rank": c["binding_rank"],
            "Name": c["name"][:25],
            "Round": c.get("growth_round", 1),
            "Vina (kcal/mol)": f"{c['vina_affinity']:.2f}",
            "LE": f"{c.get('ligand_efficiency', 0):.3f}",
            "MW": c.get("mw", ""),
            "LogP": c.get("logp", ""),
        })

    n_rounds_done = max((c.get("growth_round", 1) for c in all_docked), default=1)
    status = (
        f"**Step 4 – Docking:** ✅ Score-guided growth complete — "
        f"{len(all_docked)} compounds across {n_rounds_done} rounds"
    )
    logger.info(f"=== Score-guided GrowMax COMPLETE: {len(all_docked)} compounds ===")
    yield status, all_docked, pd.DataFrame(rows)
