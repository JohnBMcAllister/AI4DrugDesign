"""Print the 'Explain This Compound' prompt without calling the AI.

Put this file in the project root (next to app.py), then run:

    uv run python test_prompt.py          # fake compound, instant, no network
    uv run python test_prompt.py 1IEP     # real pipeline run on a PDB target

The real run fetches compounds, docks a few with AutoDock Vina, and builds the
prompt from actual pipeline output. No OpenAI call is made, so it costs nothing.
It also saves the docked compounds to docked_<PDB>.json for the benchmark.
"""

import json
import sys

from dotenv import load_dotenv

from tabs.pipeline.detail import _get_pocket_contacts, build_explain_prompt


def fake_test():
    """Placeholder values, just to check formatting."""
    protein = {
        "pdb_id": "1IEP",
        "title": "Abl kinase domain in complex with STI-571",
        "classification": "TRANSFERASE",
        "organism": "Mus musculus",
    }
    compound = {
        "name": "IMATINIB",
        "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "activity_type": "IC50", "activity_value": 25.0, "activity_units": "nM",
        "vina_affinity": -11.42, "binding_rank": 1,
        "mw": 493.6, "logp": 4.59, "tpsa": 86.3, "hbd": 2, "hba": 7,
        "rot_bonds": 7, "ro5_violations": 0, "adme_score": 6,
    }
    contacts = {
        "pocket": ["ASP381", "GLU286", "MET318", "THR315"],
        "hbonds": ["MET318", "THR315", "GLU286"],
        "hydrophobic": ["LEU248", "VAL256"],
    }
    print(build_explain_prompt(compound, protein, contacts))


def real_test(pdb_id: str, num_dock: int = 5):
    """Run Steps 1-5 of the pipeline and build the prompt from real output."""
    from tabs.pipeline.compounds import fetch_compounds
    from tabs.pipeline.docking import rank_by_activity
    from tabs.pipeline.filters import apply_adme_filter, apply_rule_of_5
    from tabs.pipeline.protein import fetch_protein_info

    pdb_id = pdb_id.strip().upper()
    protein = fetch_protein_info(pdb_id)  # skips the Step 1 AI call

    status, compounds, _ = fetch_compounds(pdb_id)
    if not compounds:
        sys.exit(status)
    status, ro5, _ = apply_rule_of_5(compounds)
    if not ro5:
        sys.exit(status)
    status, docked, _ = rank_by_activity(ro5, protein, num_dock=num_dock)
    if not docked:
        sys.exit(status)
    status, final, _ = apply_adme_filter(docked)
    if not final:
        sys.exit(status)

    final.sort(key=lambda c: c.get("vina_affinity", float("inf")))

    out_path = f"docked_{pdb_id}.json"
    with open(out_path, "w") as f:
        json.dump({"protein": protein, "compounds": final}, f, indent=2, default=str)
    print(f"Saved {len(final)} docked compounds to {out_path}\n")

    best = final[0]
    contacts = _get_pocket_contacts(best, pdb_id)
    prompt = build_explain_prompt(best, protein, contacts)

    print("=" * 70)
    print(f"PROMPT FOR {best.get('name')} (best Vina score)")
    print("=" * 70)
    print(prompt)

    with open(f"prompt_{pdb_id}.txt", "w") as f:
        f.write(prompt)


if __name__ == "__main__":  # required: docking uses a process pool
    load_dotenv()
    if len(sys.argv) > 1:
        real_test(sys.argv[1])
    else:
        fake_test()