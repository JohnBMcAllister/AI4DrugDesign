"""
Integrated Drug Discovery Pipeline

Single-page workflow:
  1. Input Protein (PDB ID) → Analyze & 3D view
  2. Discover Compounds (ChEMBL)
  3. Filter: Rule of 5
  4. Rank: Binding Activity (experimental data)
  5. Filter: ADME scores
  6. Select compound → Detail view + AI explanation
"""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import gradio as gr

from tabs.compound_optimization import OPTIMIZATION_GOAL_OPTIONS, _optimize_compound
from tabs.pipeline.compounds import fetch_compounds
from tabs.pipeline.detail import (
    COMPOUND_3D_PLACEHOLDER,
    DOCKED_3D_PLACEHOLDER,
    _get_pocket_contacts,
    ai_explain_compound,
    on_select_compound,
    populate_compound_selector,
    show_docked_view,
)
from tabs.pipeline.docking import rank_by_activity
from tabs.pipeline.filters import apply_adme_filter, apply_rule_of_5
from tabs.pipeline.growmax import fetch_growmax_compounds, run_score_guided_growth
from tabs.pipeline.helpers import logger
from tabs.pipeline.protein import (
    VIEWER_PLACEHOLDER,
    ai_explain_protein,
    analyze_protein,
)


# ── Benchmark candidate export ────────────────────────────────────────
#
# Writing benchmark candidates is OFF by default, so an ordinary pipeline run
# can never silently replace the dataset an eval was measured against. Enable
# it deliberately for a regeneration run:
#
#   DUMP_BENCHMARK_CANDIDATES=1 python app.py            # writes new files only
#   DUMP_BENCHMARK_CANDIDATES=1 \
#   DUMP_BENCHMARK_OVERWRITE=1 python app.py             # replaces existing ones
#
# Every dump also writes a <PDB>_candidates.meta.json sidecar recording the
# SHA-256 of the data file and the docking box centre it was produced with, so
# a later change to find_binding_center is detectable instead of invisible.

BENCHMARK_DUMP_VERSION = "vina-contacts-v1"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


# Fields consumed by detail.build_compound_context. Keep this in step with it:
# anything missing here renders as "not available" in the benchmark prompt.
_PROMPT_COMPOUND_FIELDS = (
    "name",
    "smiles",
    "chembl_id",
    "activity_type",
    "activity_value",
    "activity_units",
    "mw",
    "logp",
    "hbd",
    "hba",
    "tpsa",
    "rot_bonds",
    "ro5_violations",
    "adme_score",
    "vina_affinity",
    "ligand_efficiency",
    "binding_rank",
    "growth_round",
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_center_for(pdb_id: str):
    """Record the docking box centre used, for provenance."""
    try:
        from tabs.pipeline.docking import fetch_pdb_file, find_binding_center
        return list(find_binding_center(fetch_pdb_file(pdb_id)))
    except Exception as exc:
        logger.warning(f"Could not record binding centre for {pdb_id}: {exc}")
        return None


def _dump_test_candidates(pdb_id, protein_state, final_state):
    """Save the top candidates from a completed pipeline run as benchmark cases.

    Exports every field the benchmark prompt reads, plus the Vina pose and the
    binding-site contacts derived from it. The pose is the one thing that cannot
    be recomputed later, so it is stored verbatim.
    """
    if not _env_flag("DUMP_BENCHMARK_CANDIDATES"):
        return

    if not final_state or not protein_state:
        logger.warning("Benchmark dump skipped: the run produced no final compounds.")
        return

    top_n = int(os.getenv("DUMP_BENCHMARK_TOP_N", "5"))
    os.makedirs("benchmarks", exist_ok=True)

    clean_pdb = (
        pdb_id.strip().upper() if pdb_id else protein_state.get("pdb_id", "TARGET")
    )
    file_path = f"benchmarks/{clean_pdb}_candidates.json"
    meta_path = f"benchmarks/{clean_pdb}_candidates.meta.json"

    if os.path.exists(file_path) and not _env_flag("DUMP_BENCHMARK_OVERWRITE"):
        logger.warning(
            f"Benchmark dump skipped: {file_path} already exists. "
            "Set DUMP_BENCHMARK_OVERWRITE=1 to replace it."
        )
        return

    protein_meta = {
        "title": protein_state.get("title", ""),
        "pdb_id": clean_pdb,
        "classification": protein_state.get("classification", ""),
        "organism": protein_state.get("organism", ""),
    }

    payload = []
    missing_fields: set[str] = set()

    for c in final_state[:top_n]:
        compound = {k: c[k] for k in _PROMPT_COMPOUND_FIELDS if k in c}

        # activity_units is present-but-None for some ChEMBL rows; an empty
        # string keeps "RBA = 0.005 None" out of the prompt.
        compound["activity_units"] = c.get("activity_units") or ""

        for field in _PROMPT_COMPOUND_FIELDS:
            if field in ("growth_round", "chembl_id"):
                continue
            if compound.get(field) is None:
                missing_fields.add(field)

        if c.get("docked_sdf"):
            compound["docked_sdf"] = c["docked_sdf"]

        sample = {"protein": protein_meta, "compound": compound}

        try:
            sample["contacts"] = _get_pocket_contacts(c, clean_pdb)
        except Exception as exc:
            logger.warning(
                f"Could not derive contacts for {c.get('name')}: {exc}"
            )
            sample["contacts"] = None

        payload.append(sample)

    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)

    without_contacts = sum(1 for s in payload if not s.get("contacts"))
    meta = {
        "dump_version": BENCHMARK_DUMP_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pdb_id": clean_pdb,
        "sample_count": len(payload),
        "binding_center": _binding_center_for(clean_pdb),
        "samples_without_contacts": without_contacts,
        "sha256": _sha256(file_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Dumped {len(payload)} benchmark cases to {file_path} "
        f"(sha256 {meta['sha256'][:12]}, centre {meta['binding_center']})"
    )
    if without_contacts:
        logger.warning(
            f"{without_contacts}/{len(payload)} candidates have no binding-site "
            "contacts — the benchmark prompt will say so for those."
        )
    if missing_fields:
        logger.warning(
            "Benchmark candidates are missing values for: "
            + ", ".join(sorted(missing_fields))
            + ". Those lines render as 'not available' in the prompt."
        )


# ── Orchestration helpers ─────────────────────────────────────────────


def _run_protein_and_step2(pdb_id, strategy, fragment_smiles):
    """Run protein analysis and step 2 (ChEMBL or GrowMax) in parallel."""
    logger.info(f"=== Starting parallel execution: Protein Analysis + Step 2 ({strategy}) ===")

    with ThreadPoolExecutor(max_workers=2) as executor:
        protein_future = executor.submit(analyze_protein, pdb_id)
        if strategy == "GrowMax (Fragment Growing)":
            step2_future = executor.submit(fetch_growmax_compounds, fragment_smiles)
        else:
            step2_future = executor.submit(fetch_compounds, pdb_id)

        protein_results = protein_future.result()
        step2_results = step2_future.result()

    logger.info("=== Parallel execution complete ===")

    # protein_results: (status, text, info, viewer)
    # step2_results: (status, compounds, table)
    return (*protein_results, *step2_results)


def _rank_compounds(ro5_state, protein_state, num_dock, strategy, growth_rounds):
    """Branch between standard docking and score-guided GrowMax growth."""
    if strategy == "GrowMax (Fragment Growing)":
        yield from run_score_guided_growth(
            ro5_state,
            protein_state,
            rounds=int(growth_rounds),
            dock_per_round=int(num_dock),
            top_k=10,
        )
        return
    yield rank_by_activity(ro5_state, protein_state, num_dock)


def _pipeline_initial_status(strategy):
    """Reset UI and set all step statuses to their initial running/waiting state."""
    step2_msg = (
        "**Step 2 – GrowMax:** 🔄 Growing fragment variants..."
        if strategy == "GrowMax (Fragment Growing)"
        else "**Step 2 – Compounds:** 🔄 Fetching from ChEMBL..."
    )
    return (
        "**Step 1 – Protein:** 🔄 Analyzing structure from RCSB PDB...",
        step2_msg,
        "**Step 3 – Rule of 5:** ⏳ Waiting...",
        "**Step 4 – Docking:** ⏳ Waiting...",
        "**Step 5 – ADME:** ⏳ Waiting...",
        "",                       # clear detail text
        None,                     # clear detail image
        COMPOUND_3D_PLACEHOLDER,  # reset 3D compound viewer
        DOCKED_3D_PLACEHOLDER,    # reset docked view
        "",                       # clear AI explanation
    )


# ── UI ─────────────────────────────────────────────────────────────────


def create_tab():
    with gr.Tab("Drug Discovery Pipeline"):
        # --- State ---
        protein_state = gr.State(None)
        all_compounds_state = gr.State(None)
        ro5_state = gr.State(None)
        ranked_state = gr.State(None)
        final_state = gr.State(None)
        selected_state = gr.State(None)

        # ────────────────────────────────────────────────────────
        # Main Input & Run Button
        # ────────────────────────────────────────────────────────
        gr.Markdown(
            "## Quick Start: Run Complete Pipeline\n"
            "Enter a PDB ID and click the button below to run the entire pipeline automatically."
        )

        with gr.Row():
            pdb_input = gr.Textbox(
                label="PDB ID",
                placeholder="e.g., 6LU7",
                lines=1,
                scale=2,
            )
            num_dock_input = gr.Number(
                label="Compounds to Dock",
                value=10,
                minimum=1,
                maximum=500,
                step=1,
                scale=1,
                info="ChEMBL: total docked. GrowMax: docked per round (×rounds = total).",
            )

        strategy_radio = gr.Radio(
            choices=["Standard (ChEMBL)", "GrowMax (Fragment Growing)"],
            value="Standard (ChEMBL)",
            label="Discovery Strategy",
            info=(
                "Standard fetches known bioactives from ChEMBL. "
                "GrowMax grows a seed fragment into novel candidates using AutoDock Vina scoring."
            ),
        )
        with gr.Column(visible=False) as fragment_col:
            with gr.Row():
                fragment_input = gr.Textbox(
                    label="Seed Fragment SMILES",
                    placeholder="e.g., c1ccccc1",
                    lines=1,
                    scale=3,
                )
                growth_rounds_input = gr.Number(
                    label="Growth Rounds",
                    value=3,
                    minimum=1,
                    maximum=4,
                    step=1,
                    scale=1,
                    info="Each round docks the current pool, then grows from the top 10 scorers. 3 recommended.",
                )
            gr.Examples(
                examples=[
                    ["c1ccccc1"],
                    ["c1ccncc1"],
                    ["c1cnccn1"],
                    ["c1cnc[nH]1"],
                    ["c1ccsc1"],
                    ["C1CCNCC1"],
                    ["c1ccc(O)cc1"],
                    ["CC(=O)N"],
                ],
                inputs=fragment_input,
                label="Example seed fragments",
            )

        run_pipeline_btn = gr.Button(
            "🚀 Run Complete Pipeline",
            variant="primary",
            size="lg",
        )

        gr.Examples(
            examples=[["6LU7"], ["1IEP"], ["2HYY"], ["3ERT"]],
            inputs=pdb_input,
        )

        gr.Markdown("---\n## Pipeline Progress")

        # Step 1: Protein Analysis
        protein_status = gr.Markdown(value="**Step 1 – Protein:** Not started")
        with gr.Accordion("Step 1: Protein Details", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    protein_text = gr.Textbox(
                        label="Protein Information & AI Analysis",
                        interactive=False,
                        lines=20,
                    )
                with gr.Column(scale=1):
                    viewer_html = gr.HTML(value=VIEWER_PLACEHOLDER)
            explain_protein_btn = gr.Button(
                "AI: Deep Dive on Protein", variant="secondary"
            )
            protein_deep_dive = gr.Textbox(
                label="AI Protein Deep Dive",
                interactive=False,
                lines=12,
            )

        # Step 2: Compound Discovery
        compounds_status = gr.Markdown(value="**Step 2 – Compounds:** Not started")
        with gr.Accordion("Step 2: Discovered Compounds", open=False):
            compounds_table = gr.Dataframe(
                label="Compounds from ChEMBL",
                interactive=False,
                wrap=True,
                max_height=300,
            )

        # Step 3: Rule of 5
        ro5_status = gr.Markdown(value="**Step 3 – Rule of 5:** Not started")
        with gr.Accordion("Step 3: Rule of 5 Results", open=False):
            ro5_table = gr.Dataframe(
                label="Rule of 5 Filtered Compounds",
                interactive=False, wrap=True, max_height=300,
            )

        # Step 4: AutoDock Vina Docking
        rank_status = gr.Markdown(value="**Step 4 – Docking:** Not started")
        with gr.Accordion("Step 4: Docking Results", open=False):
            rank_table = gr.Dataframe(
                label="Docking Ranked Compounds",
                interactive=False, wrap=True, max_height=300,
            )

        # Step 5: ADME Filter
        adme_status = gr.Markdown(value="**Step 5 – ADME:** Not started")
        with gr.Accordion("Step 5: ADME Results", open=False):
            adme_table = gr.Dataframe(
                label="ADME Filtered Compounds",
                interactive=False, wrap=True, max_height=300,
            )

        # ────────────────────────────────────────────────────────
        # Results — Compound Detail
        # ────────────────────────────────────────────────────────
        gr.Markdown(
            "---\n## Final Results: Most Relevant Compounds"
        )

        compound_selector = gr.Dataframe(
            label="Click a row to view details",
            interactive=False,
            wrap=True,
            max_height=250,
        )

        with gr.Row():
            with gr.Column(scale=1):
                detail_text = gr.Textbox(
                    label="Compound Properties",
                    interactive=False,
                    lines=22,
                )
            with gr.Column(scale=1):
                detail_image = gr.Image(
                    label="2D Structure",
                    type="pil",
                    height=400,
                )

        # ── Compound Optimization ──────────────────────────────
        gr.Markdown("### Compound Optimization")
        gr.Markdown(
            "SMILES is auto-filled when you select a compound above, "
            "or enter any SMILES manually."
        )
        with gr.Row():
            with gr.Column(scale=1):
                optim_smiles_input = gr.Textbox(
                    label="Compound SMILES",
                    placeholder="Select a compound above or enter SMILES manually",
                    lines=3,
                )
                optim_goals_input = gr.CheckboxGroup(
                    choices=OPTIMIZATION_GOAL_OPTIONS,
                    label="Optimization Goals",
                    info="Select one or more goals.",
                )
                optimize_btn = gr.Button("Optimize Compound", variant="secondary", size="lg")
            with gr.Column(scale=1):
                optim_output = gr.Textbox(
                    label="Optimization Suggestions",
                    interactive=False,
                    lines=18,
                    placeholder=(
                        "a. Initial Properties of the Compound\n"
                        "b. Suggested Structural Modifications\n"
                        "c. Optimization Goal Mapping\n"
                        "d. Drug-Likeness Considerations"
                    ),
                )

        # 3D Structure Viewer
        gr.Markdown("### 3D Structure")
        compound_3d_viewer = gr.HTML(
            value=COMPOUND_3D_PLACEHOLDER,
            label="3D Structure",
        )

        # Receptor interaction viewer
        gr.Markdown("### Receptor Interaction Viewer")
        gr.Markdown(
            "Click below to render the compound docked inside the protein binding pocket. "
            "Binding-pocket residues are highlighted as green sticks. "
            "H-bond contacts appear as **blue dashed lines** and hydrophobic contacts as "
            "**orange dashed lines**. Use the **Toggle Surface** button inside the viewer "
            "to display the pocket surface."
        )
        show_docked_btn = gr.Button(
            "Show Receptor Interaction View",
            variant="primary",
            size="lg",
        )
        docked_viewer_html = gr.HTML(value=DOCKED_3D_PLACEHOLDER)

        explain_compound_btn = gr.Button(
            "AI: Explain This Compound", variant="secondary", size="lg"
        )
        compound_explanation = gr.Textbox(
            label="AI Compound Analysis",
            interactive=False,
            lines=15,
        )

        # ────────────────────────────────────────────────────────
        # Wire events
        # ────────────────────────────────────────────────────────

        # Main "Run Complete Pipeline" — chained steps for progressive updates
        _progress_args = dict(show_progress="hidden")

        run_pipeline_btn.click(
            _pipeline_initial_status,
            inputs=[strategy_radio],
            outputs=[protein_status, compounds_status,
                     ro5_status, rank_status, adme_status,
                     detail_text, detail_image, compound_3d_viewer,
                     docked_viewer_html, compound_explanation],
            **_progress_args,
        ).then(
            _run_protein_and_step2,
            inputs=[pdb_input, strategy_radio, fragment_input],
            outputs=[protein_status, protein_text, protein_state, viewer_html,
                     compounds_status, all_compounds_state, compounds_table],
            **_progress_args,
        ).then(
            lambda: "**Step 3 – Rule of 5:** 🔄 Filtering...",
            outputs=[ro5_status],
            **_progress_args,
        ).then(
            apply_rule_of_5,
            inputs=[all_compounds_state],
            outputs=[ro5_status, ro5_state, ro5_table],
            **_progress_args,
        ).then(
            lambda s: (
                "**Step 4 – Docking:** 🔄 Running score-guided growth..."
                if s == "GrowMax (Fragment Growing)"
                else "**Step 4 – Docking:** 🔄 Running AutoDock Vina..."
            ),
            inputs=[strategy_radio],
            outputs=[rank_status],
            **_progress_args,
        ).then(
            _rank_compounds,
            inputs=[ro5_state, protein_state, num_dock_input, strategy_radio, growth_rounds_input],
            outputs=[rank_status, ranked_state, rank_table],
            **_progress_args,
        ).then(
            lambda: "**Step 5 – ADME:** 🔄 Filtering...",
            outputs=[adme_status],
            **_progress_args,
        ).then(
            apply_adme_filter,
            inputs=[ranked_state],
            outputs=[adme_status, final_state, adme_table],
            **_progress_args,
        ).then(
            populate_compound_selector,
            inputs=[final_state],
            outputs=[compound_selector],
            **_progress_args,
        ).then(
            _dump_test_candidates,
            inputs=[pdb_input, protein_state, final_state],
            **_progress_args,
        )

        # Compound selection — click a row in either table, then auto-fill SMILES
        compound_selector.select(
            on_select_compound,
            inputs=[final_state],
            outputs=[detail_text, detail_image, compound_3d_viewer, selected_state],
        ).then(
            lambda c: c.get("smiles", "") if c else "",
            inputs=[selected_state],
            outputs=[optim_smiles_input],
        )
        adme_table.select(
            on_select_compound,
            inputs=[final_state],
            outputs=[detail_text, detail_image, compound_3d_viewer, selected_state],
        ).then(
            lambda c: c.get("smiles", "") if c else "",
            inputs=[selected_state],
            outputs=[optim_smiles_input],
        )

        # Show compound docked in binding site
        show_docked_btn.click(
            show_docked_view,
            inputs=[selected_state, protein_state],
            outputs=[docked_viewer_html],
        )

        # Compound optimization
        optimize_btn.click(
            lambda smiles, goals: _optimize_compound(smiles, goals)[0],
            inputs=[optim_smiles_input, optim_goals_input],
            outputs=[optim_output],
        )

        # AI explanations
        explain_protein_btn.click(
            ai_explain_protein,
            inputs=[protein_state],
            outputs=[protein_deep_dive],
        )
        explain_compound_btn.click(
            ai_explain_compound,
            inputs=[selected_state, protein_state],
            outputs=[compound_explanation],
        )

        strategy_radio.change(
            lambda s: (
                gr.update(visible=s == "GrowMax (Fragment Growing)"),
                gr.update(value=50 if s == "GrowMax (Fragment Growing)" else 10),
            ),
            inputs=[strategy_radio],
            outputs=[fragment_col, num_dock_input],
        )