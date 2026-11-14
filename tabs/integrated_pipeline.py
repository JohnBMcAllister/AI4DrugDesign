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

from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from tabs.compound_optimization import OPTIMIZATION_GOAL_OPTIONS, _optimize_compound
from tabs.pipeline.compounds import fetch_compounds
from tabs.pipeline.detail import (
    COMPOUND_3D_PLACEHOLDER,
    DOCKED_3D_PLACEHOLDER,
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