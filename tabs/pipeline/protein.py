"""Step 1 — Protein analysis: fetch info, 3D viewer, AI explanation."""

import html as html_module

import requests

from tabs.pipeline.helpers import (
    fetch_pdb_ligands,
    get_openai_client,
    get_uniprot_from_pdb,
    logger,
)

VIEWER_PLACEHOLDER = (
    '<div style="height:450px;display:flex;align-items:center;'
    "justify-content:center;background:#f5f5f5;border-radius:8px;\">"
    "<p style=\"color:#999;\">Enter a PDB ID and click "
    "'Analyze Protein' to view 3D structure</p></div>"
)


def build_3d_viewer_html(pdb_id: str) -> str:
    """Build an iframe with 3Dmol.js to display a protein structure with binding site highlighted."""
    inner = (
        "<!DOCTYPE html>"
        "<html><head>"
        "<script src='https://3Dmol.org/build/3Dmol-min.js'></script>"
        "<style>"
        "body{margin:0;padding:0;overflow:hidden;background:#fff}"
        "#v{width:100%;height:100%;position:absolute;top:0;left:0}"
        "#legend{position:absolute;bottom:8px;left:8px;z-index:10;"
        "background:rgba(255,255,255,0.88);padding:6px 10px;border-radius:6px;"
        "font-size:11px;font-family:sans-serif;line-height:1.9;pointer-events:none;"
        "box-shadow:0 1px 4px rgba(0,0,0,0.15)}"
        ".dot{display:inline-block;width:11px;height:11px;border-radius:50%;"
        "margin-right:5px;vertical-align:middle}"
        "</style>"
        "</head><body>"
        "<div id='v'></div>"
        "<div id='legend'>"
        "<b style='font-size:12px'>Binding Site</b><br>"
        "<div><span class='dot' style='background:linear-gradient(135deg,#6af,#fa6)'></span>Protein backbone</div>"
        "<div><span class='dot' style='background:#00bcd4'></span>Receptor pocket residues</div>"
        "<div><span class='dot' style='background:#ff9800'></span>Co-crystallized ligand</div>"
        "</div>"
        "<script>"
        "window.addEventListener('load',function(){"
        "var v=$3Dmol.createViewer('v',{backgroundColor:'white'});"
        "fetch('https://files.rcsb.org/download/" + pdb_id + ".pdb')"
        ".then(function(r){return r.ok?r.text():Promise.reject(r.status);})"
        ".then(function(d){"
        "v.addModel(d,'pdb');"
        "v.setStyle({},{cartoon:{color:'spectrum'}});"
        "v.setStyle("
        "{byres:true,within:{distance:5,sel:{hetflag:true}}},"
        "{cartoon:{color:'#00bcd4'},stick:{colorscheme:'Jmol',radius:0.15,opacity:1.0}}"
        ");"
        "v.setStyle("
        "{hetflag:true},"
        "{stick:{radius:0.28,colorscheme:'Jmol'},sphere:{scale:0.32,colorscheme:'Jmol'}}"
        ");"
        "v.zoomTo();v.render();"
        "document.getElementById('v').addEventListener('wheel',function(e){"
        "e.preventDefault();e.stopImmediatePropagation();"
        "v.zoom(e.deltaY<0?1.05:1/1.05);v.render();"
        "},{passive:false,capture:true});})"
        ".catch(function(){"
        "document.getElementById('v').innerHTML="
        "'<p style=padding:20px>Could not load structure.</p>';});"
        "});</script></body></html>"
    )
    escaped = html_module.escape(inner, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        'style="width:100%;height:450px;border:none;border-radius:8px;"></iframe>'
    )


def fetch_protein_info(pdb_id: str) -> dict:
    """Fetch protein metadata from RCSB PDB."""
    pdb_id = pdb_id.strip().upper()
    logger.info(f"Fetching protein info for PDB ID: {pdb_id}")
    base = "https://data.rcsb.org/rest/v1/core"

    r = requests.get(f"{base}/entry/{pdb_id}", timeout=15)
    r.raise_for_status()
    entry = r.json()
    logger.debug(f"Retrieved entry data for {pdb_id}")

    title = entry.get("struct", {}).get("title", "N/A")
    keywords = entry.get("struct_keywords", {}).get("pdbx_keywords", "N/A")

    # Organism — dhvani's fix: iterate per-entity via polymer_entity_ids for reliable lookup
    organisms = []
    try:
        polymer_entity_ids = entry.get(
            "rcsb_entry_container_identifiers", {}
        ).get("polymer_entity_ids", [])

        for entity_id in polymer_entity_ids:
            er = requests.get(
                f"{base}/polymer_entity/{pdb_id}/{entity_id}",
                timeout=15,
            )
            if not er.ok:
                continue

            entity = er.json()

            for s in entity.get("rcsb_entity_source_organism", []):
                org = s.get("ncbi_scientific_name") or s.get("scientific_name")
                if org and org not in organisms:
                    organisms.append(org)

            for s in entity.get("entity_src_gen", []):
                org = s.get("pdbx_gene_src_scientific_name")
                if org and org not in organisms:
                    organisms.append(org)

            for s in entity.get("entity_src_nat", []):
                org = s.get("pdbx_organism_scientific")
                if org and org not in organisms:
                    organisms.append(org)

        logger.debug(f"Found {len(organisms)} organism(s) for {pdb_id}")
    except Exception as e:
        logger.warning(f"Failed to fetch organisms for {pdb_id}: {e}")

    # Resolution
    resolution = (
        entry.get("refine", [{}])[0].get("ls_d_res_high")
        if entry.get("refine")
        else None
    )

    ligands = fetch_pdb_ligands(pdb_id, entry)

    logger.info(f"Found {len(ligands)} ligand(s) in {pdb_id}")

    # Binding sites
    binding_sites = []
    try:
        sr = requests.get(
            f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/binding_sites/{pdb_id.lower()}",
            timeout=15,
        )
        if sr.ok:
            for sid, sinfo in sr.json().get(pdb_id.lower(), {}).items():
                det = sinfo[0] if isinstance(sinfo, list) and sinfo else sinfo
                binding_sites.append({
                    "id": sid,
                    "description": det.get("site_description", sid),
                })
        logger.info(f"Found {len(binding_sites)} binding site(s) in {pdb_id}")
    except Exception as e:
        logger.warning(f"Failed to fetch binding sites for {pdb_id}: {e}")

    logger.info(f"Successfully fetched complete protein info for {pdb_id}")
    return {
        "pdb_id": pdb_id,
        "title": title,
        "classification": keywords,
        "organism": ", ".join(organisms) if organisms else "N/A",
        "resolution": f"{resolution} Å" if resolution else "N/A",
        "ligands": ligands,
        "binding_sites": binding_sites,
    }


def analyze_protein(pdb_id: str):
    """Step 1 handler: fetch info, build display text, 3D viewer."""
    logger.info(f"=== STEP 1: Analyzing protein {pdb_id} ===")
    pdb_id = (pdb_id or "").strip().upper()
    if not pdb_id:
        logger.warning("Empty PDB ID provided")
        return (
            "**Step 1 – Protein:** ❌ Please enter a PDB ID",
            "Please enter a PDB ID.",
            None,
            VIEWER_PLACEHOLDER,
        )

    try:
        info = fetch_protein_info(pdb_id)
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        logger.error(f"HTTP error fetching {pdb_id}: {code}")
        if code == 404:
            return (
                f"**Step 1 – Protein:** ❌ PDB ID '{pdb_id}' not found",
                f"PDB ID '{pdb_id}' not found.",
                None,
                VIEWER_PLACEHOLDER,
            )
        return (
            f"**Step 1 – Protein:** ❌ Error fetching data (HTTP {code})",
            f"Error fetching PDB data (HTTP {code}).",
            None,
            VIEWER_PLACEHOLDER,
        )
    except Exception as exc:
        logger.error(f"Error analyzing protein {pdb_id}: {exc}")
        return (
            f"**Step 1 – Protein:** ❌ Error: {exc}",
            f"Error: {exc}",
            None,
            VIEWER_PLACEHOLDER,
        )

    info["uniprot_id"] = get_uniprot_from_pdb(pdb_id)
    logger.info(f"UniProt ID for {pdb_id}: {info['uniprot_id']}")

    lig_str = (
        ", ".join(f"{lig['comp_id']} ({lig['name']})" for lig in info["ligands"])
        if info["ligands"]
        else "None identified"
    )
    site_str = (
        ", ".join(f"{s['id']}: {s['description']}" for s in info["binding_sites"])
        if info["binding_sites"]
        else "None available"
    )

    text = (
        f"{'=' * 55}\n"
        f"  PROTEIN: {info['title']}\n"
        f"{'=' * 55}\n"
        f"PDB ID         : {info['pdb_id']}\n"
        f"Classification : {info['classification']}\n"
        f"Organism       : {info['organism']}\n"
        f"Resolution     : {info['resolution']}\n"
        f"UniProt ID     : {info.get('uniprot_id') or 'N/A'}\n\n"
        f"LIGANDS: {lig_str}\n\n"
        f"BINDING SITES: {site_str}\n"
        f"{'=' * 55}"
    )

    # AI analysis
    client = get_openai_client()
    if client:
        try:
            logger.info(f"Requesting AI analysis for {pdb_id}")
            prompt = (
                "You are an expert structural biologist. Briefly analyze this "
                "protein for drug discovery.\n"
                f"PDB: {info['pdb_id']}, Title: {info['title']}, "
                f"Classification: {info['classification']}, "
                f"Organism: {info['organism']}, Ligands: {lig_str}, "
                f"Binding Sites: {site_str}\n\n"
                "Provide a concise analysis: druggable binding sites, key "
                "interactions, and design considerations."
            )
            resp = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
            )
            text += "\n\nAI ANALYSIS:\n" + resp.choices[0].message.content
            logger.info(f"AI analysis completed for {pdb_id}")
        except Exception as exc:
            logger.error(f"AI analysis failed for {pdb_id}: {exc}")
            text += f"\n\n[AI analysis failed: {exc}]"

    viewer = build_3d_viewer_html(pdb_id)
    logger.info(f"=== STEP 1 COMPLETE: {pdb_id} analyzed successfully ===")

    status = f"**Step 1 – Protein:** ✅ Analysis complete for {pdb_id}"
    return status, text, info, viewer


def ai_explain_protein(protein_state):
    """AI-generated deep dive on the protein target."""
    logger.info("=== Generating AI protein deep dive ===")
    if not protein_state:
        logger.warning("Protein deep dive requested with no protein data")
        return "No protein data available. Run Step 1 first."

    client = get_openai_client()
    if not client:
        logger.warning("Protein deep dive requested but OpenAI API key not set")
        return "[OpenAI API key not set]"

    p = protein_state
    logger.info(f"Generating protein deep dive for {p['pdb_id']}")
    lig_str = (
        ", ".join(f"{lig['comp_id']} ({lig['name']})" for lig in p.get("ligands", []))
        or "none"
    )

    prompt = (
        "You are an expert structural biologist and medicinal chemist.\n"
        f"Provide a detailed analysis of this protein target:\n\n"
        f"PDB ID: {p['pdb_id']}\n"
        f"Title: {p['title']}\n"
        f"Classification: {p['classification']}\n"
        f"Organism: {p['organism']}\n"
        f"Resolution: {p['resolution']}\n"
        f"Ligands: {lig_str}\n\n"
        "Cover:\n"
        "1. BINDING SITES — location and functional significance\n"
        "2. KEY RESIDUES — catalytic, allosteric, drug-interacting\n"
        "3. INTERACTION TYPES — H-bonds, hydrophobic, electrostatic, etc.\n"
        "4. DESIGN CONSIDERATIONS for drug/inhibitor design\n"
        "5. DRUG DEVELOPMENT INSIGHTS — druggability, selectivity, "
        "known inhibitor classes\n\n"
        "Be specific and concise."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": prompt}],
        )
        logger.info("AI protein deep dive completed successfully")
        return resp.choices[0].message.content
    except Exception as exc:
        logger.error(f"AI protein deep dive failed: {exc}")
        return f"AI analysis failed: {exc}"
