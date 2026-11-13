"""Step 6 — Compound detail view, selection handler, AI explanation."""

import html as html_module
import io
import json

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

from tabs.pipeline.helpers import get_openai_client, logger

_SKIP_RESIDUES_POCKET = frozenset({
    "HOH", "SO4", "PO4", "GOL", "EDO", "ACT", "FMT", "IOD",
    "CL", "MG", "ZN", "CA", "NA", "PEG", "DMS", "MPD", "IMD",
})
_HYDROPHOBIC_RESIDUES = frozenset({
    "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR", "MET", "PRO", "CYS",
})

def _parse_pdb_atoms(pdb_text: str) -> list:
    """Return list of atom dicts from ATOM/HETATM records in a PDB text block."""
    atoms = []
    for line in pdb_text.splitlines():
        rec = line[:6]
        if rec not in ("ATOM  ", "HETATM"):
            continue
        try:
            res_name = line[17:20].strip()
            chain = line[21] if len(line) > 21 else " "
            res_seq = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            element = line[76:78].strip().upper() if len(line) > 77 else ""
            if not element:
                for ch in line[12:16].strip():
                    if ch.isalpha():
                        element = ch.upper()
                        break
            atoms.append({
                "record": rec.strip(),
                "res_name": res_name,
                "chain": chain,
                "res_seq": res_seq,
                "x": x, "y": y, "z": z,
                "element": element,
            })
        except (ValueError, IndexError):
            continue
    return atoms


def _analyze_binding_pocket(mol, pdb_text: str, binding_center, cutoff: float = 6.0):
    """Find pocket residues and candidate H-bond / hydrophobic interaction pairs.

    Returns:
        pocket_residues : list[dict]  — residues with any atom within *cutoff* Å of ligand
        hbond_pairs     : list[dict]  — closest N/O … N/O contacts ≤ 3.5 Å (one per residue)
        hydro_pairs     : list[dict]  — closest C … C hydrophobic contacts ≤ 4.5 Å
    """
    if not pdb_text or mol is None:
        return [], [], []
    try:
        conf = mol.GetConformer()
        lig_pos = conf.GetPositions()
        lig_elem = [mol.GetAtomWithIdx(i).GetSymbol().upper() for i in range(mol.GetNumAtoms())]
    except Exception:
        return [], [], []

    protein_atoms = [a for a in _parse_pdb_atoms(pdb_text) if a["record"] == "ATOM"]
    pocket_set: set = set()
    hbond_best: dict = {}
    hydro_best: dict = {}

    for pa in protein_atoms:
        if pa["res_name"] in _SKIP_RESIDUES_POCKET:
            continue
        px, py, pz = pa["x"], pa["y"], pa["z"]
        pe = pa["element"]
        for j, (lx, ly, lz) in enumerate(lig_pos):
            dist = ((px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2) ** 0.5
            if dist < cutoff:
                pocket_set.add((pa["chain"], pa["res_seq"], pa["res_name"]))
            le = lig_elem[j]
            key = (pa["chain"], pa["res_seq"])
            if dist < 3.5 and pe in ("N", "O") and le in ("N", "O"):
                if key not in hbond_best or dist < hbond_best[key][0]:
                    hbond_best[key] = (dist, {
                        "lig": [round(lx, 3), round(ly, 3), round(lz, 3)],
                        "prot": [round(px, 3), round(py, 3), round(pz, 3)],
                        "res_name": pa["res_name"], "res_seq": pa["res_seq"],
                    })
            if (dist < 4.5 and pe == "C" and le == "C"
                    and pa["res_name"] in _HYDROPHOBIC_RESIDUES):
                if key not in hydro_best or dist < hydro_best[key][0]:
                    hydro_best[key] = (dist, {
                        "lig": [round(lx, 3), round(ly, 3), round(lz, 3)],
                        "prot": [round(px, 3), round(py, 3), round(pz, 3)],
                        "res_name": pa["res_name"], "res_seq": pa["res_seq"],
                    })

    pocket_list = [{"chain": r[0], "res_seq": r[1], "res_name": r[2]} for r in pocket_set]
    hbond_list = [v[1] for v in sorted(hbond_best.values(), key=lambda x: x[0])][:10]
    hydro_list = [v[1] for v in sorted(hydro_best.values(), key=lambda x: x[0])][:15]
    return pocket_list, hbond_list, hydro_list


def build_receptor_interaction_viewer_html(
    smiles: str,
    pdb_id: str,
    compound_name: str = "",
    binding_center=None,
    pdb_text: str = "",
    docked_sdf: str = "",
) -> str:
    """Build a Schrödinger-style pocket viewer: full protein cartoon, binding-pocket
    residues as thin green sticks, compound as thicker cyan sticks, H-bond/hydrophobic
    dashes, residue labels.

    If docked_sdf is provided (from an actual AutoDock Vina run) it is used directly
    as the compound geometry instead of re-generating a conformer from SMILES.
    """
    if docked_sdf:
        # Use the real Vina-docked pose — no conformer generation or translation needed
        mol = Chem.MolFromMolBlock(docked_sdf, removeHs=False, sanitize=True)
        if mol is None:
            logger.warning("Could not parse docked SDF; falling back to conformer generation")
            docked_sdf = ""

    if not docked_sdf:
        # Fallback: generate a conformer from SMILES and translate to binding site
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (
                '<div style="height:570px;display:flex;align-items:center;'
                'justify-content:center;background:#0d1117;border-radius:8px;">'
                '<p style="color:#f87171;">Could not parse SMILES for 3D generation</p></div>'
            )

        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        try:
            ff = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            if ff != 0:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass

        cx, cy, cz = (binding_center if binding_center else (0.0, 0.0, 0.0))
        try:
            conf = mol.GetConformer()
            centroid = conf.GetPositions().mean(axis=0)
            dx, dy, dz = cx - centroid[0], cy - centroid[1], cz - centroid[2]
            for i in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, (p.x + dx, p.y + dy, p.z + dz))
        except Exception as exc:
            logger.warning(f"Could not translate compound to binding site: {exc}")

    pocket_residues, hbond_pairs, hydro_pairs = _analyze_binding_pocket(
        mol, pdb_text, binding_center
    )
    pocket_resi = sorted({r["res_seq"] for r in pocket_residues})

    # Strip explicit H atoms for a clean stick display
    mol_display = Chem.RemoveAllHs(mol)
    sdf_block = Chem.MolToMolBlock(mol_display)

    sdf_json = json.dumps(sdf_block)
    pocket_json = json.dumps(pocket_resi)
    hbonds_json = json.dumps(hbond_pairs)
    hydros_json = json.dumps(hydro_pairs)

    pdb_id_safe = pdb_id.strip().upper()
    title_label = (
        f"Receptor Interaction View: {html_module.escape(compound_name)}"
        if compound_name else "Receptor Interaction View"
    )
    title_html = (
        f'<div style="padding:8px 14px;font-size:13px;border-bottom:1px solid #30363d;'
        f'background:#161b22;display:flex;justify-content:space-between;align-items:center;">'
        f'<b style="color:#79c0ff;">{title_label}</b>'
        f'<span style="color:#8b949e;font-size:11px;">'
        f'Pocket: <span style="color:#56d364;">green</span>'
        f' &nbsp;|&nbsp; Compound: <span style="color:#00bcd4;">cyan</span>'
        f' &nbsp;|&nbsp; H-bond: <span style="color:#4d94ff;">&#9473;&#9473;</span>'
        f' &nbsp;|&nbsp; Hydrophobic: <span style="color:#e3b341;">&#9473;&#9473;</span>'
        f'</span></div>'
    )

    # Rendering: full protein cartoon + pocket residues as green sticks + compound as cyan sticks only
    render_js = (
        "  document.getElementById('loading').style.display='none';"
        "  viewer.addModel(pdbData,'pdb');"
        "  viewer.setStyle({model:0},{cartoon:{color:'spectrum',opacity:0.4}});"
        "  if(pocketResi.length>0){"
        "    viewer.setStyle({model:0,resi:pocketResi},{"
        "      cartoon:{color:'spectrum',opacity:0.7},"
        "      stick:{radius:0.09,colorscheme:'greenCarbon',opacity:1.0}"
        "    });"
        "  }"
        "  viewer.addModel(sdf,'sdf');"
        "  viewer.setStyle({model:1},{"
        "    stick:{radius:0.22,colorscheme:'cyanCarbon'}"
        "  });"
        "  for(var i=0;i<hbonds.length;i++){drawDash(hbonds[i].lig,hbonds[i].prot,'#4d94ff',7);}"
        "  for(var i=0;i<hydros.length;i++){drawDash(hydros[i].lig,hydros[i].prot,'#ff8c00',5);}"
        "  for(var i=0;i<Math.min(hbonds.length,8);i++){"
        "    var hb=hbonds[i];"
        "    viewer.addLabel(hb.res_name+hb.res_seq,{"
        "      position:{x:hb.prot[0],y:hb.prot[1],z:hb.prot[2]},"
        "      fontSize:11,fontColor:'#79c0ff',"
        "      backgroundOpacity:0.60,backgroundColor:'#0d1117'"
        "    });"
        "  }"
        "  viewer.zoomTo({model:1});"
        "  viewer.render();"
        "  document.getElementById('hb-n').textContent=hbonds.length;"
        "  document.getElementById('hy-n').textContent=hydros.length;"
        "  document.getElementById('pk-n').textContent=pocketResi.length;"
        "  document.getElementById('viewer').addEventListener('wheel',function(e){"
        "    e.preventDefault();e.stopImmediatePropagation();"
        "    viewer.zoom(e.deltaY<0?1.05:1/1.05);viewer.render();"
        "  },{passive:false,capture:true});"
    )

    if pdb_text:
        scene_js = "var pdbData=" + json.dumps(pdb_text) + ";" + render_js
    else:
        scene_js = (
            f"fetch('https://files.rcsb.org/download/{pdb_id_safe}.pdb')"
            ".then(function(r){return r.ok?r.text():Promise.reject(r.status);})"
            ".then(function(pdbData){" + render_js + "})"
            ".catch(function(){"
            "  document.getElementById('loading').textContent='Failed to load protein.';"
            "});"
        )

    inner = (
        "<!DOCTYPE html><html><head>"
        "<script src='https://3Dmol.org/build/3Dmol-min.js'></script>"
        "<style>"
        "body{margin:0;padding:0;overflow:hidden;background:#0d1117;font-family:monospace}"
        "#viewer{width:100%;height:100%;position:absolute;top:0;left:0}"
        "#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);"
        "color:#8b949e;font-size:13px;z-index:10;pointer-events:none;"
        "background:rgba(13,17,23,0.88);padding:10px 18px;border-radius:6px}"
        "#panel{position:absolute;top:8px;right:8px;z-index:20;"
        "background:rgba(13,17,23,0.82);border:1px solid #30363d;"
        "border-radius:8px;padding:10px 14px;color:#c9d1d9;font-size:11px;min-width:155px}"
        "#panel h4{margin:0 0 7px;color:#79c0ff;font-size:12px}"
        ".sr{display:flex;justify-content:space-between;gap:12px;margin:3px 0}"
        ".sl{color:#8b949e}.hb{color:#79c0ff;font-weight:bold}"
        ".hy{color:#e3b341;font-weight:bold}.pk{color:#56d364;font-weight:bold}"
        "#leg{margin-top:8px;border-top:1px solid #30363d;padding-top:7px}"
        ".li{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:10px}"
        ".dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex-shrink:0}"
        "#ctrl{position:absolute;bottom:8px;left:8px;z-index:20;"
        "background:rgba(13,17,23,0.75);border-radius:6px;padding:5px 8px}"
        ".btn{background:#21262d;border:1px solid #30363d;color:#c9d1d9;"
        "padding:4px 10px;border-radius:5px;cursor:pointer;margin:2px;font-size:10px;"
        "font-family:monospace}"
        ".btn:hover{background:#30363d;color:#fff}"
        "</style>"
        "</head><body>"
        "<div id='viewer'></div>"
        "<div id='loading'>Loading structure&hellip;</div>"
        "<div id='panel'>"
        "<h4>Interactions</h4>"
        "<div class='sr'><span class='sl'>H-bonds</span><span class='hb' id='hb-n'>—</span></div>"
        "<div class='sr'><span class='sl'>Hydrophobic</span><span class='hy' id='hy-n'>—</span></div>"
        "<div class='sr'><span class='sl'>Pocket residues</span><span class='pk' id='pk-n'>—</span></div>"
        "<div id='leg'>"
        "<div class='li'><span class='dot' style='background:#4d94ff'></span>H-bond contact</div>"
        "<div class='li'><span class='dot' style='background:#ff8c00'></span>Hydrophobic contact</div>"
        "<div class='li'><span class='dot' style='background:#56d364'></span>Pocket residue (green C)</div>"
        "<div class='li'><span class='dot' style='background:#00bcd4'></span>Compound (cyan C)</div>"
        "</div></div>"
        "<div id='ctrl'>"
        "<button class='btn' onclick='toggleSurface()'>Toggle Surface</button>"
        "<button class='btn' onclick='if(viewer){viewer.zoomTo({model:1});viewer.render();}'>Reset View</button>"
        "</div>"
        "<script>"
        "var sdf=" + sdf_json + ";"
        "var pocketResi=" + pocket_json + ";"
        "var hbonds=" + hbonds_json + ";"
        "var hydros=" + hydros_json + ";"
        f"var cx={cx},cy={cy},cz={cz};"
        "var viewer=null,surfId=null;"
        "function drawDash(s,e,col,n){"
        "  if(!viewer)return;"
        "  n=n||7;"
        "  var dx=(e[0]-s[0])/(n*2),dy=(e[1]-s[1])/(n*2),dz=(e[2]-s[2])/(n*2);"
        "  for(var i=0;i<n;i++){"
        "    var sx=s[0]+dx*i*2,sy=s[1]+dy*i*2,sz=s[2]+dz*i*2;"
        "    viewer.addCylinder({start:{x:sx,y:sy,z:sz},end:{x:sx+dx,y:sy+dy,z:sz+dz},"
        "      radius:0.06,color:col,fromCap:1,toCap:1});"
        "  }"
        "}"
        "function toggleSurface(){"
        "  if(!viewer)return;"
        "  if(surfId!==null){viewer.removeSurface(surfId);surfId=null;}"
        "  else{"
        "    var sel=pocketResi.length>0?{model:0,resi:pocketResi}:{model:0};"
        "    surfId=viewer.addSurface($3Dmol.SurfaceType.VDW,"
        "      {opacity:0.18,color:'#1a3a2a'},sel);"
        "  }"
        "  viewer.render();"
        "}"
        "window.addEventListener('load',function(){"
        "  viewer=$3Dmol.createViewer('viewer',{backgroundColor:'#0d1117'});"
        + scene_js +
        "});"
        "</script>"
        "</body></html>"
    )

    escaped = html_module.escape(inner, quote=True)
    return (
        '<div style="border:1px solid #30363d;border-radius:8px;overflow:hidden;background:#161b22;">'
        + title_html
        + f'<iframe srcdoc="{escaped}" '
        'style="width:100%;height:550px;border:none;display:block;"></iframe>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Pharmacophore colour palette (RGB 0–1 tuples for rdMolDraw2D)
# ---------------------------------------------------------------------------
_DONOR_COLOR    = (0.25, 0.50, 1.00)   # blue  – H-bond donors  (N/O with H)
_ACCEPTOR_COLOR = (1.00, 0.30, 0.30)   # red   – H-bond acceptors (N/O lone-pair only)
_AROMATIC_COLOR = (0.90, 0.70, 0.10)   # gold  – aromatic atoms

# SMARTS patterns (matched on heavy-atom mol, no explicit Hs needed)
_SMARTS_DONOR    = Chem.MolFromSmarts('[N,O;!H0]')
_SMARTS_ACCEPTOR = Chem.MolFromSmarts('[N,O;H0]')
_SMARTS_AROMATIC = Chem.MolFromSmarts('[a]')


def _pharmacophore_highlights(mol):
    """Return (highlight_atoms, highlight_atom_colors) for rdMolDraw2D.

    Priority: donor > acceptor > aromatic.
    Highlights the binding-relevant atoms — those that interact directly
    with receptor residues via H-bonds or π-stacking.
    """
    seen = {}

    def _add(smarts, color):
        if smarts is None:
            return
        for match in mol.GetSubstructMatches(smarts):
            idx = match[0]
            if idx not in seen:
                seen[idx] = color

    _add(_SMARTS_DONOR,    _DONOR_COLOR)
    _add(_SMARTS_ACCEPTOR, _ACCEPTOR_COLOR)
    _add(_SMARTS_AROMATIC, _AROMATIC_COLOR)

    atoms  = list(seen.keys())
    colors = {idx: c for idx, c in seen.items()}
    return atoms, colors


def draw_2d_with_pharmacophore(mol, size=(400, 400)):
    """Render a 2D molecule image with pharmacophore-feature atom highlights.

    Returns a PIL Image. Falls back to plain MolToImage on any drawing error.
    """
    AllChem.Compute2DCoords(mol)
    highlight_atoms, highlight_colors = _pharmacophore_highlights(mol)

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.addStereoAnnotation = True
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
        )
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception as exc:
        logger.warning(f"Pharmacophore 2D draw failed ({exc}), falling back to plain image")
        return Draw.MolToImage(mol, size=size)

COMPOUND_3D_PLACEHOLDER = (
    '<div style="height:400px;display:flex;align-items:center;'
    'justify-content:center;background:#f5f5f5;border-radius:8px;">'
    '<p style="color:#999;">Select a compound to view its 3D structure</p></div>'
)

DOCKED_3D_PLACEHOLDER = (
    '<div style="height:520px;display:flex;align-items:center;'
    'justify-content:center;background:#f0f4ff;border-radius:8px;border:1px solid #c8d8f0;">'
    '<p style="color:#666;text-align:center;padding:20px;">'
    'Select a compound, then click<br><b>Show Compound in Binding Site</b>'
    '<br><br><span style="font-size:11px;color:#999;">'
    'Renders the protein with the compound positioned at the binding pocket</span>'
    '</p></div>'
)


def build_3d_compound_viewer_html(smiles: str, compound_name: str = "") -> str:
    """Build an iframe with 3Dmol.js to display a compound's 3D structure from SMILES.

    Converts SMILES → SDF via RDKit (3D coords + MMFF minimisation).  Atoms are
    coloured by pharmacophore role so binding-relevant features stand out:
      blue = H-bond donor   red = H-bond acceptor   gold = aromatic   grey = other (Jmol)
    """
    mol_orig = Chem.MolFromSmiles(smiles)
    if mol_orig is None:
        return (
            '<div style="height:400px;display:flex;align-items:center;'
            'justify-content:center;background:#f5f5f5;border-radius:8px;">'
            '<p style="color:#c00;">Could not parse SMILES for 3D generation</p></div>'
        )

    # Compute pharmacophore indices on the heavy-atom mol (indices stay valid
    # after AddHs because RDKit appends H atoms at the end)
    _, pharma_colors = _pharmacophore_highlights(mol_orig)

    # Group indices by colour for JS
    donors    = [i for i, c in pharma_colors.items() if c == _DONOR_COLOR]
    acceptors = [i for i, c in pharma_colors.items() if c == _ACCEPTOR_COLOR]
    aromatics = [i for i, c in pharma_colors.items() if c == _AROMATIC_COLOR]

    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol_orig)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)

    try:
        ff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        if ff_result != 0:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    sdf_block = Chem.MolToMolBlock(mol)
    sdf_json = json.dumps(sdf_block)

    # Pharmacophore index lists as JSON for JS
    donors_js    = json.dumps(donors)
    acceptors_js = json.dumps(acceptors)
    aromatics_js = json.dumps(aromatics)

    title_html = (
        f'<div style="padding:6px 10px;font-size:12px;color:#555;'
        f'border-bottom:1px solid #ddd;background:#fafafa;display:flex;'
        f'align-items:center;gap:12px;">'
        f'<b>3D Structure</b>'
        + (f': {html_module.escape(compound_name)}' if compound_name else '')
        + '<span style="margin-left:auto;font-size:11px;color:#777;">'
        '<span style="color:#4488ff">&#9679;</span> H-bond donor &nbsp;'
        '<span style="color:#ff4444">&#9679;</span> H-bond acceptor &nbsp;'
        '<span style="color:#cc9900">&#9679;</span> Aromatic'
        '</span>'
        '</div>'
    )

    inner = (
        "<!DOCTYPE html>"
        "<html><head>"
        "<script src='https://3Dmol.org/build/3Dmol-min.js'></script>"
        "<style>"
        "body{margin:0;padding:0;overflow:hidden;background:#fff}"
        "#viewer{width:100%;height:100%;position:absolute;top:0;left:0}"
        "</style>"
        "</head><body>"
        "<div id='viewer'></div>"
        "<script>"
        "window.addEventListener('load', function() {"
        "  var sdf = " + sdf_json + ";"
        "  var donors    = " + donors_js + ";"
        "  var acceptors = " + acceptors_js + ";"
        "  var aromatics = " + aromatics_js + ";"
        "  var viewer = $3Dmol.createViewer('viewer', {backgroundColor: 'white'});"
        "  viewer.addModel(sdf, 'sdf');"
        "  viewer.setStyle({}, {stick: {radius: 0.13, colorscheme: 'Jmol'}, sphere: {scale: 0.22, colorscheme: 'Jmol'}});"
        "  if (donors.length)    viewer.setStyle({index: donors},     {stick: {radius: 0.18, color: '#4488ff'}, sphere: {scale: 0.38, color: '#4488ff'}});"
        "  if (acceptors.length) viewer.setStyle({index: acceptors},  {stick: {radius: 0.18, color: '#ff4444'}, sphere: {scale: 0.38, color: '#ff4444'}});"
        "  if (aromatics.length) viewer.setStyle({index: aromatics},  {stick: {radius: 0.14, color: '#cc9900'}, sphere: {scale: 0.30, color: '#cc9900'}});"
        "  viewer.zoomTo();"
        "  viewer.render();"
        "  document.getElementById('viewer').addEventListener('wheel',function(e){"
        "    e.preventDefault();e.stopImmediatePropagation();"
        "    viewer.zoom(e.deltaY<0?1.05:1/1.05);viewer.render();"
        "  },{passive:false,capture:true});"
        "});"
        "</script>"
        "</body></html>"
    )

    escaped = html_module.escape(inner, quote=True)
    return (
        f'<div style="border:1px solid #ddd;border-radius:8px;overflow:hidden;">'
        + title_html
        + f'<iframe srcdoc="{escaped}" '
        'style="width:100%;height:390px;border:none;display:block;"></iframe>'
        '</div>'
    )


def on_select_compound(evt: gr.SelectData, final_state):
    """Handle click on a row in the final results table."""
    if not final_state:
        logger.warning("Compound selection attempted with no compounds available")
        return "No compound selected.", None, COMPOUND_3D_PLACEHOLDER, None
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx >= len(final_state):
        logger.warning(f"Invalid compound index selected: {idx}")
        return "Invalid selection.", None, COMPOUND_3D_PLACEHOLDER, None

    compound = final_state[idx]
    smiles = compound["smiles"]
    name = compound.get("name", "")
    logger.info(f"Selected compound: {name} (ChEMBL: {compound.get('chembl_id')})")

    detail = (
        f"{'=' * 50}\n"
        f"  COMPOUND: {compound['name']}\n"
        f"{'=' * 50}\n"
        f"ChEMBL ID       : {compound.get('chembl_id', 'N/A')}\n"
        f"SMILES           : {smiles}\n\n"
        f"PROPERTIES:\n"
        f"  Molecular Weight : {compound.get('mw', 'N/A')}\n"
        f"  LogP             : {compound.get('logp', 'N/A')}\n"
        f"  H-Bond Donors    : {compound.get('hbd', 'N/A')}\n"
        f"  H-Bond Acceptors : {compound.get('hba', 'N/A')}\n"
        f"  TPSA             : {compound.get('tpsa', 'N/A')}\n"
        f"  Rotatable Bonds  : {compound.get('rot_bonds', 'N/A')}\n\n"
        f"BINDING DATA:\n"
        f"  Assay Type : {compound.get('activity_type', 'N/A')}\n"
        f"  Value      : {compound.get('activity_value', 'N/A')} "
        f"{compound.get('activity_units', '')}\n\n"
        f"FILTERS:\n"
        f"  Rule of 5  : "
        f"{'Pass' if compound.get('ro5_pass') else 'Fail'} "
        f"({compound.get('ro5_violations', '?')} violations)\n"
        f"  ADME Score : {compound.get('adme_score', 'N/A')}/6\n"
        f"{'=' * 50}"
    )

    mol = Chem.MolFromSmiles(smiles)
    img = None
    if mol:
        img = draw_2d_with_pharmacophore(mol, size=(400, 400))

    # Build 3D viewer HTML
    logger.info(f"Building 3D viewer for {name}")
    try:
        viewer_3d = build_3d_compound_viewer_html(smiles, name)
    except Exception as exc:
        logger.error(f"3D viewer generation failed for {name}: {exc}")
        viewer_3d = (
            '<div style="height:400px;display:flex;align-items:center;'
            'justify-content:center;background:#f5f5f5;border-radius:8px;">'
            f'<p style="color:#c00;">3D viewer error: {exc}</p></div>'
        )

    return detail, img, viewer_3d, compound


def ai_explain_compound(selected_state, protein_state):
    """AI-generated explanation of how the compound interacts with the protein."""
    logger.info("=== Generating AI compound explanation ===")
    if not selected_state:
        logger.warning("AI explanation requested with no compound selected")
        return "Select a compound from the results table first."
    if not protein_state:
        logger.warning("AI explanation requested with no protein data")
        return "No protein data available."

    client = get_openai_client()
    if not client:
        logger.warning("AI explanation requested but OpenAI API key not set")
        return "[OpenAI API key not set — cannot generate explanation]"

    c = selected_state
    p = protein_state
    logger.info(f"Generating AI explanation for {c.get('name')} targeting {p['pdb_id']}")

    prompt = (
        "You are an expert medicinal chemist. Explain how this compound "
        "works in the context of its protein target.\n\n"
        f"PROTEIN:\n"
        f"  Name: {p['title']}\n"
        f"  PDB ID: {p['pdb_id']}\n"
        f"  Classification: {p['classification']}\n"
        f"  Organism: {p['organism']}\n\n"
        f"COMPOUND:\n"
        f"  Name: {c.get('name', 'Unknown')}\n"
        f"  SMILES: {c.get('smiles', 'N/A')}\n"
        f"  MW: {c.get('mw', 'N/A')}, LogP: {c.get('logp', 'N/A')}\n"
        f"  Binding: {c.get('activity_type', 'N/A')} = "
        f"{c.get('activity_value', 'N/A')} {c.get('activity_units', '')}\n\n"
        "Explain:\n"
        "1. How this compound likely binds to the protein\n"
        "2. Key functional groups and their roles\n"
        "3. Mechanism of action (inhibitor, agonist, etc.)\n"
        "4. Strengths and weaknesses\n"
        "5. Potential optimization strategies\n\n"
        "Be concise and scientifically accurate."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": prompt}],
        )
        logger.info("AI compound explanation completed successfully")
        return resp.choices[0].message.content
    except Exception as exc:
        logger.error(f"AI compound explanation failed: {exc}")
        return f"AI analysis failed: {exc}"


def build_protein_ligand_viewer_html(
    smiles: str, pdb_id: str, compound_name: str = "", binding_center=None,
    pdb_text: str = "",
) -> str:
    """Build a 3Dmol.js iframe showing the protein + compound at the binding site.

    The compound's 3D centroid is translated to the detected binding-site centre so
    it is placed inside the pocket for visual inspection (not a true docked pose).

    If *pdb_text* is supplied it is embedded directly in the HTML so the browser
    does not need to fetch the PDB from RCSB again (avoids a redundant download).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (
            '<div style="height:520px;display:flex;align-items:center;'
            'justify-content:center;background:#f5f5f5;border-radius:8px;">'
            '<p style="color:#c00;">Could not parse SMILES for 3D generation</p></div>'
        )

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    try:
        ff = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        if ff != 0:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    # Translate compound centroid to binding site centre
    cx, cy, cz = (binding_center if binding_center else (0.0, 0.0, 0.0))
    try:
        conf = mol.GetConformer()
        positions = conf.GetPositions()          # numpy array (N, 3)
        centroid = positions.mean(axis=0)
        dx, dy, dz = cx - centroid[0], cy - centroid[1], cz - centroid[2]
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (pos.x + dx, pos.y + dy, pos.z + dz))
    except Exception as exc:
        logger.warning(f"Could not translate compound to binding site: {exc}")

    sdf_block = Chem.MolToMolBlock(mol)
    sdf_json = json.dumps(sdf_block)

    pdb_id_safe = pdb_id.strip().upper()
    title_label = (
        f"Compound in Binding Site: {html_module.escape(compound_name)}"
        if compound_name
        else "Compound in Binding Site"
    )

    title_html = (
        f'<div style="padding:6px 12px;font-size:12px;color:#444;'
        f'border-bottom:1px solid #ddd;background:#f0f4ff;'
        f'display:flex;justify-content:space-between;align-items:center;">'
        f'<b>{title_label}</b>'
        f'<span style="color:#888;font-size:11px;">'
        f'Protein: semi-transparent cartoon &nbsp;|&nbsp; '
        f'Compound: stick/sphere &nbsp;|&nbsp; '
        f'Yellow sphere: binding region</span>'
        f'</div>'
    )

    # JS that runs once pdbData is in scope (viewer/sdf/cx/cy/cz are closures)
    render_js = (
        "  document.getElementById('loading').style.display = 'none';"
        "  viewer.addModel(pdbData, 'pdb');"
        "  var n = viewer.getNumModels();"
        "  viewer.setStyle({}, {cartoon: {color: 'spectrum', opacity: 0.72}});"
        "  viewer.addModel(sdf, 'sdf');"
        "  viewer.setStyle({model: n}, {"
        "    stick: {radius: 0.18, colorscheme: 'Jmol'},"
        "    sphere: {scale: 0.28, colorscheme: 'Jmol'}"
        "  });"
        "  if (cx !== 0 || cy !== 0 || cz !== 0) {"
        "    viewer.addSphere({center: {x: cx, y: cy, z: cz},"
        "      radius: 9, color: 'yellow', opacity: 0.09, wireframe: false});"
        "    viewer.addSphere({center: {x: cx, y: cy, z: cz},"
        "      radius: 9.2, color: 'orange', opacity: 0.18, wireframe: true});"
        "  }"
        "  viewer.zoomTo({model: n});"
        "  viewer.render();"
    )

    if pdb_text:
        pdb_json = json.dumps(pdb_text)
        scene_js = (
            "  var pdbData = " + pdb_json + ";"
            + render_js
        )
    else:
        scene_js = (
            f"  fetch('https://files.rcsb.org/download/{pdb_id_safe}.pdb')"
            "    .then(function(r){return r.ok?r.text():Promise.reject(r.status);})"
            "    .then(function(pdbData){"
            + render_js +
            "    })"
            "    .catch(function(){"
            "      document.getElementById('loading').textContent='Failed to load protein structure.';"
            "    });"
        )

    inner = (
        "<!DOCTYPE html><html><head>"
        "<script src='https://3Dmol.org/build/3Dmol-min.js'></script>"
        "<style>"
        "body{margin:0;padding:0;overflow:hidden;background:#fff;font-family:sans-serif}"
        "#viewer{width:100%;height:100%;position:absolute;top:0;left:0}"
        "#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);"
        "color:#555;font-size:13px;z-index:10;pointer-events:none;"
        "background:rgba(255,255,255,0.85);padding:8px 16px;border-radius:6px;}"
        "</style>"
        "</head><body>"
        "<div id='viewer'></div>"
        "<div id='loading'>Loading protein structure&hellip;</div>"
        "<script>"
        "window.addEventListener('load', function() {"
        "  var sdf = " + sdf_json + ";"
        f"  var cx = {cx}, cy = {cy}, cz = {cz};"
        "  var viewer = $3Dmol.createViewer('viewer', {backgroundColor: 'white'});"
        + scene_js +
        "});"
        "</script>"
        "</body></html>"
    )

    escaped = html_module.escape(inner, quote=True)
    return (
        '<div style="border:1px solid #c8d8f0;border-radius:8px;overflow:hidden;">'
        + title_html
        + f'<iframe srcdoc="{escaped}" '
        'style="width:100%;height:510px;border:none;display:block;"></iframe>'
        '</div>'
    )


def show_docked_view(selected_state, protein_state):
    """Callback: render the selected compound positioned at the protein binding site."""
    if not selected_state or not protein_state:
        return DOCKED_3D_PLACEHOLDER

    smiles = selected_state.get("smiles", "")
    name = selected_state.get("name", "")
    pdb_id = protein_state.get("pdb_id", "")

    if not smiles or not pdb_id:
        return DOCKED_3D_PLACEHOLDER

    # Detect binding centre from the crystal structure.
    # Keep the fetched PDB text so we can embed it directly in the HTML
    # instead of downloading it a second time inside the browser iframe.
    binding_center = None
    pdb_text = ""
    try:
        from tabs.pipeline.docking import fetch_pdb_file, find_binding_center
        pdb_text = fetch_pdb_file(pdb_id)
        binding_center = find_binding_center(pdb_text)
        logger.info(f"Binding centre for {pdb_id}: {binding_center}")
    except Exception as exc:
        logger.warning(f"Could not determine binding centre: {exc}")

    docked_sdf = selected_state.get("docked_sdf", "")
    if docked_sdf:
        logger.info(f"Using Vina-docked pose for {name}")
    else:
        logger.info(f"No docked pose available for {name}, using conformer fallback")

    try:
        return build_receptor_interaction_viewer_html(smiles, pdb_id, name, binding_center, pdb_text, docked_sdf)
    except Exception as exc:
        logger.error(f"Receptor interaction viewer failed: {exc}")
        return (
            '<div style="height:570px;display:flex;align-items:center;'
            'justify-content:center;background:#0d1117;border-radius:8px;">'
            f'<p style="color:#f87171;">Viewer error: {html_module.escape(str(exc))}</p></div>'
        )


def populate_compound_selector(final_state):
    """Build a summary table of final compounds for selection."""
    if not final_state:
        return None

    rows = []
    for c in final_state:
        vina = c.get("vina_affinity")
        rows.append({
            "Rank": c.get("binding_rank", ""),
            "Name": c["name"][:25],
            "Vina (kcal/mol)": (
                f"{vina:.2f}" if isinstance(vina, (int, float)) and vina != float("inf")
                else "N/A"
            ),
            "ADME": f"{c.get('adme_score', '?')}/6",
            "MW": c.get("mw", ""),
            "LogP": c.get("logp", ""),
        })

    logger.info(f"Populated compound selector with {len(rows)} compounds")
    return pd.DataFrame(rows)