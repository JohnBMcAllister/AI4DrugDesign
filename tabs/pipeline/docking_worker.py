"""
Docking worker module for parallel compound docking.

This module is separate from integrated_pipeline.py to avoid importing Gradio
in worker processes, which would cause each worker to try launching a Gradio server.
"""

import os
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina


def smiles_to_pdbqt(smiles: str, output_path: str) -> None:
    """Convert SMILES to PDBQT file."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Exception(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        raise Exception("Failed to generate 3D coordinates")

    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    if not is_ok:
        raise Exception(f"PDBQT conversion failed: {error_msg}")

    with open(output_path, "w") as f:
        f.write(pdbqt_string)


def dock_single_compound(args: tuple) -> tuple[int, float, str]:
    """
    Worker function to dock a single compound (for parallel processing).

    Args:
        args: Tuple of (idx, smiles, receptor_pdbqt_path, center_coords)

    Returns:
        Tuple of (compound_index, vina_affinity_score, docked_sdf_block)
        docked_sdf_block is an empty string if pose extraction fails.
    """
    idx, smiles, receptor_pdbqt, center = args

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ligand_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")
            smiles_to_pdbqt(smiles, ligand_pdbqt)

            v = Vina(sf_name="vina", cpu=1, verbosity=0)
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_file(ligand_pdbqt)
            v.compute_vina_maps(center=list(center), box_size=[20, 20, 20])
            v.dock(exhaustiveness=4, n_poses=1)
            energies = v.energies(n_poses=1)
            affinity = float(energies[0][0])

            # Extract the docked 3D pose and convert to SDF for the visualizer
            docked_sdf = ""
            try:
                poses_pdbqt = v.poses(n_poses=1)
                from meeko import PDBQTMolecule
                pdbqt_mol = PDBQTMolecule(poses_pdbqt, skip_typing=True)
                for pose in pdbqt_mol:
                    rdkit_mol = pose.export_rdkit_mol()
                    docked_sdf = Chem.MolToMolBlock(rdkit_mol)
                    break
            except Exception:
                pass  # viewer will fall back to conformer generation

            return (idx, affinity, docked_sdf)

    except Exception:
        return (idx, float('inf'), "")
