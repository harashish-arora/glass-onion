"""
3D Conformer Visualization for Solute-Solvent Pairs

Generates 3D-looking conformer images from a random solute-solvent pair
in train.csv. Suitable for publication figures.

Usage:
    python visualize_conformers.py
    
Or import and use interactively:
    from visualize_conformers import visualize_random_pair
    solute_img, solvent_img = visualize_random_pair()
"""

import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import os


def generate_3d_conformer_image(smiles, name, size=(400, 350), output_dir="."):
    """
    Generate a 3D-looking conformer image from SMILES.
    
    Args:
        smiles: SMILES string
        name: Name for the saved file
        size: Tuple of (width, height) in pixels
        output_dir: Directory to save the image
        
    Returns:
        Path to saved image file, or None if failed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES: {smiles}")
        return None
    
    # Add hydrogens and generate 3D conformer using ETKDGv3
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    
    if result == -1:
        # Fallback to random coords if embedding fails
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    # Optimize geometry with MMFF force field
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass  # Continue even if optimization fails
    
    # Remove Hs for cleaner visualization
    mol_no_h = Chem.RemoveHs(mol)
    
    # Generate 2D coords from 3D (gives nice pseudo-3D perspective)
    AllChem.Compute2DCoords(mol_no_h)
    
    # Draw with nice styling for publication
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    opts.addStereoAnnotation = True
    opts.addAtomIndices = False
    opts.padding = 0.15
    
    drawer.DrawMolecule(mol_no_h)
    drawer.FinishDrawing()
    
    img_data = drawer.GetDrawingText()
    
    # Save to file
    filename = os.path.join(output_dir, f"{name}_conformer.png")
    with open(filename, 'wb') as f:
        f.write(img_data)
    print(f"Saved: {filename}")
    
    return filename


def visualize_random_pair(csv_path="data/train.csv", output_dir=".", seed=None):
    """
    Pick a random solute-solvent pair and generate conformer images.
    
    Args:
        csv_path: Path to train.csv
        output_dir: Directory to save images
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Tuple of (solute_image_path, solvent_image_path)
    """
    if seed is not None:
        random.seed(seed)
    
    # Load data and pick random pair
    df = pd.read_csv(csv_path)
    row = df.sample(1).iloc[0]
    
    solute_smiles = row['Solute']
    solvent_smiles = row['Solvent']
    
    print("=" * 50)
    print("Random Solute-Solvent Pair")
    print("=" * 50)
    print(f"  Solute:      {solute_smiles}")
    print(f"  Solvent:     {solvent_smiles}")
    print(f"  Temperature: {row['Temperature']} K")
    print(f"  LogS:        {row['LogS']:.4f}")
    print()
    
    print("Generating conformer images...")
    print()
    
    solute_path = generate_3d_conformer_image(
        solute_smiles, "solute", output_dir=output_dir
    )
    solvent_path = generate_3d_conformer_image(
        solvent_smiles, "solvent", output_dir=output_dir
    )
    
    print()
    print("Done! Images ready for copy/paste into figures.")
    
    

    return solute_path, solvent_path

    


if __name__ == "__main__":
    visualize_random_pair()
