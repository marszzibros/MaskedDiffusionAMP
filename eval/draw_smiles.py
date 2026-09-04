"""Render SMILES strings to PNG.

Single molecule or a grid, from the command line or as an import:

    python draw_smiles.py "CC(=O)Oc1ccccc1C(=O)O" -o figures/aspirin.png
    python draw_smiles.py "CCO" "c1ccccc1" -o figures/two.png --per-row 2
    python draw_smiles.py --from-file smiles.txt -o figures/sheet.png

    from draw_smiles import draw_smiles, draw_grid
    draw_smiles("CCO", "figures/ethanol.png")

Invalid SMILES are reported rather than silently dropped -- the generated
strings coming out of sampling are frequently unparseable, and a grid that
quietly shrinks hides that.
"""
import argparse
import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")


def parse(smiles, sanitize=True):
    """SMILES -> Mol with 2D coordinates, or None if RDKit cannot read it."""
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        return None
    if not sanitize:
        # Partial sanitisation: enough to lay the molecule out without
        # rejecting it for a bad valence or an unkekulisable ring.
        mol.UpdatePropertyCache(strict=False)
        Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_RDKIT)
    rdDepictor.Compute2DCoords(mol)
    return mol


def _write(drawer, out_path):
    drawer.FinishDrawing()
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(drawer.GetDrawingText())
    return out_path


def draw_smiles(smiles, out_path, size=(500, 500), legend="",
                highlight_smarts=None, sanitize=True, transparent=False):
    """Write one molecule to `out_path` as PNG. Returns the path.

    highlight_smarts: optional SMARTS whose matching atoms/bonds are shaded,
    useful for pointing at the pharmacophore in a generated molecule.
    """
    mol = parse(smiles, sanitize=sanitize)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    atoms, bonds = [], []
    if highlight_smarts:
        patt = Chem.MolFromSmarts(highlight_smarts)
        if patt is None:
            raise ValueError(f"bad SMARTS: {highlight_smarts!r}")
        for match in mol.GetSubstructMatches(patt):
            atoms.extend(match)
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match:
                    bonds.append(bond.GetIdx())

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    if transparent:
        opts.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, legend=legend,
        highlightAtoms=atoms or None, highlightBonds=bonds or None,
    )
    return _write(drawer, out_path)


def draw_grid(smiles_list, out_path, per_row=4, sub_size=(300, 300),
              legends=None, sanitize=True):
    """Write several molecules to one PNG grid. Returns (path, n_failed)."""
    mols, kept_legends, failed = [], [], []
    legends = legends if legends is not None else list(smiles_list)
    for smi, leg in zip(smiles_list, legends):
        mol = parse(smi, sanitize=sanitize)
        if mol is None:
            failed.append(smi)
            continue
        mols.append(mol)
        kept_legends.append(leg)

    if not mols:
        raise ValueError("no parseable SMILES to draw")

    img = Draw.MolsToGridImage(
        mols, molsPerRow=min(per_row, len(mols)),
        subImgSize=sub_size, legends=kept_legends, useSVG=False,
        returnPNG=False,
    )
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    img.save(out_path)
    return out_path, failed


def main():
    ap = argparse.ArgumentParser(description="Render SMILES to a PNG image.")
    ap.add_argument("smiles", nargs="*", help="one or more SMILES strings")
    ap.add_argument("--from-file", help="file with one SMILES per line "
                                        "(optionally 'SMILES<tab>legend')")
    ap.add_argument("-o", "--out", default="molecule.png", help="output PNG path")
    ap.add_argument("--size", type=int, nargs=2, default=(500, 500),
                    metavar=("W", "H"), help="image size for a single molecule")
    ap.add_argument("--per-row", type=int, default=4, help="grid columns")
    ap.add_argument("--sub-size", type=int, nargs=2, default=(300, 300),
                    metavar=("W", "H"), help="per-cell size in a grid")
    ap.add_argument("--legend", default="", help="caption for a single molecule")
    ap.add_argument("--no-legend", action="store_true",
                    help="drop the SMILES captions under grid cells")
    ap.add_argument("--highlight", help="SMARTS to highlight (single molecule)")
    ap.add_argument("--no-sanitize", action="store_true",
                    help="draw structures RDKit would otherwise reject")
    ap.add_argument("--transparent", action="store_true",
                    help="transparent background (single molecule)")
    args = ap.parse_args()

    smiles, legends = list(args.smiles), None
    if args.from_file:
        legends = []
        with open(args.from_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                smiles.append(parts[0])
                legends.append(parts[1] if len(parts) > 1 else parts[0])
        if len(smiles) != len(legends):  # positional SMILES mixed with a file
            legends = None
    if not smiles:
        ap.error("give at least one SMILES, or --from-file")

    sanitize = not args.no_sanitize
    if len(smiles) == 1:
        try:
            path = draw_smiles(smiles[0], args.out, size=tuple(args.size),
                               legend=args.legend, highlight_smarts=args.highlight,
                               sanitize=sanitize, transparent=args.transparent)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            print("       (try --no-sanitize to draw it anyway)", file=sys.stderr)
            sys.exit(1)
        print(f"wrote {path}")
        return

    if args.no_legend:
        legends = [""] * len(smiles)
    path, failed = draw_grid(smiles, args.out, per_row=args.per_row,
                             sub_size=tuple(args.sub_size), legends=legends,
                             sanitize=sanitize)
    print(f"wrote {path} ({len(smiles) - len(failed)}/{len(smiles)} drawn)")
    for smi in failed:
        print(f"  unparseable: {smi}", file=sys.stderr)


if __name__ == "__main__":
    main()
