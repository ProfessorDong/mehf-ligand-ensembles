"""Build pocket10 PDBs for the 100 CrossDocked test pockets, using TargetDiff's
OWN released code (utils.data.PDBProtein.query_residues_ligand /
residues_to_pdb_block, radius 10), so the pocket definition is theirs, not ours.
The pocket list is taken from the DiffSBDD released sample filenames so that the
two published models are measured on the IDENTICAL 100 pockets."""
import sys, os, re, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / 'data/external/targetdiff/code'
sys.path.insert(0, str(TD))
from utils.data import PDBProtein, parse_sdf_file

GEN = ROOT / 'data/external/diffsbdd/crossdocked_fullatom_cond'
TEST = ROOT / 'data/external/crossdocked/test_set'
OUT = ROOT / 'data/external/crossdocked/pockets10'
OUT.mkdir(parents=True, exist_ok=True)

index = []
missing = []
for f in sorted(GEN.glob('*.sdf')):
    stem = f.name.split('-pocket10_')[0]          # 14gs-A-rec-20gs-cbd-lig-tt-min-0
    lig = stem.replace('-', '_') + '.sdf'          # 14gs_A_rec_20gs_cbd_lig_tt_min_0.sdf
    hits = list(TEST.glob(f'*/{lig}'))
    if not hits:
        missing.append(lig); continue
    ligp = hits[0]
    recname = lig.split('_rec_')[0] + '_rec.pdb'
    recp = ligp.parent / recname
    if not recp.exists():
        missing.append(str(recp)); continue
    protein = PDBProtein(recp.read_text())
    ligand = parse_sdf_file(str(ligp))
    block = protein.residues_to_pdb_block(protein.query_residues_ligand(ligand, 10))
    dest = OUT / (stem.replace('-', '_') + '_pocket10.pdb')
    dest.write_text(block)
    index.append({'gen_file': f.name, 'pocket_pdb': dest.name, 'ref_ligand': str(ligp),
                  'n_pocket_atoms': sum(1 for l in block.splitlines() if l.startswith('ATOM'))})
json.dump({'n': len(index), 'missing': missing, 'index': index},
          open(OUT / 'index.json', 'w'), indent=2)
print('built', len(index), 'pockets; missing', len(missing))
import numpy as np
na = [d['n_pocket_atoms'] for d in index]
print('pocket atoms: min %d median %d max %d' % (min(na), int(np.median(na)), max(na)))
