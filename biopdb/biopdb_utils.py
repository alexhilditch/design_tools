import numpy as np

from Bio.PDB import *
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import Superimposer
from Bio.PDB.Chain import Chain
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1

from common.common import expand

pdb_parser = PDBParser(QUIET=True)
super_imposer = Superimposer()
pdb_io = PDBIO()

def get_ca_atoms(model, chain_id='A'):

    '''
    get all the Ca atoms for a chain of a biopdb model

    '''

    if type(model) != Bio.PDB.Model.Model:
        model = pdb_parser.get_structure("sample", model)[0]

    ca_atoms = []

    for chain in model.get_chains():
        if chain.id == chain_id:
            for residue in chain:
                try:
                    ca_atoms.append(residue['CA'])
                except KeyError:
                    continue

    return ca_atoms

def compute_protein_ligand_sasa(model, ligand_name):

    '''
    compute the SASA for a protein-ligand complex

    returns protein_sasa, ligand_sasa
    '''

    if type(model) != Bio.PDB.Model.Model:
        model = pdb_parser.get_structure("sample", model)[0]

    ShrakeRupley().compute(model, level="R")

    protein_sasa = sum([r.sasa for r in model.get_residues()])

    ligand_in_complex = [res for res in model.get_residues() if res.get_resname() == ligand_name]
    assert len(ligand_in_complex) == 1, f'Error: Multiple copies of ligand found - {len(ligand_in_complex)}'
    ligand_in_complex = ligand_in_complex[0]

    ligand_sasa = ligand_in_complex.sasa
 
    return protein_sasa, ligand_sasa

def compute_protein_sasa(model):

    '''
    compute the SASA for a protein

    returns protein_sasa
    '''

    if type(model) != Bio.PDB.Model.Model:
        model = pdb_parser.get_structure("sample", model)[0]

    ShrakeRupley().compute(model, level="R")

    protein_sasa = sum([r.sasa for r in model.get_residues()])
 
    return protein_sasa

def move_chains(chains: list, vector: list) -> None:
    """
    chains: list of Bio.PDB.chain objects to translate
    vector: list, [x, y, z] 
    
    Utility function to move chains of a pdb in a direction defined by the vector
    """
    
    p = Bio.PDB.vectors.Vector(0, 0, 0)
    q = Bio.PDB.vectors.Vector(0, 0, 0)
    rotation = Bio.PDB.vectors.rotmat(p, q)
    
    for chain in chains:
        for residue in chain:
            for atom in residue:
                atom.transform(rotation, vector)
    
    return None

def get_sequence(model):

    if type(model) != Bio.PDB.Model.Model:
        model = pdb_parser.get_structure("sample", model)[0]

    sequence_dict = {}

    for chain in model:
        sequence = []
        for residue in chain:
            if residue.id[0] == " ":
                sequence.append(seq1(residue.resname))
        sequence_dict[chain.id] = sequence

    return sequence_dict

def align_pdbs(ref_model, sample_model, residues:list=[], ref_residues:list=[], 
               mode:str='CA', return_aligned:bool=False, per_res_rmsd: bool=False):
    """
        ref_model:  Bio.PDB.Model.Model, model of the reference, must be called 'reference' 
                    (or str: path to the pdb file)
        sample_model:   Bio.PDB.Model.Model, model to be aligned to ref_model, must be called 'sample'
                        (or str: path to the pdb file)
        residues: list, list of residues to be aligned (pdb numbering), will align all residues if empty
        ref_residues: list, residues for alignment in the reference pdb. Will use residues if not provided
        mode: str, either 'CA' or 'all_atom'
        return_aligned: bool, whether to return the aligned sample_pdb
        per_res_rmsd: bool, whether to return per residue rmsd
        
        Aligns the C alphas in two pdb structures using Biopython. Returns the C alpha rmsd of the alignment.
        You can specify the chain letter in the residues list like this: ['A10', 'A13', 'A16-18'].
        
        NOTE: each residue in residues needs to start with the chain letter!
        NOTE: list of residues for reference and sample structure must be of same length!
        NOTE: in all atom mode hydrogen atoms are removed!
    """

    # Sanity check
    mode = mode.upper()
    if mode not in ['CA', 'ALL_ATOM']:
        print(f'ERROR: align_pdbs() mode must be "CA" or "ALL_ATOM"! You provided "{mode}"')
        return None

    if type(ref_model) != Bio.PDB.Model.Model or type(sample_model) != Bio.PDB.Model.Model:
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)    
    if type(ref_model) != Bio.PDB.Model.Model:
        ref_model = pdb_parser.get_structure("reference", ref_model)[0]
    if type(sample_model) != Bio.PDB.Model.Model:
        sample_model = pdb_parser.get_structure("sample", sample_model)[0]

    super_imposer = Bio.PDB.Superimposer()

    if residues == []:
        align_all = True
    else:
        align_all = False
        residues = expand(residues)
                    
        if ref_residues == []:
            ref_residues = residues
        else:
            ref_residues = expand(ref_residues)
    
    # creating dict with residues to be aligned
    alignment_residues = {'reference': ref_residues, 'sample': residues}

    # now creating lists of atoms to align - stored in a dict
    align_atoms = {'reference':[], 'sample':[]}

    for model in [ref_model, sample_model]:

        for residue in model.get_residues():
            # check if residue is HETATM and skip if True
            hetatm = residue.get_full_id()[3][0]
            if hetatm != " ":
                continue

            # generating residue name (e.g. A10) for residue
            res_name = residue.parent.get_id() + str(residue.get_id()[1])

            # name of the structure (either 'reference' or 'sample')
            structure = residue.get_full_id()[0]
            
            if res_name in alignment_residues[structure] or align_all:
                if mode == 'CA':
                    align_atoms[structure].append(residue['CA'])
                elif mode == 'ALL_ATOM':                    
                    for atom in residue.get_atoms():
                        align_atoms[structure].append(atom)
                
    # removing hydrogen atoms from all atom mode
    if mode == 'ALL_ATOM':
        for structure in align_atoms:
            align_atoms[structure] = [atom for atom in align_atoms[structure] if not atom.id.startswith('H')]

    # Superimpose sample model on reference model
    super_imposer.set_atoms(align_atoms['reference'], align_atoms['sample'])
    super_imposer.apply(sample_model.get_atoms())

    # this is probably not the best idea but works
    if per_res_rmsd:
        rmsd = {}
        for ref_atm, spl_atm in zip(align_atoms['reference'], align_atoms['sample']):
            dst = np.sum((ref_atm.coord - spl_atm.coord) ** 2)
            chain = spl_atm.parent.parent.get_id()
            res_num = str(spl_atm.parent.get_id()[1])
            res_id = chain + res_num
            if res_id not in rmsd.keys():
                rmsd[res_id] = []

            rmsd[res_id].append(dst)
        
        # convert to mean of entire residue
        for res_id in rmsd.keys():
            rmsd[res_id] = np.sqrt(sum(rmsd[res_id]) / len(rmsd[res_id]))
            
    else:
        rmsd = super_imposer.rms

    if return_aligned:
        return rmsd, sample_model
    else:
        return rmsd
