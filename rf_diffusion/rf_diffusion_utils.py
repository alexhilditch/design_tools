import os
import shutil
import glob
import numpy as np
from common.common_utils import make_directory

def get_rfdiff_pdbs(directory):

    '''
    function to search a directory of rfdiffusion runs and return only those
    with an associated trb file
    '''

    pdbs = glob.glob(f'{directory}/*.pdb')

    print(f'Found {len(pdbs)} PDB files in {directory}')

    # check whether all of the pdb files have an associated trb file.
    # if they do not, we will move them to a new directory and exclude them

    print(f'Checking for the associated trb files')

    incomplete_dir = os.path.join(directory, 'incomplete')
    make_directory(incomplete_dir)

    no_trb_file = []

    for n, file in enumerate(pdbs):
        trb_file = file.replace('.pdb', '.trb')

        if os.path.exists(trb_file) == False:
            no_trb_file.append(file)

    if len(no_trb_file) == 0:
        print(f'All files have an associated trb file')

    elif len(no_trb_file) != 0:
        print(f'Found {len(no_trb_file)} files without an associated trb file')
        print(f'Moving {len(no_trb_file)} incomplete PDB files to: {incomplete_dir}')

        for file in no_trb_file:
            fname = os.path.basename(file)
            new_path = os.path.join(incomplete_dir, fname)
            shutil.move(file, new_path)

    pdbs = glob.glob(f'{directory}/*.pdb')

    print(f'{len(pdbs)} completed RFdiffusion runs in {directory}')

    pdbs.sort()

    return pdbs

def get_contigs(file:str):
    '''
    get the contigs from the trb file and return them as a list
    
    accepts either a trb file, or a pdb file
    '''
    
    #pulling the trb file to find the fixed and diffused residues

    file_extension = os.path.splitext(file)[1]

    assert file_extension in ['.pdb', '.trb'], f'File {file} must be of type .pdb or .trb'

    if file_extension == '.pdb':
        trb_file = file.replace('.pdb', '.trb')
    elif file_extension == '.trb':
        trb_file = file

    trb = np.load(trb_file, allow_pickle=True)

    # get the original contigs
    contigs = list(trb['config']['contigmap']['contigs'][0].split(','))

    return contigs

def get_fixed_positions(file):
    '''
    get the fixed positions from the trb file and return them as a list

    returns positions in pdb indexing! (1 indexed)

    returns the reference_fixed positions (from the input) and the new fixed positions (from the output)
    
    accepts either a trb file, or a pdb file
    '''
    
    #pulling the trb file to find the fixed and diffused residues

    file_extension = os.path.splitext(file)[1]

    assert file_extension in ['.pdb', '.trb'], f'File {file} must be of type .pdb or .trb'

    if file_extension == '.pdb':
        trb_file = file.replace('.pdb', '.trb')
    elif file_extension == '.trb':
        trb_file = file

    trb = np.load(trb_file, allow_pickle=True)

    con_hal_idx0 = trb['con_hal_idx0']
    con_ref_idx0 = trb['con_ref_idx0']

    reference_fixed = [x + 1 for x in con_ref_idx0] # pymol numbering
    new_fixed = [x + 1 for x in con_hal_idx0] # pymol numbering

    return reference_fixed, new_fixed