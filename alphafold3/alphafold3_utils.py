# general imports
import os
import shutil
import glob
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from common.common import make_directory

def check_af3_completion(input_directory, output_directory, delete_unrecognised=False):

    '''
    A function to check which of the input json files have been completed

    input directory is the top level input directory - this can contain several sub directories
    is the job has been split up

    the output directory is a single directory, where every sub directory is a prediction
    i.e., all input directories must point to a single output directory.

    if the job has been begun but not completed the directory will be made, but the ranking_scores.csv
    file not made. This will be added as a partial completion.

    If there is a directory in the output directory that does not match an input json (this happens most 
    frequently if one of the predictions has been duplicated, and a unique timestamp appended to the directory
    name), it will be added as unrecognised, and can be optionally deleted to not interfere with scoring.
    (delete_unrecognised = True)

    Care must be taken that the output directory does not contain any subfolders that should not be deleted, as
    these will be identified as unrecognised.

    returns: complete, partially_complete, incomplete, unrecognised
    '''

    jsons = glob.glob(f'{input_directory}/**/*.json', recursive=True)
    input_job_names = [os.path.basename(x).split('.json')[0] for x in jsons]

    num_jsons = len(jsons)

    complete = []
    partially_complete = []
    unrecognised = []

    # iterate over all directories
    for folder in os.listdir(output_directory):
        job_name = os.path.basename(folder)
        if job_name not in input_job_names:
            unrecognised.append(job_name)
        else:
            # prediction complete if ranking was done
            if f'{job_name}_ranking_scores.csv' in os.listdir(os.path.join(output_directory, folder)):
                complete.append(job_name)
            elif f'{job_name}_ranking_scores.csv' not in os.listdir(os.path.join(output_directory, folder)):
                partially_complete.append(job_name)
    
    incomplete = [x for x in input_job_names if x not in complete and x not in partially_complete]

    num_completed = len(complete)
    num_partially_completed = len(partially_complete)
    num_incomplete = len(incomplete)
    num_unrecognised = len(unrecognised)

    print(f'Found {num_jsons} json files in input directory: {input_directory}. \n')
    print(f'{num_completed} predictions have been completed.')
    print(f'{num_partially_completed} predictions are partially complete')
    print(f'There are {num_incomplete} incomplete predictions.')
    print(f'There are {num_unrecognised} unrecognised predictions (matching json not found).\n')

    unrecognised_predictions_deleted = 0

    if delete_unrecognised:
        for file_name in unrecognised:
            shutil.rmtree(f'{output_directory}/{file_name}')
            unrecognised_predictions_deleted += 1

        print(f'Deleted {unrecognised_predictions_deleted} unrecognised predictions.\n')

    return complete, partially_complete, incomplete, unrecognised

def collect_af3_scores(scores_directory):

    '''
    A function to collect all af3 scores from the output directory

    The expected input is the output directory from af3 - each sub directory in this directory is a prediction

    This function collects some basic scores from the af3 output

    Note there are 4 possible molecule types - protein, ligand, dna, and rna

    Returns the per chain plddt, and the plddt by molecule type (i.e., the average protein plddt)

    Scores returned:

    ID
    top model path
    Sequence (by chain - if protein, dna, or rna)
    plddt (by chain)
    plddt (by molecule type)
    iptm
    ptm

    returns a pandas dataframe
    '''

    out = {}
    failed = []

    num_predictions = len(os.listdir(scores_directory))

    print(f'Parsing {num_predictions} predictions')

    for element in tqdm(os.listdir(scores_directory)):
        directory = os.path.join(scores_directory, element)

        if os.path.isdir(directory):

            # get the 3 json files that we need
            data = os.path.join(directory, element + '_data.json')
            summary = os.path.join(directory, element + '_summary_confidences.json')
            details = os.path.join(directory, element + '_confidences.json')

            try:
                with open(data, 'r') as fobj:
                    meta_data = json.load(fobj)
                with open(summary, 'r') as fobj:
                    summary_data = json.load(fobj)
                with open(details, 'r') as fobj:
                    detail_data = json.load(fobj)
            except FileNotFoundError:
                failed.append(element)
                continue

            # get some basic information
            id = meta_data['name']

            # set up an empty dictionary to add scores for this prediction
            out[id] = {}

            out[id]['path'] = os.path.join(directory, element + '_model.cif')

            # from the data file, parse the chains in the prediction
            # and the molecule types for each chain
            # if the chain is a protein or nucleic acid, get the sequence

            molecule_types = {}
            sequences = {}

            for item in meta_data['sequences']:
                for molecule_type, value in item.items():
                    chain = value['id']
                    molecule_types[chain] = molecule_type
                    if molecule_type in ['protein', 'dna', 'rna']:
                        sequences[chain] = value['sequence']

            out[id]['molecule_type'] = molecule_types
            out[id]['sequences'] = sequences

            # for each of the chains, get the mean plddt
            chains_np = np.array(detail_data['atom_chain_ids'])
            plddt_np = np.array(detail_data['atom_plddts'])

            # get the per chain plddt
            for chain in molecule_types.keys():
                mask = np.isin(chains_np, chain)
                plddt = np.average(plddt_np[mask])
                out[id][f'plddt_chain_{chain}'] = plddt
            
            # group the chains by molecule type, and get the mean plddt by molecule type
            protein_chains = [x for x, y in molecule_types.items() if y == 'protein']
            ligand_chains = [x for x, y in molecule_types.items() if y == 'ligand']
            rna_chains = [x for x, y in molecule_types.items() if y == 'rna']
            dna_chains = [x for x, y in molecule_types.items() if y == 'dna']

            if len(protein_chains) != 0:
                mask = np.isin(chains_np, protein_chains)
                mean_protein_plddt = np.average(plddt_np[mask])
            else:
                mean_protein_plddt = np.nan

            if len(ligand_chains) != 0:
                mask = np.isin(chains_np, ligand_chains)
                mean_ligand_plddt = np.average(plddt_np[mask])
            else:
                mean_ligand_plddt = np.nan

            if len(rna_chains) != 0:
                mask = np.isin(chains_np, rna_chains)
                mean_rna_plddt = np.average(plddt_np[mask])
            else:
                mean_rna_plddt = np.nan
            
            if len(dna_chains) != 0:
                mask = np.isin(chains_np, dna_chains)
                mean_dna_plddt = np.average(plddt_np[mask])
            else:
                mean_dna_plddt = np.nan

            out[id]['mean_protein_plddt'] = mean_protein_plddt
            out[id]['mean_ligand_plddt'] = mean_ligand_plddt
            out[id]['mean_rna_plddt'] = mean_rna_plddt
            out[id]['mean_dna_plddt'] = mean_dna_plddt

            # get some final summary data
            out[id]['iptm'] = summary_data['iptm']
            out[id]['ptm'] = summary_data['ptm']

    df = pd.DataFrame().from_dict(out, orient='index').reset_index().rename(columns={"index": "id"})

    print(f'Collected scores for {len(df)} predictions\n')

    if len(failed) != 0:
        print(f'{len(failed)} predictions could not be loaded')
        print(f'Failed predictions:')
        print(failed)

    return df

def restart_af3_predictions(input_directory, output_directory, complete, partially_complete, task_file, num_dirs=1):

    '''
    A function to redistribute af3 jsons to restart predictions.
    
    Inputs:
    input_directory: the top level input directory - will be subdivided later
    output_directory: the output directory, all jobs will point to this directory
    complete: the list of already completed predictions from the input (this will be moved to a new 'complete' directory)
    partially_complete: the list of partially completed precitions (the partial prediction will be deleted)
    task_file: the task file to submitting the new job
    num_dirs: the number of subdirectories to divide the job into
    
    outputs:
    none, but writes the task file
    '''

    completed_dir = os.path.join(input_directory, 'complete')
    make_directory(completed_dir)

    jsons_moved = 0
    jsons_skipped = 0
    partial_predictions_deleted = 0
    partial_predictions_skipped = 0

    jsons = glob.glob(f'{input_directory}/**/*.json', recursive=True)

    for file in jsons:
        file_name = os.path.basename(file).split('.json')[0]
        if file_name in complete:
            new_path = f'{completed_dir}/{file_name}.json'
            try:
                shutil.move(file, new_path)
                jsons_moved += 1
            except FileNotFoundError:
                jsons_skipped += 1
                continue
        elif file_name in partially_complete:
            try:
                shutil.rmtree(f'{output_directory}/{file_name}')
                partial_predictions_deleted += 1
            except FileNotFoundError:
                partial_predictions_skipped += 1
                continue

    print(f'Moved {jsons_moved} json files to {completed_dir}')
    print(f'Skipped {jsons_skipped} files (already moved or not found)')
    print(f'{partial_predictions_deleted} partial predictions have been deleted')
    print(f'{partial_predictions_skipped} partial predictions were skipped (already deleted or not found)')

    task_lines = []

    completed_jsons = glob.glob(f'{completed_dir}/**/*.json', recursive=True)
    all_jsons = glob.glob(f'{input_directory}/**/*.json', recursive=True)

    jsons_remaining = list(set(all_jsons) - set(completed_jsons))

    num_jsons_remaining = len(jsons_remaining)

    print(f'\nFound {num_jsons_remaining} json files remaining in {input_directory}')

    files_moved = 0

    # optional block to split the input directory into several smaller directories
    num_directories = num_dirs

    print(f'Splitting remaining json files into {num_directories} directories')

    for i in range(1, num_directories+1):
        sub_dir = f'{input_directory}/input_{i}'
        
        make_directory(sub_dir)

        start_num = int((i * (num_jsons_remaining/num_directories)) - (num_jsons_remaining/num_directories))
        end_num = int(i * (num_jsons_remaining/num_directories))

        for k in range(start_num, end_num):
            file = os.path.basename(jsons_remaining[k])
            dest = sub_dir+'/'+file
            shutil.move(jsons_remaining[k], dest)
            files_moved += 1

        line = (f'python /work/lpdi/users/dobbelst/tools/alphafold3/run_alphafold.py '
                '--model_dir=\'/work/lpdi/users/dobbelst/tools/alphafold3\' '
                '--db_dir=\'/work/lpdi/users/dobbelst/databases/alphafold3_dbs\' '
                f'--input_dir=\'{sub_dir}\' '
                f'--output_dir=\'{output_directory}\' '
                '--run_data_pipeline=\'false\'\n'
                )
        
        task_lines.append(line)

    with open(task_file, 'a+') as f:
        for line in task_lines:
            f.write(line)

    print(f'\nMoved {files_moved} files into {num_dirs} directories')
    print(f'Wrote to task file {task_file}\n')