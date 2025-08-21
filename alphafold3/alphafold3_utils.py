# general imports
import os
import shutil
import glob
import pandas as pd
import json
import numpy as np

def check_af3_completion(input_directory, output_directory):
    '''
    check which jsons have been completed

    returns complete, partially_complete, and incomplete lists
    '''

    jsons = glob.glob(f'{input_directory}/**/*.json', recursive=True)
    input_job_names = [os.path.basename(x).split('.json')[0] for x in jsons]

    num_jsons = len(jsons)

    complete = []
    partially_complete = []

    # iterate over all directories
    for folder in os.listdir(output_directory):
        job_name = os.path.basename(folder)
        # prediction complete if ranking was done
        if f'{job_name}_ranking_scores.csv' in os.listdir(os.path.join(output_directory, folder)):
            complete.append(job_name)
        elif f'{job_name}_ranking_scores.csv' not in os.listdir(os.path.join(output_directory, folder)):
            partially_complete.append(job_name)
    
    incomplete = [x for x in input_job_names if x not in complete and x not in partially_complete]

    num_completed = len(complete)
    num_partially_completed = len(partially_complete)
    num_incomplete = len(incomplete)

    print(f'Found {num_jsons} json files in input directory. \n{num_completed} predictions have been completed. \n{num_partially_completed} predictions are partially complete. \nThere are {num_incomplete} incomplete predictions.')

    return complete, partially_complete, incomplete

def collect_af3_scores(scores_directory):

    '''
    A function to collect all af3 scores from the output directory

    returns a pandas dataframe
    '''

    out = {}

    for element in os.listdir(scores_directory):
        directory = os.path.join(scores_directory, element)

        if os.path.isdir(directory):

            # get the 3 json files that we need
            data = os.path.join(directory, element + '_data.json')
            summary = os.path.join(directory, element + '_summary_confidences.json')
            details = os.path.join(directory, element + '_confidences.json')

            with open(data, 'r') as fobj:
                meta_data = json.load(fobj)
            with open(summary, 'r') as fobj:
                summary_data = json.load(fobj)
            with open(details, 'r') as fobj:
                detail_data = json.load(fobj)

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

    return df