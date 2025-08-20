import jax
import numpy as np

from alphafold.relax import relax
from alphafold.common import protein as p_cf

from colabdesign.af.alphafold.common import protein
from colabdesign.shared.protein import renum_pdb_str
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import *
from colabdesign.af.loss import _get_pw_loss

MODRES = {'MSE':'MET','MLY':'LYS','FME':'MET','HYP':'PRO',
          'TPO':'THR','CSO':'CYS','SEP':'SER','M3L':'LYS',
          'HSK':'HIS','SAC':'SER','PCA':'GLU','DAL':'ALA',
          'CME':'CYS','CSD':'CYS','OCS':'CYS','DPR':'PRO',
          'B3K':'LYS','ALY':'LYS','YCM':'CYS','MLZ':'LYS',
          '4BF':'TYR','KCX':'LYS','B3E':'GLU','B3D':'ASP',
          'HZP':'PRO','CSX':'CYS','BAL':'ALA','HIC':'HIS',
          'DBZ':'ALA','DCY':'CYS','DVA':'VAL','NLE':'LEU',
          'SMC':'CYS','AGM':'ARG','B3A':'ALA','DAS':'ASP',
          'DLY':'LYS','DSN':'SER','DTH':'THR','GL3':'GLY',
          'HY3':'PRO','LLP':'LYS','MGN':'GLN','MHS':'HIS',
          'TRQ':'TRP','B3Y':'TYR','PHI':'PHE','PTR':'TYR',
          'TYS':'TYR','IAS':'ASP','GPL':'LYS','KYN':'TRP',
          'CSD':'CYS','SEC':'CYS'}

aa_order = residue_constants.restype_order
order_aa = {b:a for a,b in aa_order.items()}

def pdb_to_string(pdb_file, chains=None, models=[1]):
  '''read pdb file and return as string'''

  if chains is not None:
    if "," in chains: chains = chains.split(",")
    if not isinstance(chains,list): chains = [chains]
  if models is not None:
    if not isinstance(models,list): models = [models]

  modres = {**MODRES}
  lines = []
  seen = []
  model = 1
  for line in open(pdb_file,"rb"):
    line = line.decode("utf-8","ignore").rstrip()
    if line[:5] == "MODEL":
      model = int(line[5:])
    if models is None or model in models:
      if line[:6] == "MODRES":
        k = line[12:15]
        v = line[24:27]
        if k not in modres and v in residue_constants.restype_3to1:
          modres[k] = v
      if line[:6] == "HETATM":
        k = line[17:20]
        if k in modres:
          line = "ATOM  "+line[6:17]+modres[k]+line[20:]
      if line[:4] == "ATOM":
        chain = line[21:22]
        if chains is None or chain in chains:
          atom = line[12:12+4].strip()
          resi = line[17:17+3]
          resn = line[22:22+5].strip()
          if resn[-1].isalpha(): # alternative atom
            resn = resn[:-1]
            line = line[:26]+" "+line[27:]
          key = f"{model}_{chain}_{resn}_{resi}_{atom}"
          if key not in seen: # skip alternative placements
            lines.append(line)
            seen.append(key)
      if line[:5] == "MODEL" or line[:3] == "TER" or line[:6] == "ENDMDL":
        lines.append(line)
  return "\n".join(lines)

def rank_array(input_array):
    # numpy.argsort returns the indices that would sort an array.
    # We convert it to a python list before returning
    return list(np.argsort(input_array))

def rank_and_write_pdb(af_model, name, write_all=False, renum_pdb = True, predict=False):
    
    def to_pdb_str(x):
      '''Convert pdb file to string'''
      p_str = protein.to_pdb(protein.Protein(**x))
      p_str = "\n".join(p_str.splitlines()[1:-2])
      if renum_pdb: p_str = renum_pdb_str(p_str, af_model._lengths)
      return p_str

    if predict==True:
      # sorting the models based on plddt
      # reverse since np.argsort sorts low to high, but we need high to low sorting
      ranking = rank_array(np.mean(af_model.aux['all']['plddt'],-1))
      ranking.reverse()
      aux = af_model.aux
      aux = aux["all"]

    elif predict==False:
      ranking = rank_array(af_model.aux['all']['loss'])
      aux = af_model._tmp["best"]["aux"]
      aux = aux["all"]

    if write_all != True:
      ranking = [ranking[0]]

    p = {k:aux[k] for k in ["aatype","residue_index","atom_positions","atom_mask"]}
    p["b_factors"] = 100 * p["atom_mask"] * aux["plddt"][...,None]

    m=1
    
    pdbs_out = []
    
    for n in ranking:
      p_str = ""
      p_str += to_pdb_str(jax.tree_map(lambda x:x[n],p))
      p_str += "END\n"

      if predict==True:
        out_file = name + '_model_{n}_rank_{m}.pdb'.format(n=n, m=m)
        print(f'...writing prediction {out_file}')
        
        with open(out_file, 'w') as f:
            f.write(p_str)
        pdbs_out.append(out_file)
        m+=1
      
      elif predict==False:
        for n in ranking:
          p_str = ""
          p_str += to_pdb_str(jax.tree_map(lambda x:x[n],p))
          p_str += "END\n"
      
          with open(name + '_backprop_model.pdb', 'w') as f:
              f.write(p_str)
          m+=1
    
    if predict==True:
      return pdbs_out, ranking[0]  