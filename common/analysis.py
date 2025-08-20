'''
common analysis functions for protein design

not associated with any particular software

'''

import numpy as np

def n_clashes(xyz_binder, xyz_receptor, d_0=6, r_0=8):
    """
        xyz_binder: xyz coordinates of binder (NumPy array of shape (M, 3))
        xyz_receptor: xyz coordinates of receptor (NumPy array of shape (N, 3))

        Estimates the number of clashes between two lists of atoms using NumPy.
        This function estimates the probability of two atoms clashing based on 
        their Euclidean distance. This enables estimation of number of clashes 
        without explicit side chains
    """

    # Compute pairwise Euclidean distances (distogram) Shape: (M, N)
    dgram = np.sqrt(np.sum((xyz_binder[:, None, :] - xyz_receptor[None, :, :]) ** 2, axis=-1))

    # Estimate number of contacts
    divide_by_r_0 = (dgram - d_0) / r_0
    numerator = np.power(divide_by_r_0, 15)
    denominator = np.power(divide_by_r_0, 20)
    nclashes = (1 - numerator) / (1 - denominator)
    
    return nclashes.sum()

def n_contacts(xyz_binder, xyz_receptor, d_0=6, r_0=8):
    """
        xyz_binder: xyz coordinates of binder (NumPy array of shape (M, 3))
        xyz_receptor: xyz coordinates of receptor (NumPy array of shape (N, 3))

        Estimates the number of contacts between two lists of atoms using NumPy.
        This function estimates the probability of two atoms being in contact based
        on their Euclidean distance. This enables estimation of number of contacts 
        without explicit side chains
    """

    # Compute pairwise Euclidean distances (distogram) Shape: (M, N)
    dgram = np.sqrt(np.sum((xyz_binder[:, None, :] - xyz_receptor[None, :, :]) ** 2, axis=-1))

    # Estimate number of contacts
    divide_by_r_0 = (dgram - d_0) / r_0
    numerator = np.power(divide_by_r_0, 6)
    denominator = np.power(divide_by_r_0, 12)
    ncontacts = (1 - numerator) / (1 - denominator)
    
    return ncontacts.sum()

def rmsd(coords1, coords2):
    """
    simple rmsd calculation from raw atom coordinates
    coordinates should be np.array of shape (L, 3)
    """
    assert coords1.shape == coords2.shape
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff * diff, axis=-1)))

def compute_rog(xyz, min_dist: float=0):
    """
        xyz: np.array, xyz coordinates - shape (N, 3)
        min_dist: float, minimal distance to consider

        Function to estimate the radius of gyration. 
        Approximations:
            - geometric mean instead of center of mass
    """

    centroid = np.mean(xyz, axis=0, keepdims=True) # [1,3]
    dgram = np.sqrt(np.sum((xyz[:, None, :] - centroid[None, :, :]) ** 2, axis=-1)) # [N, 1]
    dgram = np.maximum(min_dist * np.ones_like(dgram), dgram) # [L,1]
    rad_of_gyration = np.sqrt(np.sum(np.square(dgram)) / xyz.shape[0] )# [1]

    return rad_of_gyration

# this function is from the alphafold2 code:
# https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/lddt.py
def lddt(predicted_points,
         true_points,
         true_points_mask,
         cutoff=15.,
         per_residue=False):
  """
        Measure (approximate) lDDT for a batch of coordinates.

        lDDT reference:
        Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
        superposition-free score for comparing protein structures and models using
        distance difference tests. Bioinformatics 29, 2722–2728 (2013).

        lDDT is a measure of the difference between the true distance matrix and the
        distance matrix of the predicted points.  The difference is computed only on
        points closer than cutoff *in the true structure*.

        This function does not compute the exact lDDT value that the original paper
        describes because it does not include terms for physical feasibility
        (e.g. bond length violations). Therefore this is only an approximate
        lDDT score.

        Args:
            predicted_points: (batch, length, 3) array of predicted 3D points
            true_points: (batch, length, 3) array of true 3D points
            true_points_mask: (batch, length, 1) binary-valued float array.  This mask
            should be 1 for points that exist in the true points.
            cutoff: Maximum distance for a pair of points to be included
            per_residue: If true, return score for each residue.  Note that the overall
            lDDT is not exactly the mean of the per_residue lDDT's because some
            residues have more contacts than others.

        Returns:
            An (approximate, see above) lDDT score in the range 0-1.
  """

  assert len(predicted_points.shape) == 3
  assert predicted_points.shape[-1] == 3
  assert true_points_mask.shape[-1] == 1
  assert len(true_points_mask.shape) == 3

  # Compute true and predicted distance matrices.
  dmat_true = np.sqrt(1e-10 + np.sum(
      (true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

  dmat_predicted = np.sqrt(1e-10 + np.sum(
      (predicted_points[:, :, None] -
       predicted_points[:, None, :])**2, axis=-1))

  dists_to_score = (
      (dmat_true < cutoff).astype(np.float32) * true_points_mask *
      np.transpose(true_points_mask, [0, 2, 1]) *
      (1. - np.eye(dmat_true.shape[1]))  # Exclude self-interaction.
  )

  # Shift unscored distances to be far away.
  dist_l1 = np.abs(dmat_true - dmat_predicted)

  # True lDDT uses a number of fixed bins.
  # We ignore the physical plausibility correction to lDDT, though.
  score = 0.25 * ((dist_l1 < 0.5).astype(np.float32) +
                  (dist_l1 < 1.0).astype(np.float32) +
                  (dist_l1 < 2.0).astype(np.float32) +
                  (dist_l1 < 4.0).astype(np.float32))

  # Normalize over the appropriate axes.
  reduce_axes = (-1,) if per_residue else (-2, -1)
  norm = 1. / (1e-10 + np.sum(dists_to_score, axis=reduce_axes))
  score = norm * (1e-10 + np.sum(dists_to_score * score, axis=reduce_axes))

  return score

def euclidian_distance(pos_1: list, pos_2: list) -> float:
    """Returns euclidian distance between two points at [x, y, z]"""
    
    if len(pos_1) != 3 or len(pos_2) != 3:
        raise ValueError("Both points must have 3 coordinates")
    
    pos_1 = np.array(pos_1)
    pos_2 = np.array(pos_2)

    return np.linalg.norm(pos_1 - pos_2, ord=2)


def vector_to_x(origin: list, destination: list):
    """
    origin: [x, y, z] of current position
    destiantion: [x, y, z] of destination
    
    Returns the translation vector to move a point in 3D space to a specified point
    """
    
    return np.subtract(destination, origin)