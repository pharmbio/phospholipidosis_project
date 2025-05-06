import torch
import random
import numpy as np
from torch.utils.data import Dataset
# For SMILES randomization
from rdkit import Chem

def randomize_smiles(original_smiles: str, verbose: bool = False) -> str:
    """
    Attempts to generate a randomized version of a SMILES string by 
    shuffling atom indices and reconstructing the molecule. Falls back 
    to the original SMILES if any error occurs.

    Parameters:
    - original_smiles (str): The input SMILES string.
    - verbose (bool): If True, prints debug information.

    Returns:
    - str: A randomized SMILES string or the original if something fails.
    """
    try:
        mol = Chem.MolFromSmiles(original_smiles)
        if mol is None:
            if verbose:
                print(f"[Warning] RDKit could not parse SMILES: {original_smiles}")
            return original_smiles
        
        num_atoms = mol.GetNumAtoms()
        if num_atoms < 2:
            if verbose:
                print(f"[Info] Not enough atoms to shuffle in: {original_smiles}")
            return original_smiles

        atom_indices = list(range(num_atoms))
        random.shuffle(atom_indices)
        new_mol = Chem.RenumberAtoms(mol, atom_indices)

        new_smiles = Chem.MolToSmiles(new_mol, canonical=True)
        if not new_smiles:
            if verbose:
                print(f"[Warning] Failed to convert shuffled mol back to SMILES: {original_smiles}")
            return original_smiles

        return new_smiles

    except Exception as e:
        if verbose:
            print(f"[Error] Exception during SMILES randomization: {original_smiles}")
            print(f"        {type(e).__name__}: {e}")
        return original_smiles

def augment_tabular_features(
    features: np.ndarray,
    noise_std: float = 0.01,
    col_drop_prob: float = 0.05
) -> np.ndarray:
    """
    Adds mild Gaussian noise and randomly zeroes out some columns.
    features: shape (num_features,)
    noise_std: fraction of row std dev to use for noise scale
    col_drop_prob: probability that a given feature is zeroed out
    """
    feats = features.copy()
    if feats.size == 0:
        return feats  # no features, return as is

    row_std = feats.std() if feats.std() > 1e-8 else 1e-8
    # 1) Add Gaussian noise
    noise = np.random.normal(0, noise_std * row_std, size=feats.shape)
    feats = feats + noise
    # (Optionally clamp if your data should remain in [0,1]):
    # feats = np.clip(feats, 0.0, 1.0)

    # 2) Column dropout
    mask = (np.random.rand(feats.shape[0]) > col_drop_prob).astype(float)
    feats = feats * mask

    return feats


class MultiModalDataset(Dataset):
    def __init__(
        self,
        image_features,
        smiles_list,
        labels,
        augment=False,
        noise_std=0.01,
        col_drop_prob=0.05,
        smiles_aug_prob=0.5,
        num_views=1,
        augment_classes=None
    ):
        super().__init__()
        self.original_image_features = torch.tensor(image_features, dtype=torch.float32)
        self.original_smiles_list = smiles_list
        self.original_labels = torch.tensor(labels, dtype=torch.long)

        self.augment = augment
        self.noise_std = noise_std
        self.col_drop_prob = col_drop_prob
        self.smiles_aug_prob = smiles_aug_prob
        self.num_views = num_views
        self.augment_classes = set(augment_classes) if augment_classes is not None else set()

        self.image_features = []
        self.smiles_list = []
        self.labels = []

        for i in range(len(self.original_labels)):
            label = int(self.original_labels[i].item())
            self.image_features.append(self.original_image_features[i])
            self.smiles_list.append(self.original_smiles_list[i])
            self.labels.append(label)

            if self.augment and label in self.augment_classes:
                feats_np = self.original_image_features[i].numpy()
                smi = self.original_smiles_list[i]
                for _ in range(self.num_views):
                    f_aug = augment_tabular_features(feats_np, self.noise_std, self.col_drop_prob)
                    s_aug = randomize_smiles(smi) if random.random() < self.smiles_aug_prob else smi
                    self.image_features.append(torch.tensor(f_aug, dtype=torch.float32))
                    self.smiles_list.append(s_aug)
                    self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "img_feat": self.image_features[idx],
            "smiles": self.smiles_list[idx],
            "label": self.labels[idx]
        }

