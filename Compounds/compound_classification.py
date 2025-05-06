#!/usr/bin/env python
"""
SMILES Embedding Extraction and Model Evaluation Pipeline (using Click)

This script:
  - Loads specs and PLP CSV files and merges them.
  - Extracts unique SMILES strings.
  - Extracts embeddings using a chosen method:
       - Transformer-based ("molt5" or "chemberta") with various pooling options, or
       - Morgan fingerprints ("morgan").
  - Generates labels from the PLP information.
  - Creates a feature DataFrame (with an added column for the embedding model name)
    and an AnnData object.
  - Performs PCA/UMAP visualization.
  - Splits the data (stratified with no batch overlap) and evaluates classifiers.
  - Saves evaluation results, predictions, and the feature DataFrame (including the embedding model name) to CSV files.

Usage example:
    python smiles_pipeline_click.py \
        --specs_csv "/path/to/specs_smiles.csv" \
        --plp_csv "/path/to/plp_data.csv" \
        --embedding_model chemberta \
        --pooling mean \
        --batch_size 8 \
        --label_col "label" \
        --batch_col "Batch_nr" \
        --obsm_key X_pca \
        --undersample
"""

import click
import os
import pandas as pd
import numpy as np
import torch
import tqdm
import scanpy as sc
import anndata as ad
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Data Loading Functions
# --------------------------
def load_data(specs_csv, plp_csv):
    specs_smiles = pd.read_csv(specs_csv)
    plp_data = pd.read_csv(plp_csv, sep=";")
    specs_smiles_plp = specs_smiles.merge(plp_data, left_on="Batch nr", right_on="Batch_nr", how="left")
    specs_smiles_plp = specs_smiles_plp.dropna(subset=["%Induction"]).reset_index(drop=True)
    return specs_smiles_plp

def get_unique_smiles(df):
    return df["SMILES"].drop_duplicates().tolist()

# --------------------------
# Morgan Fingerprint Function
# --------------------------
def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    """Convert a SMILES string to a Morgan fingerprint (bit vector)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# --------------------------
# Embedding Extraction Functions
# --------------------------
def extract_embeddings(unique_smiles, embedding_model, batch_size=8, pooling="cls", device=None):
    """
    Extract embeddings from a list of SMILES strings using the selected method.
    
    Args:
        unique_smiles: list of SMILES strings.
        embedding_model: "molt5", "chemberta", or "morgan"
        batch_size: number of SMILES processed per batch (only for transformer-based methods)
        pooling: one of "cls", "mean", "max", "attention", "last" (only used for transformer-based methods)
        device: torch.device; if None, it will be auto-selected.
        
    Returns:
        A dictionary mapping each SMILES string to its embedding (NumPy array).
    """
    results = {}
    if embedding_model.lower() == "morgan":
        for smile in tqdm.tqdm(unique_smiles, desc="Extracting Morgan fingerprints"):
            results[smile] = smiles_to_morgan(smile)
        return results

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if embedding_model.lower() == "molt5":
        model_name = "laituan245/molt5-large-smiles2caption"
        tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif embedding_model.lower() == "chemberta":
        model_name = "DeepChem/ChemBERTa-77M-MTR"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    else:
        raise ValueError("Unsupported embedding model. Choose 'molt5', 'chemberta', or 'morgan'.")
    
    model.to(device)
    unique_smiles = list(set(unique_smiles))
    
    for i in tqdm.tqdm(range(0, len(unique_smiles), batch_size), desc="Extracting embeddings"):
        batch_smiles = unique_smiles[i : i + batch_size]
        if embedding_model.lower() == "molt5":
            inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            with torch.no_grad():
                outputs = model.encoder(inputs)
        elif embedding_model.lower() == "chemberta":
            inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
        
        if pooling.lower() == "cls":
            pooled = hidden_states[:, 0, :]
        elif pooling.lower() == "mean":
            pooled = hidden_states.mean(dim=1)
        elif pooling.lower() == "max":
            pooled = hidden_states.max(dim=1)[0]
        elif pooling.lower() == "last":
            pooled = hidden_states[:, -1, :]
        elif pooling.lower() == "attention":
            attention_scores = F.softmax(hidden_states.mean(dim=-1), dim=-1)
            pooled = torch.einsum("bs,bsd->bd", attention_scores, hidden_states)
        else:
            raise ValueError("Unsupported pooling method.")
        
        pooled = pooled.cpu().numpy()
        for j, smile in enumerate(batch_smiles):
            results[smile] = pooled[j]
        
        del inputs, outputs, hidden_states, pooled
        torch.cuda.empty_cache()
    
    return results

# --------------------------
# Label Generation Function
# --------------------------
def generate_labels(feature_df):
    """
    Generate label columns based on PLP-related information.
    This function adds/updates two columns:
      - "label": Default is "no_plp", then set to "high_tox_plp" or "low_plp" based on conditions.
      - "label_refined": Further refined labels based on specific PLP columns.
    
    Modify the column names and conditions as needed to match your data.
    """
    # Identify columns related to high toxicity or nontoxicity and low toxicity
    high_cols = [col for col in feature_df.columns if "High" in col or "nontoxic" in col]
    low_cols = [col for col in feature_df.columns if "Low" in col]
    
    feature_df["label"] = "no_plp"
    feature_df["label_refined"] = "no_plp"
    
    # If any high columns sum > 0, label as high toxicity PLP
    feature_df.loc[feature_df[high_cols].sum(axis=1) > 0, "label"] = "high_tox_plp"
    
    # Refine label based on specific columns if available
    if "High_PLP_10uM_>80" in feature_df.columns:
        feature_df.loc[feature_df["High_PLP_10uM_>80"] == 1, "label_refined"] = "high_tox_plp"
    if "nontoxic_10uM_>80" in feature_df.columns:
        feature_df.loc[feature_df["nontoxic_10uM_>80"] == 1, "label_refined"] = "non_tox_plp"
    
    # If any low columns sum > 0, label as low PLP
    feature_df.loc[feature_df[low_cols].sum(axis=1) > 0, "label"] = "low_plp"
    feature_df.loc[feature_df[low_cols].sum(axis=1) > 0, "label_refined"] = "low_plp"
    
    return feature_df

# --------------------------
# Feature DataFrame and AnnData Creation
# --------------------------
def create_feature_dataframe(embeddings_dict, specs_df, embedding_model_name):
    """Create a feature DataFrame from embeddings and merge with specs data."""
    feature_df = pd.DataFrame(list(embeddings_dict.items()), columns=["SMILES", "embedding"])
    emb_matrix = np.vstack(feature_df["embedding"].values)
    emb_df = pd.DataFrame(emb_matrix, index=feature_df["SMILES"])
    emb_df.reset_index(inplace=True)
    emb_df.rename(columns={'index': 'SMILES'}, inplace=True)
    emb_df.columns = ['SMILES'] + [f"Feature {i+1}" for i in range(emb_df.shape[1] - 1)]
    emb_df["embedding_model"] = embedding_model_name  # Add embedding model name
    merged_df = pd.merge(emb_df, specs_df, on="SMILES", how="left")
    # Generate labels based on PLP information
    merged_df = generate_labels(merged_df)
    return merged_df

def create_anndata(feature_df, feature_prefix="Feature"):
    """Create an AnnData object from the feature DataFrame."""
    features = [col for col in feature_df.columns if col.startswith(feature_prefix)]
    meta_feats = [col for col in feature_df.columns if col not in features]
    adata = ad.AnnData(X=feature_df[features].values, obs=feature_df[meta_feats])
    return adata

# --------------------------
# Stratified Split Function
# --------------------------
def stratified_split_no_batch_overlap(adata, label_col="label", batch_col="batch_id", test_size=0.2, val_size=0.1, random_seed=42, obsm_key=None):
    np.random.seed(random_seed)
    obs_df = adata.obs.copy()
    if obsm_key != "None":
        if obsm_key not in adata.obsm.keys():
            raise ValueError(f"obsm_key '{obsm_key}' not found. Available: {list(adata.obsm.keys())}")
        feature_matrix = adata.obsm[obsm_key]
        feature_names = [f"Feature_{i}" for i in range(feature_matrix.shape[1])]
    else:
        feature_matrix = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        feature_names = [f"Feature_{i}" for i in range(adata.X.shape[1])]
    feature_df = pd.DataFrame(feature_matrix, columns=feature_names, index=adata.obs.index)
    df = pd.concat([obs_df, feature_df], axis=1)
    if batch_col not in df.columns:
        raise ValueError(f"Batch column '{batch_col}' not found.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    batch_label_groups = df[[batch_col, label_col]].drop_duplicates()
    train_groups, test_groups = train_test_split(
        batch_label_groups, stratify=batch_label_groups[label_col], test_size=test_size, random_state=random_seed
    )
    val_groups, train_groups = train_test_split(
        train_groups, stratify=train_groups[label_col], test_size=1 - (val_size / (1 - test_size)), random_state=random_seed
    )
    train_df = df[df[batch_col].isin(train_groups[batch_col])].reset_index(drop=True)
    val_df = df[df[batch_col].isin(val_groups[batch_col])].reset_index(drop=True)
    test_df = df[df[batch_col].isin(test_groups[batch_col])].reset_index(drop=True)
    return train_df, val_df, test_df



# --------------------------
# Model Evaluation Functions
# --------------------------
def evaluate_models(adata, obsm_keys=["X"], label_col="label", batch_col="batch_id", undersample=False):
    results = []
    predictions = []
    for split_seed in range(5):
        for obsm_key in obsm_keys:
            click.echo(f"Evaluating for split seed {split_seed} and features {obsm_key}")
            from sklearn.model_selection import train_test_split  # local import

            train, val, test = stratified_split_no_batch_overlap(
                adata, label_col=label_col, batch_col=batch_col, random_seed=split_seed, obsm_key=obsm_key
            )
            train = pd.concat([train, val], axis=0)
            train_X = train[[col for col in train.columns if col.startswith("Feature")]]
            train_y = train[label_col]
            test_X = test[[col for col in test.columns if col.startswith("Feature")]]
            test_y = test[label_col]
            label_encoder = LabelEncoder()
            train_y_encoded = label_encoder.fit_transform(train_y)
            test_y_encoded = label_encoder.transform(test_y)
            if undersample:
                class_counts = np.bincount(train_y_encoded)
                min_class_size = max(3 * np.min(class_counts), np.min(class_counts) * 2)
                train_balanced_indices = np.hstack([
                    np.random.choice(np.where(train_y_encoded == cls)[0], min(min_class_size, count), replace=False)
                    for cls, count in enumerate(class_counts)
                ])
                train_X = train_X.iloc[train_balanced_indices]
                train_y_encoded = train_y_encoded[train_balanced_indices]
            from xgboost import XGBClassifier  # local import
            models = {
                'RandomForest': RandomForestClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42),
                'TabPFN': TabPFNClassifier(ignore_pretraining_limits=True, random_state=42),
            }
            for name, clf in models.items():
                click.echo(f"Training {name} on {obsm_key}")
                clf.fit(train_X, train_y_encoded)
                test_preds = clf.predict(test_X)
                if hasattr(clf, "predict_proba") and len(label_encoder.classes_) == 2:
                    test_probs = clf.predict_proba(test_X)[:, 1]
                    test_roc = roc_auc_score(test_y_encoded, test_probs, multi_class="ovr")
                else:
                    test_roc = None
                test_f1 = f1_score(test_y_encoded, test_preds, average='weighted')
                test_precision = precision_score(test_y_encoded, test_preds, average='weighted')
                test_recall = recall_score(test_y_encoded, test_preds, average='weighted')
                results.append({
                    "Model": name,
                    "Features": obsm_key,
                    "Seed": split_seed,
                    "ROC AUC": test_roc,
                    "Precision": test_precision,
                    "Recall": test_recall,
                    "F1 Score": test_f1,
                })
                preds_df = pd.DataFrame({
                    "True Label": test_y,
                    "Predicted Label": label_encoder.inverse_transform(test_preds),
                    "Model": name,
                    "Features": obsm_key,
                    "Seed": split_seed
                })
                predictions.append(preds_df)
    return pd.DataFrame(results), pd.concat(predictions)

# --------------------------
# Main Pipeline Function using Click
# --------------------------
@click.command()
@click.option("--specs_csv", type=str, required=True, help="Path to the specs SMILES CSV file")
@click.option("--plp_csv", type=str, required=True, help="Path to the PLP data CSV file")
@click.option("--embedding_model", type=click.Choice(["molt5", "chemberta", "morgan"]), default="chemberta",
              help="Embedding model to use: 'molt5', 'chemberta', or 'morgan'")
@click.option("--pooling", type=click.Choice(["cls", "mean", "max", "attention", "last"]), default="cls",
              help="Pooling method for transformer-based embedding extraction")
@click.option("--batch_size", type=int, default=8, help="Batch size for embedding extraction (for transformer models)")
@click.option("--label_col", type=str, default="label", help="Label column name for classification")
@click.option("--batch_col", type=str, default="Batch_nr", help="Batch column name for stratified split")
@click.option("--obsm_key", type=str, default="X_pca", help="AnnData obsm key for model evaluation")
@click.option("--undersample", is_flag=True, help="Apply undersampling for training")
def main(specs_csv, plp_csv, embedding_model, pooling, batch_size, label_col, batch_col, obsm_key, undersample):
    # Ensure SCIPY_ARRAY_API is set
    os.environ["SCIPY_ARRAY_API"] = "1"
    
    click.echo("Loading data...")
    specs_df = load_data(specs_csv, plp_csv)
    unique_smiles = get_unique_smiles(specs_df)
    
    click.echo(f"Extracting embeddings using {embedding_model}...")
    embeddings_dict = extract_embeddings(unique_smiles, embedding_model, batch_size=batch_size, pooling=pooling)
    
    click.echo("Creating feature DataFrame...")
    feature_df = create_feature_dataframe(embeddings_dict, specs_df, embedding_model)
    
    click.echo("Creating AnnData object...")
    adata = create_anndata(feature_df)
    
    click.echo("Performing PCA and UMAP visualization...")
    sc.tl.pca(adata, n_comps=150)

    click.echo("Evaluating models...")
    results_df, predictions_df = evaluate_models(adata, obsm_keys=[obsm_key], label_col=label_col, batch_col=batch_col, undersample=undersample)
    
    embedding_tag = embedding_model.lower()
    click.echo("Saving results...")
    results_df.to_csv(f"evaluation_results_{embedding_tag}_{pooling}_{obsm_key}.csv", index=False)
    predictions_df.to_csv(f"predictions_{embedding_tag}_{pooling}_{obsm_key}.csv", index=False)
    feature_df.to_csv(f"feature_dataframe_{embedding_tag}_{pooling}_{obsm_key}.csv", index=False)
    
    click.echo("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
