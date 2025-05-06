import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import MultiModalDataset
from torch.utils.data import DataLoader

def stratified_split_no_batch_overlap(df, label_col, batch_col, test_size, val_size, random_state):
    batch_label_groups = df[[batch_col, label_col]].drop_duplicates()

    # Step 1: Test split stratified by labels
    remaining_groups, test_groups = train_test_split(
        batch_label_groups, 
        stratify=batch_label_groups[label_col], 
        test_size=test_size, 
        random_state=random_state
    )

    # Step 2: Validation split stratified by labels
    train_groups, val_groups = train_test_split(
        remaining_groups, 
        stratify=remaining_groups[label_col], 
        test_size=val_size / (1 - test_size),
        random_state=random_state
    )

    # Assign data based on batch splits
    train_df = df[df[batch_col].isin(train_groups[batch_col])].reset_index(drop=True)
    val_df = df[df[batch_col].isin(val_groups[batch_col])].reset_index(drop=True)
    test_df = df[df[batch_col].isin(test_groups[batch_col])].reset_index(drop=True)

    return train_df, val_df, test_df

def undersample_max_ratio(train_df, label_col, max_ratio=8.0, random_state=42):
    class_counts = train_df[label_col].value_counts()
    min_count = class_counts.min()
    max_allowed = int(min_count * max_ratio)

    sampled_dfs = []
    for label, count in class_counts.items():
        label_df = train_df[train_df[label_col] == label]
        if count > max_allowed:
            label_df = label_df.sample(max_allowed, random_state=random_state)
        sampled_dfs.append(label_df)

    return pd.concat(sampled_dfs).reset_index(drop=True)

def load_and_prepare_data(config):
    smiles_df = pd.read_csv(config["data"]["smiles_csv"])
    img_df = pd.read_csv(config["data"]["image_features_csv"])
    merged_df = pd.concat([smiles_df[["SMILES"]], img_df], axis = 1)

    img_feat_cols = [col for col in img_df.columns if col not in ["batch_id", "encoded_label"]]

    train_df, val_df, test_df = stratified_split_no_batch_overlap(
        merged_df, config["data"]["label_column"], config["data"]["batch_col"],
        config["data"]["test_size"], config["data"]["val_size"], config["data"]["random_state"]
    )

    if config["data"].get("undersample", False):
        train_df = undersample_max_ratio(train_df = train_df, label_col = config["data"]["label_column"],  max_ratio=8.0, random_state = config["data"]["random_state"])

    return train_df, val_df, test_df, img_feat_cols

def get_dataloaders(
    train_df,
    val_df,
    test_df,
    smiles_col,
    img_feat_cols,
    label_col,
    batch_size,
    augment=False,
    noise_std=0.01,
    col_drop_prob=0.05,
    smiles_aug_prob=0.5,
    num_views=1
):
    """
    Create train/val/test dataloaders for MultiModalDataset.
    Only train set will be augmented, optionally with multiple views for minority classes.
    """

    # Determine minority classes (below median count)
    class_counts = train_df[label_col].value_counts()
    median_count = class_counts.median()
    augment_classes = class_counts[class_counts < median_count].index.tolist()

    train_dataset = MultiModalDataset(
        image_features=train_df[img_feat_cols].values,
        smiles_list=train_df[smiles_col].tolist(),
        labels=train_df[label_col].values,
        augment=augment,
        noise_std=noise_std,
        col_drop_prob=col_drop_prob,
        smiles_aug_prob=smiles_aug_prob,
        num_views=num_views,
        augment_classes=augment_classes
    )
    val_dataset = MultiModalDataset(
        image_features=val_df[img_feat_cols].values,
        smiles_list=val_df[smiles_col].tolist(),
        labels=val_df[label_col].values,
        augment=False
    )
    test_dataset = MultiModalDataset(
        image_features=test_df[img_feat_cols].values,
        smiles_list=test_df[smiles_col].tolist(),
        labels=test_df[label_col].values,
        augment=False
    )
    
    from collections import Counter
    label_counts = Counter(train_dataset.labels)
    print("\n[INFO] Training class distribution (after augmentation):")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} samples")

    # Total number of examples
    print(f"[INFO] Total training samples (after augmentation): {len(train_dataset)}")

    # Also print number of unique compounds per class
    smiles_per_class = {label: set() for label in label_counts}
    for smile, label in zip(train_dataset.smiles_list, train_dataset.labels):
        smiles_per_class[label].add(smile)

    print("[INFO] Unique SMILES (treatments) per class:")
    for label, smile_set in smiles_per_class.items():
        print(f"  Class {label}: {len(smile_set)} unique SMILES")
    print()


    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )