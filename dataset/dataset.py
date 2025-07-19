import numpy as np
from torch.utils.data import Dataset

def load_feat(feat_path, feat_dim=256):
    if '.npy' in feat_path:
        feat = np.load(feat_path).astype(np.float32)
    else:
        feat = np.fromfile(feat_path, dtype=np.float32)
        feat = feat.reshape(-1, feat_dim)
    return feat

def load_labels(label_path):
    with open(label_path, 'r') as f:
        gt_labels = f.readlines()
        f.close()
    gt_labels = np.array([k.strip() for k in gt_labels], dtype=np.int64)
    return gt_labels

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        assert features.shape[0] == labels.shape[0]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]