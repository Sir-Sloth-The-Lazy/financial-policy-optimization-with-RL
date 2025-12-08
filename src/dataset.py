import torch
from torch.utils.data import Dataset
import numpy as np
import os

class LoanDataset(Dataset):
    def __init__(self, data_dir, mode='train', split_ratio=(0.7, 0.15, 0.15), seed=42):
        """
        Args:
            data_dir (str): Path to directory containing X.npy and y.npy
            mode (str): 'train', 'val', or 'test'
            split_ratio (tuple): Ratios for train, val, test splits
            seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.mode = mode
        
        # Load data
        X = np.load(os.path.join(data_dir, 'X.npy')).astype(np.float32)
        y = np.load(os.path.join(data_dir, 'y.npy')).astype(np.float32) # BCE needs float target
        # If using CrossEntropy, y should be long
        
        # Verify sizes
        assert len(X) == len(y), "X and y must have same length"
        
        # Create splits indices
        total_size = len(X)
        indices = np.arange(total_size)
        
        # Deterministic split
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_end = int(total_size * split_ratio[0])
        val_end = int(total_size * (split_ratio[0] + split_ratio[1]))
        
        if mode == 'train':
            self.indices = indices[:train_end]
        elif mode == 'val':
            self.indices = indices[train_end:val_end]
        else: # test
            self.indices = indices[val_end:]
            
        self.X = torch.from_numpy(X[self.indices])
        self.y = torch.from_numpy(y[self.indices]).unsqueeze(1) # [batch_size, 1] for BCELoss
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    # Test dataset
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    
    if os.path.exists(DATA_DIR):
        ds = LoanDataset(DATA_DIR, mode='train')
        print(f"Train Dataset size: {len(ds)}")
        x, y = ds[0]
        print(f"Sample X shape: {x.shape}, y shape: {y.shape}")
        print(f"Sample y value: {y}")
