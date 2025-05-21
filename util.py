import torch
import torch.nn as nn
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MatrixDataset(Dataset):
    def __init__(self, json_file, target_key='area_avg', fill_strategy='average', missing_strategy='fill', tokenize=False, token_threshold=5):
        """
        Args:
            json_file (string): Path to the json file with matrix data
            target_key (string): Key in json file that contains the target value
            fill_strategy (string): Strategy to fill missing values ('average', 'minimum', or 'median')
            missing_strategy (string): 'fill' to fill missing values, 'drop' to remove samples with missing target
            tokenize (bool): Whether to tokenize E and S values
            token_threshold (int): Threshold for tokenization (values > threshold become 1, between 0 and threshold become 0)
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.target_key = target_key
        self.tokenize = tokenize
        self.token_threshold = token_threshold
        
        # Handle missing values
        if missing_strategy == 'drop':
            original_len = len(self.data)
            self.data = [item for item in self.data if not np.isnan(item[target_key])]
            dropped = original_len - len(self.data)
            if dropped > 0:
                print(f"Dropped {dropped} samples with missing {target_key}.")
        elif missing_strategy == 'fill':
            valid_data = [item for item in self.data if not np.isnan(item[target_key])]
            if len(valid_data) < len(self.data):
                print(f"Warning: Found {len(self.data) - len(valid_data)} missing values in {target_key}")
                valid_values = [item[target_key] for item in valid_data]
                if fill_strategy == 'average':
                    fill_value = np.mean(valid_values)
                elif fill_strategy == 'minimum':
                    fill_value = np.min(valid_values)
                elif fill_strategy == 'median':
                    fill_value = np.median(valid_values)
                else:
                    raise ValueError(f"Invalid fill_strategy: {fill_strategy}. Must be one of: average, minimum, median")
                for item in self.data:
                    if np.isnan(item[target_key]):
                        item[target_key] = fill_value
                print(f"Filled missing values with {fill_strategy}: {fill_value}")
        else:
            raise ValueError(f"Invalid missing_strategy: {missing_strategy}. Must be 'fill' or 'drop'.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        matrix = np.array(item['input_matrix'])
        
        if self.tokenize:
            # Create a copy of the matrix
            tokenized_matrix = matrix.copy()
            # Tokenize E values (row 1)
            tokenized_matrix[1] = np.where(matrix[1] == 0, -1,
                                         np.where(matrix[1] > self.token_threshold, 1, 0))
            # Tokenize S values (row 2)
            tokenized_matrix[2] = np.where(matrix[2] == 0, -1,
                                         np.where(matrix[2] > self.token_threshold, 1, 0))
            matrix = tokenized_matrix
        
        matrix = torch.FloatTensor(matrix)
        target = torch.FloatTensor([item[self.target_key]])
        return matrix, target

class Simple1DCNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        """
        Args:
            input_size (int): Size of input matrix
            hidden_size (int): Size of hidden layers
            output_size (int): Size of output (default=1 for regression)
        """
        super(Simple1DCNN, self).__init__()
        
        # 1D CNN layer - accepting 3 input channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),  # Input: [batch, 3, 20]
            nn.ReLU(),
            nn.MaxPool1d(2)  # Output: [batch, 16, 10]
        )
        
        # Calculate the size after CNN and pooling
        cnn_output_size = input_size // 2
        
        # Calculate the flattened size
        self.flattened_size = 16 * 10
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, hidden_size)  # 160 -> 64
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)  # 64 -> 1
        
    def forward(self, x):
        # Input shape: [batch_size, 3, input_size]
        # Reshape if needed
        if len(x.shape) == 2:
            x = x.view(x.size(0), 3, -1)
        
        # Apply CNN
        x = self.conv1(x)  # Shape: [batch_size, 16, input_size/2]
        
        # Flatten
        x = x.view(x.size(0), -1)  # Shape: [batch_size, 16 * (input_size/2)]
        
        # Apply fully connected layers
        x = self.fc1(x)  # Shape: [batch_size, hidden_size]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Shape: [batch_size, output_size]
        
        return x

def get_data_loader(json_file, batch_size=32, target_key='area_avg', fill_strategy='average', 
                   missing_strategy='fill', tokenize=False, token_threshold=5):
    """
    Create and return a DataLoader for the matrix dataset
    
    Args:
        json_file (string): Path to the json file
        batch_size (int): Batch size for training
        target_key (string): Key in json file that contains the target value
        fill_strategy (string): Strategy to fill missing values ('average', 'minimum', or 'median')
        missing_strategy (string): 'fill' to fill missing values, 'drop' to remove samples with missing target
        tokenize (bool): Whether to tokenize E and S values
        token_threshold (int): Threshold for tokenization (values > threshold become 1, between 0 and threshold become 0)
    
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = MatrixDataset(json_file, target_key, fill_strategy, missing_strategy, tokenize, token_threshold)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
