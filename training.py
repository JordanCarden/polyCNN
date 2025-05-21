import torch
import torch.nn as nn
import torch.optim as optim
from util import Simple1DCNN, get_data_loader, MatrixDataset
import argparse
import json
import os
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np

# Default configurations
DEFAULT_CONFIG = {
    'train_data': 'Data/processed_data.json',  # Default training data path
    'target_key': 'rg_avg',              # Key in JSON for target values
    'batch_size': 16,                         # Batch size for training
    'epochs': 150,                            # Number of training epochs
    'learning_rate': 0.005,                   # Learning rate
    'hidden_size': 64,                        # Size of hidden layers
    'log_interval': 20,                       # Logging interval
    'train_test_split': 0.7,                  # Train-test split ratio
    'fill_strategy': 'average',               # Strategy to fill missing values ('average', 'minimum', 'median')
    'missing_strategy': 'drop',               # Strategy for missing data: fill or drop
    'tokenize': True,                        # Whether to tokenize E and S values
    'token_threshold':2,                     # Threshold for tokenization
}

def plot_unity_plot(y_true, y_pred, title, save_path):
    """Create and save unity plot with R² and MAE metrics"""
    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{title}\nR² = {r2:.3f}, MAE = {mae:.3f}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return r2, mae

def train_model(args):
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('Output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = MatrixDataset(args.train_data, args.target_key, args.fill_strategy, args.missing_strategy)
    
    # Split dataset
    train_size = int(args.train_test_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get input size from first batch
    sample_matrix, _ = next(iter(train_loader))
    input_size = sample_matrix.shape[1]
    
    # Initialize model
    model = Simple1DCNN(
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=1
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training history
    train_losses = []
    test_losses = []
    
    # For collecting predictions/targets
    all_train_predictions = []
    all_train_targets = []
    all_test_predictions = []
    all_test_targets = []
    
    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        epoch_train_preds = []
        epoch_train_tgts = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            epoch_train_preds.extend(output.detach().cpu().numpy())
            epoch_train_tgts.extend(target.detach().cpu().numpy())
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # Save last epoch's predictions for train set
        if epoch == args.epochs - 1:
            all_train_predictions = epoch_train_preds
            all_train_targets = epoch_train_tgts
        # Testing phase
        model.eval()
        total_test_loss = 0
        epoch_test_preds = []
        epoch_test_tgts = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_test_loss += loss.item()
                epoch_test_preds.extend(output.cpu().numpy())
                epoch_test_tgts.extend(target.cpu().numpy())
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        # Save last epoch's predictions for test set
        if epoch == args.epochs - 1:
            all_test_predictions = epoch_test_preds
            all_test_targets = epoch_test_tgts
        print(f'Epoch {epoch} - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    # Save model configuration
    train_r2, train_mae = plot_unity_plot(
        np.array(all_train_targets).flatten(),
        np.array(all_train_predictions).flatten(),
        'Train Set Unity Plot',
        os.path.join(output_dir, 'train_unity_plot.png')
    )
    test_r2, test_mae = plot_unity_plot(
        np.array(all_test_targets).flatten(),
        np.array(all_test_predictions).flatten(),
        'Test Set Unity Plot',
        os.path.join(output_dir, 'test_unity_plot.png')
    )
    
    # Save model configuration with metrics
    config = {
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'output_size': 1,
        'target_key': args.target_key,
        'train_size': train_size,
        'test_size': test_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'train_test_split': args.train_test_split,
        'fill_strategy': args.fill_strategy,
        'missing_strategy': args.missing_strategy,
        'tokenize': args.tokenize,
        'token_threshold': args.token_threshold,
        'metrics': {
            'train': {
                'r2': float(train_r2),
                'mae': float(train_mae)
            },
            'test': {
                'r2': float(test_r2),
                'mae': float(test_mae)
            }
        }
    }
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    # Save predictions and targets for train and test
    train_df = pd.DataFrame({
        'true_values': np.array(all_train_targets).flatten(),
        'predictions': np.array(all_train_predictions).flatten()
    })
    test_df = pd.DataFrame({
        'true_values': np.array(all_test_targets).flatten(),
        'predictions': np.array(all_test_predictions).flatten()
    })
    train_df.to_csv(os.path.join(output_dir, 'train_predictions.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss History')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()
    print(f"All results have been saved to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 1D CNN model')
    parser.add_argument('--train_data', type=str, default=DEFAULT_CONFIG['train_data'],
                      help='Path to training data JSON file')
    parser.add_argument('--target_key', type=str, default=DEFAULT_CONFIG['target_key'],
                      help='Key in JSON file containing target values')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                      help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_CONFIG['learning_rate'],
                      help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_CONFIG['hidden_size'],
                      help='Size of hidden layers')
    parser.add_argument('--log_interval', type=int, default=DEFAULT_CONFIG['log_interval'],
                      help='How many batches to wait before logging training status')
    parser.add_argument('--train_test_split', type=float, default=DEFAULT_CONFIG['train_test_split'],
                      help='Train-test split ratio')
    parser.add_argument('--fill_strategy', type=str, default=DEFAULT_CONFIG['fill_strategy'],
                      help='Strategy to fill missing values (average, minimum, or median)')
    parser.add_argument('--missing_strategy', type=str, default=DEFAULT_CONFIG['missing_strategy'],
                      help='How to handle missing target values: fill (default) or drop')
    parser.add_argument('--tokenize', action='store_true', default=DEFAULT_CONFIG['tokenize'],
                      help='Whether to tokenize E and S values')
    parser.add_argument('--token_threshold', type=int, default=DEFAULT_CONFIG['token_threshold'],
                      help='Threshold for tokenization (values > threshold become 1, between 0 and threshold become 0)')
    
    args = parser.parse_args()
    train_model(args) 