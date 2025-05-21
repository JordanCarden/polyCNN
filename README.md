# Poly-CNN: Polymer Property Prediction using 1D CNN

This project implements a 1D Convolutional Neural Network (CNN) for predicting polymer properties based on monomer sequence information. The model processes input matrices representing monomer sequences and predicts various polymer properties such as radius of gyration (Rg), radial distribution function (RDF) peaks, and more.

## Features

- **1D CNN Architecture**: Specifically designed for processing polymer sequence data
- **Flexible Data Processing**:
  - Support for handling missing values (fill or drop strategies)
  - Tokenization of monomer counts (E and S types)
  - Configurable data splitting for training and testing
- **Comprehensive Training Pipeline**:
  - Automatic model training and evaluation
  - Unity plots for both training and test sets
  - Detailed metrics (R² and MAE) tracking
  - Training history visualization
- **Output Generation**:
  - Model checkpoints
  - Prediction results in CSV format
  - Visualization plots
  - Detailed configuration logs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hguangshuai/poly_CNN.git
cd poly_CNN
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Format

The input data should be in JSON format with the following structure:
```json
[
    {
        "input_matrix": [
            [row0_values],
            [row1_values],  // E-type monomer counts
            [row2_values]   // S-type monomer counts
        ],
        "target_key": value  // e.g., "rg_avg", "rdf_peak", etc.
    }
]
```

## Usage

### Basic Training

```bash
python training.py
```

### Advanced Training Options

```bash
python training.py \
    --train_data Data/processed_data.json \
    --target_key rg_avg \
    --batch_size 16 \
    --epochs 150 \
    --learning_rate 0.005 \
    --hidden_size 64 \
    --tokenize \
    --token_threshold 2 \
    --missing_strategy drop
```

### Key Parameters

- `--train_data`: Path to training data JSON file
- `--target_key`: Target property to predict (e.g., 'rg_avg', 'rdf_peak')
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 150)
- `--learning_rate`: Learning rate (default: 0.005)
- `--hidden_size`: Size of hidden layers (default: 64)
- `--tokenize`: Enable tokenization of monomer counts
- `--token_threshold`: Threshold for tokenization (default: 2)
- `--missing_strategy`: How to handle missing values ('fill' or 'drop')
- `--fill_strategy`: Strategy to fill missing values ('average', 'minimum', or 'median')

## Output

The training process generates an output directory with timestamp containing:
- `model.pth`: Trained model weights
- `model_config.json`: Model configuration and metrics
- `train_unity_plot.png`: Unity plot for training set
- `test_unity_plot.png`: Unity plot for test set
- `loss_history.png`: Training and testing loss history
- `train_predictions.csv`: Training set predictions
- `test_predictions.csv`: Test set predictions

## Recent Updates

- Added tokenization feature for E and S monomer counts
- Implemented flexible missing value handling
- Added comprehensive metrics tracking
- Improved visualization capabilities

## Future Improvements

- Add support for more polymer properties
- Implement cross-validation
- Add model interpretability features
- Support for different monomer types
- Integration with molecular dynamics simulations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and suggestions, please open an issue in the GitHub repository. 