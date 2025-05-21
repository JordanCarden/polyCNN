import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_monomer_distribution(json_file):
    """
    Analyze the distribution of E and S monomers in the dataset
    """
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize lists for E and S values
    e_values = []
    s_values = []
    
    # Process each sample
    for item in data:
        matrix = np.array(item['input_matrix'])
        # Row 1 contains E-type values
        e_values.extend(matrix[1])
        # Row 2 contains S-type values
        s_values.extend(matrix[2])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot E monomer distribution
    ax1.hist(e_values, bins=range(int(min(e_values)), int(max(e_values)) + 2), 
             color='blue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('E-type Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of E-type Values')
    ax1.grid(True, alpha=0.3)
    
    # Plot S monomer distribution
    ax2.hist(s_values, bins=range(int(min(s_values)), int(max(s_values)) + 2),
             color='red', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('S-type Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of S-type Values')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('monomer_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nE-type Values Distribution:")
    print(f"Min: {min(e_values)}")
    print(f"Max: {max(e_values)}")
    print(f"Mean: {np.mean(e_values):.2f}")
    print(f"Std: {np.std(e_values):.2f}")
    
    print("\nS-type Values Distribution:")
    print(f"Min: {min(s_values)}")
    print(f"Max: {max(s_values)}")
    print(f"Mean: {np.mean(s_values):.2f}")
    print(f"Std: {np.std(s_values):.2f}")

if __name__ == '__main__':
    json_file = 'Data/processed_data.json'
    analyze_monomer_distribution(json_file) 