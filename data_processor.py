import json
import re
import numpy as np
import ast
import pandas as pd

def parse_input_list(input_str):
    """
    Parse the input list string from CSV into a list of tuples.
    
    Args:
        input_str (str): Input string in format "[(1, 'E4'), (2, 'E4'), ...]"
        
    Returns:
        list: List of tuples containing (position, type)
    """
    try:
        # Remove any extra whitespace and parse the string as a Python literal
        input_str = input_str.strip()
        return ast.literal_eval(input_str)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing input string: {e}")
        return []

def convert_to_matrix(input_list, max_length=20):
    """
    Convert input list into a 3x20 matrix format according to user requirements.
    
    Args:
        input_list (list): List of tuples (position, type)
        max_length (int): Maximum length of the sequence (default 20)
        
    Returns:
        numpy.ndarray: 3x20 matrix where:
            - Row 0: 0/1, 1 if position has a monomer, else 0
            - Row 1: E-type count (+1), 0 if not E
            - Row 2: S-type count (+1), 0 if not S
    """
    # Initialize the matrix with zeros
    matrix = np.zeros((3, max_length))
    
    # Process each tuple in the input list
    for pos, type_code in input_list:
        if 1 <= pos <= max_length:
            idx = pos - 1
            matrix[0, idx] = 1  # Mark presence
            if type_code.startswith('E'):
                matrix[1, idx] = int(type_code[1:]) + 1
                matrix[2, idx] = 0
            elif type_code.startswith('S'):
                matrix[1, idx] = 0
                matrix[2, idx] = int(type_code[1:]) + 1
    return matrix

def process_csv_to_json(csv_file, output_file):
    """
    Process CSV file and convert to JSON format.
    
    Args:
        csv_file (str): Path to input CSV file
        output_file (str): Path to output JSON file
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Process each row
    results = []
    for _, row in df.iterrows():
        # Parse input list
        input_list = parse_input_list(row['Input List'])
        
        # Convert to matrix
        matrix = convert_to_matrix(input_list)
        
        # Create result dictionary
        result = {
            'name': row['Name'],
            'input_matrix': matrix.tolist(),
            'start_frame': float(row['Start Frame']),
            'area_avg': float(row['Area AVG']),
            'area_std': float(row['Area STD']),
            'rg_avg': float(row['RG AVG']),
            'rg_std': float(row['RG STD']),
            'rdf_peak': float(row['RDF Peak']),
            'coordination_at_minimum': float(row['Coordination at Minimum'])
        }
        
        results.append(result)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # Example usage
    csv_file = "Data/yxan.csv"
    output_file = "Data/processed_data.json"
    process_csv_to_json(csv_file, output_file) 