import os
from evaluate_llm_utils import process_file_pairs

def process_all_output_directories(base_directory, csv_file, threshold_normalized_levenshtein=0.3, threshold_wer=0.4):
    """
    Scans a directory for pairs of folders starting with 'out_transctiption_' and 'out_llm_',
    extracts the key from the folder names, and processes each pair using evaluate_changes.
    
    Args:
        base_directory: The parent directory containing the output folders
        csv_file: Path to the CSV file to create or update
        threshold_normalized_levenshtein: Threshold for normalized Levenshtein distance
        threshold_wer: Threshold for Word Error Rate
        
    Returns:
        Dictionary with keys and counts of processed files
    """
    # Get all directories in the base directory
    all_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    
    # Find transcription and LLM directories
    transcription_dirs = [d for d in all_dirs if d.startswith('out_transcription_')]
    llm_dirs = [d for d in all_dirs if d.startswith('out_llm_')]
    
    # Extract keys
    transcription_keys = {d.replace('out_transcription_', ''): d for d in transcription_dirs}
    llm_keys = {d.replace('out_llm_', ''): d for d in llm_dirs}
    
    # Find common keys
    common_keys = set(transcription_keys.keys()) & set(llm_keys.keys())
    
    if not common_keys:
        print("No matching directory pairs found.")
        return {}
    
    # Process each directory pair
    results = {}
    for key in common_keys:
        transcription_dir = os.path.join(base_directory, transcription_keys[key])
        llm_dir = os.path.join(base_directory, llm_keys[key])
        
        print(f"\nProcessing key: {key}")
        print(f"Transcription directory: {transcription_dir}")
        print(f"LLM directory: {llm_dir}")
        
        # Process the directory pair
        entries = process_file_pairs(
            tesseract_folder=transcription_dir,
            llm_folder=llm_dir,
            key=key,
            csv_file=csv_file,
        )
        
        results[key] = {
            'total': len(entries)
        }
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total key types processed: {len(results)}")    
    return results

def visualize_summary_by_key(csv_file, output_directory="plots"):
    """
    Creates visualizations comparing metrics across different keys/document types
    
    Args:
        csv_file: Path to the CSV file with evaluation results
        output_directory: Directory to save the generated plots
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get unique keys
    keys = df['key'].unique()
    
    if len(keys) <= 1:
        print("Need multiple keys for comparison visualization")
        return
    
    # Create comparison plots
    metrics = ['normalized_levenshtein', 'word_error_rate', 'jaro_winkler_similarity']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Create boxplot for each key
        boxplot_data = [df[df['key'] == key][metric] for key in keys]
        plt.boxplot(boxplot_data, labels=keys)
        
        plt.title(f'{metric} by Document Type')
        plt.ylabel(metric)
        plt.xlabel('Document Type')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig(os.path.join(output_directory, f'{metric}_by_key.png'))
        plt.close()
    
    print(f"Created comparison visualizations in {output_directory}")