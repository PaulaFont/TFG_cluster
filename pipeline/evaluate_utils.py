import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import Levenshtein
import numpy as np
import re
from difflib import SequenceMatcher


def normalize_text(text):
    """Normalize text by removing extra whitespace and line breaks"""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()

def levenshtein_distance(text1, text2):
    """Calculate the Levenshtein (edit) distance between two texts"""
    return Levenshtein.distance(text1, text2)

def normalized_levenshtein(text1, text2):
    """Calculate the normalized Levenshtein distance (0-1)"""
    distance = Levenshtein.distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 0.0
    return distance / max_len

def jaro_winkler_similarity(text1, text2):
    """Calculate Jaro-Winkler similarity (higher means more similar)"""
    return Levenshtein.jaro_winkler(text1, text2)

def word_error_rate(text1, text2):
    """Calculate Word Error Rate"""
    words1 = text1.split()
    words2 = text2.split()
    
    # Dynamic programming matrix
    dp = np.zeros((len(words1) + 1, len(words2) + 1))
    
    # Initialize first row and column
    for i in range(len(words1) + 1):
        dp[i, 0] = i
    for j in range(len(words2) + 1):
        dp[0, j] = j
    
    # Fill the matrix
    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i-1] == words2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]) + 1
    
    # Calculate WER
    return dp[len(words1), len(words2)] / len(words1) if len(words1) > 0 else 0

def sequence_matcher_ratio(text1, text2):
    """Use Python's difflib to get a similarity ratio (higher means more similar)"""
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_all_metrics(ocr_text, llm_text, normalize=True):
    """Calculate all text comparison metrics"""
    if normalize:
        ocr_text = normalize_text(ocr_text)
        llm_text = normalize_text(llm_text)
    
    results = {
        "levenshtein_distance": levenshtein_distance(ocr_text, llm_text),
        "normalized_levenshtein": normalized_levenshtein(ocr_text, llm_text),
        "jaro_winkler_similarity": jaro_winkler_similarity(ocr_text, llm_text),
        "word_error_rate": word_error_rate(ocr_text, llm_text),
        "sequence_matcher_ratio": sequence_matcher_ratio(ocr_text, llm_text)
    }
    
    return results

def analyze_results(df, threshold_normalized_levenshtein=0.3, threshold_wer=0.4):
    """
    Analyze results and identify potential cases of excessive LLM modification
    
    Args:
        df: DataFrame with comparison metrics
        threshold_normalized_levenshtein: Threshold for normalized Levenshtein distance
        threshold_wer: Threshold for Word Error Rate
    
    Returns:
        DataFrame with problematic files flagged
    """
    # Add flags for potential issues
    df['flag_levenshtein'] = df['normalized_levenshtein'] > threshold_normalized_levenshtein
    df['flag_wer'] = df['word_error_rate'] > threshold_wer
    df['flagged'] = df['flag_levenshtein'] | df['flag_wer']
    
    # Print summary
    print(f"Total files processed: {len(df)}")
    print(f"Files with high normalized Levenshtein (>{threshold_normalized_levenshtein}): {df['flag_levenshtein'].sum()}")
    print(f"Files with high WER (>{threshold_wer}): {df['flag_wer'].sum()}")
    print(f"Total flagged files: {df['flagged'].sum()}")
    
    return df

def visualize_results(df, output_dir="plots"):
    """
    Create visualizations of the comparison metrics
    
    Args:
        df: DataFrame with comparison metrics
        output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Histogram of normalized Levenshtein distance
    plt.figure(figsize=(10, 6))
    plt.hist(df['normalized_levenshtein'], bins=20, alpha=0.7)
    plt.axvline(df['normalized_levenshtein'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.title('Distribution of Normalized Levenshtein Distance')
    plt.xlabel('Normalized Levenshtein Distance')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'levenshtein_hist.png'))
    
    # Histogram of Word Error Rate
    plt.figure(figsize=(10, 6))
    plt.hist(df['word_error_rate'], bins=20, alpha=0.7)
    plt.axvline(df['word_error_rate'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.title('Distribution of Word Error Rate')
    plt.xlabel('Word Error Rate')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'wer_hist.png'))
    
    # Scatter plot of Levenshtein vs WER
    plt.figure(figsize=(10, 6))
    plt.scatter(df['normalized_levenshtein'], df['word_error_rate'], alpha=0.7)
    plt.title('Normalized Levenshtein vs Word Error Rate')
    plt.xlabel('Normalized Levenshtein Distance')
    plt.ylabel('Word Error Rate')
    plt.savefig(os.path.join(output_dir, 'levenshtein_vs_wer.png'))
    
    # Length difference analysis
    plt.figure(figsize=(10, 6))
    plt.scatter(df['ocr_length'], df['llm_length'], alpha=0.7)
    plt.plot([0, df['ocr_length'].max()], [0, df['ocr_length'].max()], 'r--')
    plt.title('OCR Text Length vs LLM Text Length')
    plt.xlabel('OCR Text Length (characters)')
    plt.ylabel('LLM Text Length (characters)')
    plt.savefig(os.path.join(output_dir, 'length_comparison.png'))
    
    print(f"Plots saved to {output_dir}")

def process_file_pairs(tesseract_folder, llm_folder, key, csv_file):
    """
    Adds information (or creates) csv_file. 
    The csv will add the "some" distance between the files in both folders with same name. 
    The key will be the type of file we are analysing (binary, normal, binary_simple, with layout...)
    The columns of the csv are: name_document, len_tess, len_llm, distances calculated, key, cut (0 or 1)
    
    Args:
        tesseract_folder: Directory containing Tesseract OCR output files
        llm_folder: Directory containing LLM corrected files
        key: Type of file being analyzed (e.g., 'binary', 'normal', 'with_layout')
        csv_file: Path to the CSV file to create or update
    
    Returns LIST of new entries added
    """
    # Check if CSV file exists and load existing data if it does
    existing_data = []
    file_exists = os.path.isfile(csv_file)
    
    if file_exists:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
                fieldnames = reader.fieldnames
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            file_exists = False
    
    # Define CSV column names if the file doesn't exist
    if not file_exists:
        fieldnames = [
            'name_document', 'len_tess', 'len_llm', 'levenshtein_distance', 
            'normalized_levenshtein', 'jaro_winkler_similarity', 'word_error_rate',
            'sequence_matcher_ratio', 'length_difference', 'key', 'layout', 'binary'
        ]
    
    # Get list of files in Tesseract directory
    tesseract_files = [f for f in os.listdir(tesseract_folder) if os.path.isfile(os.path.join(tesseract_folder, f))]
    
    # Process each file
    new_entries = []
    for filename in tesseract_files:
        # Check if corresponding LLM file exists
        tesseract_path = os.path.join(tesseract_folder, filename)
        llm_path = os.path.join(llm_folder, filename)
        
        if not os.path.exists(llm_path):
            print(f"Warning: No matching LLM file for {filename}")
            continue
            
        # Read files
        with open(tesseract_path, 'r', encoding='utf-8') as f:
            tesseract_text = f.read()
            
        with open(llm_path, 'r', encoding='utf-8') as f:
            llm_text = f.read()
        
        # Normalize texts
        normalized_tesseract = normalize_text(tesseract_text)
        normalized_llm = normalize_text(llm_text)
            
        # Calculate metrics
        metrics = calculate_all_metrics(tesseract_text, llm_text)
        
        # Determine if layout and binary
        cut = 1 if "cut" in key else 0

        binary = 0
        if "binarysimple" in key:
            binary = 1
        elif "binary" in key:
            binary = 2

        # Create entry
        entry = {
            'name_document': filename,
            'len_tess': len(normalized_tesseract),
            'len_llm': len(normalized_llm),
            'levenshtein_distance': metrics['levenshtein_distance'],
            'normalized_levenshtein': metrics['normalized_levenshtein'],
            'jaro_winkler_similarity': metrics['jaro_winkler_similarity'],
            'word_error_rate': metrics['word_error_rate'],
            'sequence_matcher_ratio': metrics['sequence_matcher_ratio'],
            'length_difference': len(normalized_llm) - len(normalized_tesseract),
            'key': key,
            'layout': cut,
            'binary' : binary
        }
        
        new_entries.append(entry)
    
    # Write results to CSV (either create new file or append to existing)
    mode = 'a' if file_exists else 'w'
    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_entries)
    
    print(f"Added {len(new_entries)} entries to {csv_file}")

    return new_entries
