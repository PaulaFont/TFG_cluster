import os
import csv
import re
import string
from tqdm import tqdm


def analyze_text(text):
    lines = text.splitlines()
    num_lines = len(lines)
    total_length = len(text)

    words = re.findall(r'\b\w+\b', text.lower())
    num_words = len(words)
    avg_word_length = sum(len(w) for w in words) / num_words if num_words else 0

    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    alphabetic_letter_count = sum(1 for c in text if 65 <= ord(c) <= 90 or 97 <= ord(c) <= 122)

    short_lines = sum(1 for line in lines if len(line.strip()) <= 3)
    garbage_lines = sum(1 for line in lines if len(re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]', line)) / (len(line) + 1e-5) < 0.3)

    return {
        "total_length": total_length,
        "num_lines": num_lines,
        "avg_line_length": total_length / num_lines if num_lines else 0,
        "num_words": num_words,
        "avg_word_length": avg_word_length,
        "non_ascii_chars": non_ascii_count,
        "short_lines": short_lines,
        "garbage_lines": garbage_lines,
        "alphabetic_letter_count": alphabetic_letter_count,
    }

def analyze_documents(base_dir, output_csv="document_analysis.csv"):
    versions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("out_transcription_")]
    results = []

    for version_dir in versions:
        version_name = version_dir.replace("out_transcription_", "")
        full_path = os.path.join(base_dir, version_dir)

        for filename in tqdm(os.listdir(full_path)):
            file_path = os.path.join(full_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    metrics = analyze_text(text)
                    metrics["filename"] = filename
                    metrics["version"] = version_name
                    results.append(metrics)

    # Write to CSV
    print("Saving")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    analyze_documents("/data/users/pfont/")