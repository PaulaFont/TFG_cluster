import os
from jiwer import wer, cer

# Define file paths
# Each entry should be a tuple of: (ground_truth_file, baseline_ocr_file, method_output_file)

documents = [
    ("/home/pfont/result_generation/files_ocr/278_gt.txt", "/home/pfont/result_generation/files_ocr/278_baseline.txt", "/home/pfont/result_generation/files_ocr/278_metod.txt"),
    ("/home/pfont/result_generation/files_ocr/477_gt.txt", "/home/pfont/result_generation/files_ocr/477_baseline.txt", "/home/pfont/result_generation/files_ocr/477_metod.txt"),
    ("/home/pfont/result_generation/files_ocr/717_gt.txt", "/home/pfont/result_generation/files_ocr/717_baseline.txt", "/home/pfont/result_generation/files_ocr/717_metod.txt"),
    ("/home/pfont/result_generation/files_ocr/876_gt.txt", "/home/pfont/result_generation/files_ocr/876_baseline.txt", "/home/pfont/result_generation/files_ocr/876_metod.txt"),
    ("/home/pfont/result_generation/files_ocr/816_gt.txt", "/home/pfont/result_generation/files_ocr/816_baseline.txt", "/home/pfont/result_generation/files_ocr/816_metod.txt"),
    ]

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Initialize error accumulators
baseline_wer_total = 0.0
method_wer_total = 0.0
baseline_cer_total = 0.0
method_cer_total = 0.0

for idx, (gt_file, base_file, method_file) in enumerate(documents, 1):
    gt = read_file(gt_file)
    baseline = read_file(base_file)
    method = read_file(method_file)

    baseline_wer = wer(gt, baseline)
    method_wer = wer(gt, method)
    baseline_cer = cer(gt, baseline)
    method_cer = cer(gt, method)

    print(f"Document {idx}:")
    print(f"  Baseline WER: {baseline_wer:.3f}, CER: {baseline_cer:.3f}")
    print(f"  Method   WER: {method_wer:.3f}, CER: {method_cer:.3f}\n")

    baseline_wer_total += baseline_wer
    method_wer_total += method_wer
    baseline_cer_total += baseline_cer
    method_cer_total += method_cer

# Average results
n = len(documents)
print("Average Metrics:")
print(f"  Baseline WER: {baseline_wer_total/n:.3f}, CER: {baseline_cer_total/n:.3f}")
print(f"  Method   WER: {method_wer_total/n:.3f}, CER: {method_cer_total/n:.3f}")
