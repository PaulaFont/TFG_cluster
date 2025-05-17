import jiwer
import os

def calculate_metrics(predicted_text, ground_truth):
    print ("Calculating metrics...")
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])
    transformed_gt = transformation(ground_truth)
    transformed_pred = transformation(predicted_text)
    wer = jiwer.wer(transformed_gt, transformed_pred)
    cer = jiwer.cer(ground_truth, predicted_text)

    return wer, cer

def save_corrected_text(dir, model_name, prompt_name, doc_id, corrected_text):
    """Guarda el texto corregido en un archivo y devuelve su path."""
    file_name = f"{model_name}_{prompt_name}_{doc_id}.txt"
    file_path = os.path.join(dir, file_name)

    print (f"Saving new files at {file_path}")

    with open(file_path, "w") as f:
        f.write(corrected_text)

    return file_path