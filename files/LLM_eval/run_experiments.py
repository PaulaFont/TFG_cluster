import argparse
import logging
import glob, csv, os
from src.llm_eval import evaluate_model
from src.utils import calculate_metrics, save_corrected_text

DATA_DIR = "data"
OCR_TEXTS_DIR = os.path.join(DATA_DIR, "ocr_raw")
CORRECTED_TEXTS_DIR = os.path.join(DATA_DIR, "corrected_texts")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")

os.makedirs(CORRECTED_TEXTS_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    filename="experiments.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()  # To show in terminal
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

parser = argparse.ArgumentParser(description="Ejecuta experimentos con un modelo LLM.")
parser.add_argument("--model", type=str, required=True, help="Nombre del modelo a usar.")
parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura para la generación.")
parser.add_argument("--chat", type=int, default=0, help="Whether if chat template is being used")
args = parser.parse_args()

logging.info(f"Starting experiment with model {args.model} and temperature {args.temperature}")

# Load OCR Texts
ocr_text_samples = {
    "278": open(OCR_TEXTS_DIR+"/278.txt").read(),
    "477": open(OCR_TEXTS_DIR+"/477.txt").read(),
    "717": open(OCR_TEXTS_DIR+"/717.txt").read()
}

# Archivo CSV para guardar resultados
csv_file = "experiment_results.csv"
first_write = not glob.glob(csv_file)

# Ejecutar evaluación para cada documento
results = []
with open(csv_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    if first_write:
        writer.writerow(["model", "temperature", "doc", "prompt", "WER", "CER", "query_time", "text_path", "chat"])

    for doc_name, ocr_text in ocr_text_samples.items():
        logging.info(f"📜 Procesando documento: {doc_name}")

        corrected_texts, times = evaluate_model(args.model, ocr_text, args.temperature)
        gt_text = open(f"data/ground_truth/{doc_name}.txt").read()
        for (prompt_type, corrected_text), query_time in zip(corrected_texts.items(), times):
            wer, cer = calculate_metrics(corrected_text, gt_text)
            model_name = (args.model).split("/")[1].replace('.', '_') # we get the model name without "."
            text_path = save_corrected_text(CORRECTED_TEXTS_DIR, model_name, prompt_type, doc_name, corrected_text)
            results.append([args.model, args.temperature, doc_name, prompt_type, wer, cer, query_time, text_path, args.chat])
            writer.writerow([args.model, args.temperature, doc_name, prompt_type, wer, cer, query_time, text_path, args.chat])

            logging.info(f"📊 {doc_name} - {prompt_type}: WER={wer:.4f}, CER={cer:.4f}")

# Ordenar por mejor WER
results.sort(key=lambda x: x[4]) #WER is the index 4 in the list ?

# Mostrar resumen final
logging.info("✅ Experimento finalizado. Resultados ordenados por WER:")
for res in results:
    logging.info(res)
