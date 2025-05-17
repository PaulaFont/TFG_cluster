import os
import re
from PIL import Image
import pypdfium2 as pdfium

def pdf2image(pdfpath, input_folder):
    pdfname = os.path.basename(pdfpath)
    filename, _ = os.path.splitext(pdfname)
    setImages = []
    Image.MAX_IMAGE_PIXELS = 400000000
    pdf = pdfium.PdfDocument(pdfpath)
    for i, page in enumerate(pdf):
        imgpath = os.path.join(input_folder, f"{filename}_{i}.jpg")
        setImages.append(imgpath)
        if os.path.exists(imgpath):
            continue
        img = page.render(scale=4.861111).to_pil()
        img.save(imgpath, dpi=(350, 350))
    pdf.close()
    return setImages

def get_pdf_paths(folder_path):
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]

def save_final_output(text, filename, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, filename)
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Output saved to {out_path}")

def merge_text_files_pages(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    files = [f for f in os.listdir(source_dir) if f.endswith(".txt")]
    grouped_files = {}
    pattern = re.compile(r'(.+)_([0-9]+)_([0-9]+)\.txt$')

    for file in files:
        match = pattern.match(file)
        if match:
            base_name, doc_id, part_id = match.groups()
            key = f"{base_name}_{doc_id}"
            grouped_files.setdefault(key, []).append((int(part_id), file))

    for key, parts in grouped_files.items():
        parts.sort()
        output_file = os.path.join(target_dir, f"{key}.txt")
        with open(output_file, "w", encoding="utf-8") as outfile:
            for _, file in parts:
                with open(os.path.join(source_dir, file), "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n")
    print("Merged files saved in:", target_dir)


def merge_text_files_by_cut(source_dir, target_dir, min_confidence = 55):

    os.makedirs(target_dir, exist_ok=True)
    files = [f for f in os.listdir(source_dir) if f.endswith(".txt")]
    grouped_files = {}
    
    pattern = re.compile(r'^(.+?_\d+)_([0-9]+)_order_([0-9]+)_conf_([0-9]+)\.txt$')
    # Example: rsc37_rsc176_291_1_order_3_conf_84.txt
    # groups: ('rsc37_rsc176_291', '1', '3', '84')
    
    for file in files:
        match = pattern.match(file)
        if match:
            doc_key, page_num, order_num, confidence = match.groups()
            if int(confidence) > min_confidence:
                grouped_files.setdefault(doc_key, []).append(
                    (int(page_num), int(order_num), file)
                )
    
    for doc_key, parts in grouped_files.items():
        # Sort by page number first, then order number
        parts.sort()
        output_file = os.path.join(target_dir, f"{doc_key}.txt")
        with open(output_file, "w", encoding="utf-8") as outfile:
            for _, __, file in parts:
                with open(os.path.join(source_dir, file), "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n")
    print("Merged files saved in:", target_dir)
    