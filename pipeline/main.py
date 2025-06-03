import argparse
import os
from openai import OpenAI
import utils, llm_utils, ocr_utils, utils, preprocessing
from llm_utils import *
from layout_utils import *
from utils import *
from directory_scanner import *
import numpy as np
from tqdm import tqdm
# from PIL import Image

"""
    == GUIDE TO DATA UNDERSTANDING ==
    /input (direct images from PDF)
    /out_binary_... (applied some binarization to input images)
    /{}_cut (same thing as in base folder but cut according to layout_analysis)
    /layout_data (with json containing results of the layout analysis)

    /out_tesseract... (applied ocr onto those images)
    /out_transcription... (put together the text into one file per document. Before could be one per page or per layout framgent)
    /out_llm... (applied llm improvement to the transcription)

"""

def pdf_to_images(args):
    ## Convert from PDF and all images into input folder
    pdf_paths = get_pdf_paths(args.folder_pdf)
    for pdfpath in pdf_paths:
        pdf2image(pdfpath, args.input_folder)

def pre_process_images(args):
    ## PreProcess Images
    ## TODO: update to be more flexible
    preprocessing.preprocess_all(args.input_folder, args.output_binary_1, args.output_binary_2)

#input_path has the original or binary images (ex: out_binary_...[without cut] or input)
def apply_layout_module(args, input_path, layout_analysis = False):
    ## Takes images in input_path folder, cuts them acording to results_json and saves them in "_cut" folder
    input_name = os.path.basename(os.path.normpath(input_path))
    result_json = f"/data/users/pfont/layout_data/{input_name}/results.json"
    input_path_cut = input_path + "_cut"
    if (layout_analysis):
        surya_save_folder(input_path, args.output_surya)
    cut_save_boxes(result_json, input_path, input_path_cut)

#input_path has the already pre-processed images (ex: out_binary_...) It will also process the cut version
def tesseract_merge_all(input_path, args):
    name = f"_{os.path.basename(os.path.normpath(input_path))}"
    #take out the out_ of the input_name
    input_name = '_'.join(name.split("_")[1:])
    input_path_cut = input_path + "_cut"
    
    # Entire Page
    out_tesseract = args.output_folder_tesseract + input_name
    ocr_utils.file_per_page_command(input_path, out_tesseract)

    out_transcription = args.output_transcriptions + input_name
    merge_text_files_pages(out_tesseract, out_transcription)

    # Cut by Layout
    out_tesseract_cut = args.output_folder_tesseract + input_name + "_cut"
    ocr_utils.file_per_page_command(input_path_cut, out_tesseract_cut)

    out_transcription_cut = args.output_transcriptions + input_name + "_cut"
    merge_text_files_by_cut(out_tesseract_cut, out_transcription_cut)

def apply_llm(input_path, args):
    name = f"_{os.path.basename(os.path.normpath(input_path))}"
    #take out the out_ of the input_name
    input_name = '_'.join(name.split("_")[1:])
    input_name_cut = input_name + "_cut"

    extensions = [input_name, input_name_cut]

    start_llm_server(args.model_name, port=8000)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )

    ## Do prompts
    for extension in extensions:
        folder = args.output_transcriptions + extension
        out_folder = args.output_folder_llm + extension
        for filename in os.listdir(folder):
            text_path = os.path.join(folder, filename)
            f = open(text_path, "r")
            ocr_text = f.read()
            prompt = get_prompts(ocr_text, "prompt6")
            output = query_llm(client, args.model_name, prompt)
            save_final_output(output, filename, out_folder)

    #Kill Server
    end_llm_server()

def llm_10_times_generation(args, harmonized_folder="/data/users/pfont/out_harmonized_ocr/",filename_json="llm_versions.json"):
    # Start LLM server
    start_llm_server(args.model_name, port=8000)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )

    # Load previous data
    json_path = os.path.join(args.base_directory , filename_json)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
    else:
        data_dict = {}

    for filename in tqdm(os.listdir(harmonized_folder)):
        # Open text
        text_path = os.path.join(harmonized_folder , filename)
        with open(text_path, "r", encoding="utf-8") as f:
            ocr_text = f.read()

        # Create prompt
        prompt = get_prompts(ocr_text, "prompt6")

        # Get and save metadata
        doc_id = filename.split("_")[-1].split(".")[0] #TODO: Fix. In the case of document_cut.txt, we get the cut instead of the ID
        extension = "normal"
        rep = 6
        if ("cut" in filename):
            extension = "cut"
            rep = 4
        
        if doc_id not in data_dict:
            data_dict[doc_id] = {}
            data_dict[doc_id]["llm_version_info"] = []
            data_dict[doc_id]["llm_version_text"] = []
        data_dict[doc_id][f"original_ocr_{extension}"] = ocr_text

        for i in range(rep):
            version_info = f"v_{extension}_{i+1}"
            version_text = query_llm(client, args.model_name, prompt)
            
            data_dict[doc_id]["llm_version_info"].append(version_info)
            data_dict[doc_id]["llm_version_text"].append(version_text)

    # Save with new data
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

def evaluate_all_transcriptions():
    # Calculate editing distance between all pais of tesseract output and llm_output
    csv_file = "complete_results.csv"
    results = process_all_output_directories(
        base_directory="/data/users/pfont/",
        csv_file=csv_file
    )
    visualize_summary_by_key(csv_file)


def main(args):    
    """input_folder = args.output_binary_1 + '_hisam' + '_inverted'
    apply_layout_module(args, input_folder, layout_analysis = False)
    tesseract_merge_all(input_folder, args)
    apply_llm(input_folder, args)
    evaluate_all_transcriptions()"""
    harmonized_folder="/data/users/pfont/out_harmonized_ocr/"
    filename_json="llm_versions.json"
    # Start LLM server
    start_llm_server(args.model_name, port=8000)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )

    # Load previous data
    json_path = os.path.join(args.base_directory , filename_json)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
    else:
        data_dict = {}

    for filename in tqdm(os.listdir(harmonized_folder)):
        if "cut" in filename:
            # Open text
            text_path = os.path.join(harmonized_folder , filename)
            with open(text_path, "r", encoding="utf-8") as f:
                ocr_text = f.read()

            # Create prompt
            prompt = get_prompts(ocr_text, "prompt6")

            # Get and save metadata
            doc_id = filename.split("_")[-2].split(".")[0]
            data_dict[doc_id][f"original_ocr_cut"] = ocr_text

            for i in range(4):
                version_info = f"v_cut_{i+1}"
                print(doc_id, data_dict[doc_id].keys(), version_info)
                version_text = query_llm(client, args.model_name, prompt)
                data_dict[doc_id]["llm_version_info"].append(version_info)
                data_dict[doc_id]["llm_version_text"].append(version_text)

    # Save with new data
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Process text files and compute metrics.")
    parser.add_argument("--base_directory", type=str, default="/data/users/pfont", help="Base directory with all data")
    parser.add_argument("--folder_pdf", type=str, default="/data/users/pfont/03-GT_MASSIU_VE_DE_DEDALO", help="Folder path containing the PDF files")
    parser.add_argument("--input_folder", type=str, default="/data/users/pfont/input", help="Folder path containing the input files")
    parser.add_argument("--output_folder_tesseract", type=str, default="/data/users/pfont/out_tesseract", help="Folder path containing the output files")
    parser.add_argument("--output_binary_1", type=str, default="/data/users/pfont/out_binary")
    parser.add_argument("--output_binary_2", type=str, default="/data/users/pfont/out_binary_simple")
    parser.add_argument("--output_surya", type=str, default="/data/users/pfont/layout_data")
    parser.add_argument("--output_folder_llm", type=str, default="/data/users/pfont/out_llm", help="Folder path containing the output files after LLM")
    parser.add_argument("--output_transcriptions", type=str, default="/data/users/pfont/out_transcription", help="Folder path containing the output files after LLM")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-4", help="LLM model name")
    args = parser.parse_args()

    main(args)


