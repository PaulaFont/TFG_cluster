import argparse
import os
from openai import OpenAI
import utils, llm_utils, ocr_utils, utils, preprocessing
from llm_utils import *
from layout_utils import *
from utils import *
from directory_scanner import *

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
    cut_save_boxes(results_json, input_path, input_path_cut)

#input_path has the already pre-processed images (ex: out_binary_...) It will also process the cut version
def tesseract_merge_all(input_path, args):
    input_name = f"_{os.path.basename(os.path.normpath(input_path))}"
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
    input_name = f"_{os.path.basename(os.path.normpath(input_path))}"
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


def evaluate_all_transcriptions()
    # Calculate editing distance between all pais of tesseract output and llm_output
    csv_file = "complete_results.csv"
    results = process_all_output_directories(
        base_directory="/data/users/pfont/",
        csv_file=csv_file
    )
    visualize_summary_by_key(csv_file)


def main(args):    
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Process text files and compute metrics.")
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
