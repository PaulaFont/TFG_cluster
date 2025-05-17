import argparse
import os
from openai import OpenAI
import utils, llm_utils, ocr_utils, utils, preprocessing
from llm_utils import *
from layout_utils import *
from utils import *
from directory_scanner import *

def main(args):
    """
    ## Convert from PDF and all images into input folder
    pdf_paths = get_pdf_paths(args.folder_pdf)
    for pdfpath in pdf_paths:
        pdf2image(pdfpath, args.input_folder)
    """

    """
    ## PreProcess Images
    preprocessing.preprocess_all(args.input_folder, args.output_binary_1, args.output_binary_2)
    """
    input_path = "/data/users/pfont/out_binary_hisam"
    input_path_cut = input_path + "_cut"
    #surya_save_folder(input_path, args.output_surya)
    cut_save_boxes("/data/users/pfont/layout_data/out_binary_hisam/results.json", input_path, input_path_cut)

    new_path1 = args.output_folder_tesseract + "_binary_hisam"
    ocr_utils.file_per_page_command(input_path, new_path1)
    new_path2 = args.output_folder_tesseract + "_binary_hisam_cut"
    ocr_utils.file_per_page_command(input_path_cut, new_path2)

    #TODO: put together
    # Put together txt files (separated per pages)
    # merge_text_files_pages(args.output_folder_tesseract, args.output_transcriptions)
    in_path = new_path1
    out_path = args.output_transcriptions + "_binary_hisam"
    merge_text_files_pages(in_path, out_path)

    in_path = new_path2
    out_path = args.output_transcriptions + "_binary_hisam_cut"
    merge_text_files_by_cut(in_path, out_path)
    

    start_llm_server(args.model_name, port=8000)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )
    
    extensions = ["_binary_hisam", "_binary_hisam_cut"]
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
    
    """## Layout Analysis
    surya_save_folder(args.output_binary_1, args.output_surya)
    surya_save_folder(args.output_binary_2, args.output_surya)
    
    ##Cut images
    #TODO: fix hardcoding
    cut_save_boxes("/data/users/pfont/layout_data/out_binary/results.json", args.output_binary_1, args.output_binary_1_cut)
    cut_save_boxes("/data/users/pfont/layout_data/out_binary_simple/results.json", args.output_binary_2, args.output_binary_2_cut)"""
    
    """
    ## Process all input images, save tesseract ocr output to output folder
    new_path1 = args.output_folder_tesseract + "_binary"
    ocr_utils.file_per_page_command(args.output_binary_1, new_path1)
    new_path2 = args.output_folder_tesseract + "_binarysimple"
    ocr_utils.file_per_page_command(args.output_binary_2, new_path2)
    
    ## Process all input images (cut) with tesseract and save
    new_path1 = args.output_folder_tesseract + "_binary_cut"
    ocr_utils.file_per_page_command(args.output_binary_1_cut, new_path1)
    new_path2 = args.output_folder_tesseract + "_binarysimple_cut"
    ocr_utils.file_per_page_command(args.output_binary_2_cut, new_path2)
    """

    
    """    ## Serve model
    start_llm_server(args.model_name, args.chat_template)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )
    #TODO: change cuda_visible_devices on llm_utils.py
    
    extensions = ["_cut", "_binarysimple_cut", "_binarysimple", "_binary_cut", "_binary"]
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
    end_llm_server()"""
    
    """
    # Put together txt files (separated per pages)
    # merge_text_files_pages(args.output_folder_tesseract, args.output_transcriptions)
    in_path = args.output_folder_tesseract + "_binary"
    out_path = args.output_transcriptions + "_binary"
    merge_text_files_pages(in_path, out_path)
    in_path = args.output_folder_tesseract + "_binarysimple"
    out_path = args.output_transcriptions + "_binarysimple"
    merge_text_files_pages(in_path, out_path)

    # Put together txt files (separated per cut)
    in_path = args.output_folder_tesseract + "_binary_cut"
    out_path = args.output_transcriptions + "_binary_cut"
    merge_text_files_by_cut(in_path, out_path)
    in_path = args.output_folder_tesseract + "_binarysimple_cut"
    out_path = args.output_transcriptions + "_binarysimple_cut"
    merge_text_files_by_cut(in_path, out_path)
    """
    csv_file = "complete_results.csv"
    results = process_all_output_directories(
        base_directory="/data/users/pfont/",
        csv_file=csv_file
    )
    visualize_summary_by_key(csv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Process text files and compute metrics.")
    parser.add_argument("--folder_pdf", type=str, default="/data/users/pfont/03-GT_MASSIU_VE_DE_DEDALO", help="Folder path containing the PDF files")
    parser.add_argument("--input_folder", type=str, default="/data/users/pfont/input", help="Folder path containing the input files")
    parser.add_argument("--output_folder_tesseract", type=str, default="/data/users/pfont/out_tesseract", help="Folder path containing the output files")
    parser.add_argument("--output_binary_1", type=str, default="/data/users/pfont/out_binary")
    parser.add_argument("--output_binary_1_cut", type=str, default="/data/users/pfont/out_binary_cut")
    parser.add_argument("--output_binary_2", type=str, default="/data/users/pfont/out_binary_simple")
    parser.add_argument("--output_binary_2_cut", type=str, default="/data/users/pfont/out_binary_simple_cut")
    parser.add_argument("--output_surya", type=str, default="/data/users/pfont/layout_data")
    parser.add_argument("--output_folder_llm", type=str, default="/data/users/pfont/out_llm", help="Folder path containing the output files after LLM")
    parser.add_argument("--output_transcriptions", type=str, default="/data/users/pfont/out_transcription", help="Folder path containing the output files after LLM")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-4", help="LLM model name")
    parser.add_argument("--chat_template", type=str, default=None, help="path to chat template to use")
    parser.add_argument("--temperature", type=str, default="0.1", help="Model sampling parameters: Temperature")
    parser.add_argument("--max_tokens", type=str, default="300", help="Model sampling parameters: Max Tokens")
    args = parser.parse_args()
    args = parser.parse_args()
    main(args)
