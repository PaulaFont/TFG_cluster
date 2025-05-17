import os
from PIL import Image
import argparse
import tqdm


"""
Using surya from command line, saves all outputs in JSONS
"""
def process_layout_images(folder_image_path, output_folder, save_images=False):
    os.makedirs(output_folder, exist_ok=True) 
    command = f"surya_layout {folder_image_path} --images --output_dir {output_folder}"
    os.system(command) 

def binarize_image(image):
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files and compute metrics.")
    parser.add_argument("--folder_pdf", type=str, default="/data/users/pfont/03-GT_MASSIU_VE_DE_DEDALO", help="Folder path containing the PDF files")
    parser.add_argument("--input_folder", type=str, default="/data/users/pfont/input", help="Folder path containing the input files")
    parser.add_argument("--output_folder_layout", type=str, default="/data/users/pfont/layout_data", help="Folder path containing the output files")
    args = parser.parse_args()

    process_layout_images(args.input_folder, args.output_folder_layout)
     