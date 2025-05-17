import os
from PIL import Image
import argparse
import pypdfium2 as pdfium
import tqdm
from vllm import LLM, SamplingParams
import requests
from openai import OpenAI
import time
import re

# ---- CREATING AND OPENING IMAGES ----

"""
From pdfpath converts to JPG and saves in input folder
"""
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

"""
Returns list with all paths to PDF from whitin the folder_path
"""
def get_pdf_paths(folder_path):
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]

"""
Opens one image, makes sure it exists
"""
def get_image(image_path):    
    try:
        im = Image.open(image_path)
        return im
    except Exception as e:
        print(f"Error al abrir {image_path}: {e}")
        return None
    
def get_dpi(im):
    try:
        return im.info.get('dpi', [None])[0]  # get() to avoid KeyError
    except Exception as e:
        print(f"Error al obtener el DPI: {e}")
        return None

# ---- OCR TESSERACT ----

"""
Using pytesseract saves all tesseract outputs from one fodler to another
"""
def get_text(im, dpi):
    config_string = "dpi="+ str(dpi)
    output = pyt.image_to_string(im, lang='spa', config=config_string)
    return output

def file_per_page(folder_path, file_number):
    output_folder = f"./out_{file_number}/"
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        im = get_image(image_path)

        if (im == None):
            continue

        #For each page image, get text string and save to file
        file_info= get_text(im, get_dpi(im))

        page_number = filename.split(".")[0]
        out_path = os.path.join(output_folder, page_number) 
        text_file = open(out_path + ".txt", "w")
        text_file.write(file_info)
        text_file.close() 


"""
Using tesseract from command line, saves all  outputs from one folder to another
"""
def file_per_page_command(folder_image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True) 

    for filename in os.listdir(folder_image_path):
        image_path = os.path.join(folder_image_path, filename)
        filename = os.path.basename(image_path)
        filename, _ = os.path.splitext(filename)
        im = get_image(image_path)
        if (im == None):
            continue

        out_path = os.path.join(output_folder, filename) 

        command = f"tesseract {image_path} {out_path} --dpi {get_dpi(im)} -l spa"
        os.system(command) 


# ---- LLM SERVER ----

"""
Starts LLM according to model_name, opens server in llm_server screen
"""
def start_llm_server(model_name: str, chat_template: str = None):
    if not model_name:
        print("Error: You must give a model as an argument.")
        return
    
    print(f"Starting server LLM with model: {model_name}...")
    
    base_command = (
        f"CUDA_VISIBLE_DEVICES=1,2 vllm serve {model_name} --dtype auto --api-key token-abc123 --tensor-parallel-size 2 --enforce-eager --port 8000"
    )
    
    if chat_template:
        print(f"Using chat template {chat_template}")
        base_command += f" --chat-template {chat_template}"
    
    os.system(f"screen -dmS llm_server bash -c \"{base_command}\"")
    
    print("Waiting until server is ready...")
    
    while True:
        try:
            headers = {"Authorization": "Bearer token-abc123"}
            response = requests.get("http://localhost:8000/v1/models", headers=headers)
            if response.status_code == 200:
                print("Server is ready!")
                break
        except requests.RequestException:
            pass
        
        # Check if screen process is still running
        screen_list = os.popen("screen -list").read()
        if "llm_server" not in screen_list:
            print("Error: The server process is not running!")
            return
        
        print("Server not available yet, waiting 5 seconds...")
        time.sleep(5)

def end_llm_server():
    print("Terminating server...")
    os.system("screen -XS llm_server quit")
    print("Experiment completed.")

"""
Returns all prompts in folder. Creates the final prompts adding the ocr_text at the end. 
It only returns the prompt specified by key (same as filename)
"""
def get_prompts(ocr_text, key):
    prompts = {}
    for filename in os.listdir("./LLM_eval/data/prompt_texts/"):
        text_path = os.path.join("./LLM_eval/data/prompt_texts/", filename)
        filename = filename.split(".")[0]
        prompt_text = open(text_path).read()
        prompts[filename] = prompt_text + ocr_text
    return prompts[key]


"""
Gets a response to a prompt, asking the model in server
"""
def query_llm(client, model, prompt, temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        #temperature=temperature
    )
    return response.choices[0].message.content

"""
Saves "text" to file "filename" into folder "output_folder"
"""
def save_final_output(text, filename, output_folder):
    os.makedirs(output_folder, exist_ok=True) 
    out_path = os.path.join(output_folder, filename)
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Output saved to {out_path}")

"""
Reads all files in source_dir and merges the files from the same original pdf into one 
Saves it into new directory
"""
def merge_text_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    files = [f for f in os.listdir(source_dir) if f.endswith(".txt")]
    grouped_files = {}
    
    pattern = re.compile(r'(.+)_([0-9]+)_([0-9]+)\.txt$')
    
    for file in files:
        match = pattern.match(file)
        if match:
            base_name, doc_id, part_id = match.groups()
            key = f"{base_name}_{doc_id}"
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append((int(part_id), file))
    
    for key, parts in grouped_files.items():
        parts.sort()  # Ordenar por el número de parte
        output_file = os.path.join(target_dir, f"{key}.txt")
        with open(output_file, "w", encoding="utf-8") as outfile:
            for _, file in parts:
                with open(os.path.join(source_dir, file), "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n")
    
    print("Archivos combinados y guardados en:", target_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files and compute metrics.")
    parser.add_argument("--folder_pdf", type=str, default="/data/users/pfont/03-GT_MASSIU_VE_DE_DEDALO", help="Folder path containing the PDF files")
    parser.add_argument("--input_folder", type=str, default="/data/users/pfont/input", help="Folder path containing the input files")
    parser.add_argument("--output_folder_tesseract", type=str, default="/data/users/pfont/out_tesseract", help="Folder path containing the output files")
    parser.add_argument("--output_folder_llm", type=str, default="/data/users/pfont/out_llm", help="Folder path containing the output files after LLM")
    parser.add_argument("--output_transcriptions", type=str, default="/data/users/pfont/out_transcription", help="Folder path containing the output files after LLM")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-4", help="LLM model name")
    parser.add_argument("--chat_template", type=str, default=None, help="path to chat template to use")
    parser.add_argument("--temperature", type=str, default="0.1", help="Model sampling parameters: Temperature")
    parser.add_argument("--max_tokens", type=str, default="300", help="Model sampling parameters: Max Tokens")
    args = parser.parse_args()

    """
    ## Convert from PDF and all images into input folder
    pdf_paths = get_pdf_paths(args.folder_pdf)
    for pdfpath in pdf_paths:
        pdf2image(pdfpath, args.input_folder)
    
    ## Process all input images, save tesseract ocr output to output folder
    file_per_page_command(args.input_folder, args.output_folder_tesseract)"
    """

    """
    ## Serve model
    start_llm_server(args.model_name, args.chat_template)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )
    
    ## Do prompts
    for filename in os.listdir(args.output_transcriptions):
        text_path = os.path.join(args.output_transcriptions, filename)
        f = open(text_path, "r")
        ocr_text = f.read()
        prompt = get_prompts(ocr_text, "prompt6")
        output = query_llm(client, args.model_name, prompt)
        save_final_output(output, filename, args.output_folder_llm)

    #Kill Server
    end_llm_server()
    """

    
    # Put together txt files 
    # merge_text_files(args.output_folder_tesseract, args.output_transcriptions)
    
     