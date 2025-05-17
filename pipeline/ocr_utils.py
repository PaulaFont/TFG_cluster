import os
from PIL import Image
#import pytesseract as pyt
from tqdm import tqdm

def get_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error al abrir {image_path}: {e}")
        return None

def get_dpi(im):
    try:
        return im.info.get('dpi', [None])[0]
    except Exception as e:
        print(f"Error al obtener el DPI: {e}")
        return None

def get_text(im, dpi):
    config_string = "dpi=" + str(dpi)
    return pyt.image_to_string(im, lang='spa', config=config_string)

def file_per_page_command(folder_image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_image_path):
        image_path = os.path.join(folder_image_path, filename)
        filename = os.path.splitext(filename)[0]
        im = get_image(image_path)
        if im is None:
            continue
        out_path = os.path.join(output_folder, filename)
        command = f"tesseract {image_path} {out_path} --dpi {get_dpi(im)} -l spa"
        os.system(command)
