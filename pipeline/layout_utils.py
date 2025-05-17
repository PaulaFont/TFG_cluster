import os
import json
from PIL import Image

def surya_save_folder(folder_image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    command = f"surya_layout {folder_image_path} --output_dir {output_folder} --images"
    os.system(command)


def crop_with_margin(image, bbox, margin=10):
    """Crop image with margin and clamp to image boundaries."""
    width, height = image.size
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width, x1 + margin)
    y1 = min(height, y1 + margin)
    return image.crop((x0, y0, x1, y1))


def cut_save_boxes(json_path, folder_image_path, output_folder, margin=10, confidence_threshold=0.5):
    os.makedirs(output_folder, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for file_base, values in data.items():
        # Open Image
        image_path = os.path.join(folder_image_path, file_base)
        image = Image.open(image_path+".png")  #TODO: fix hardcoding

        # Cut each box and save
        for box in values[0]["bboxes"]:
            coords = list(map(int, box["bbox"]))  # (x0, y0, x1, y1)
            confidence = box["confidence"]
            position = box["position"]
            if confidence >= confidence_threshold:
                cropped = crop_with_margin(image, coords, margin)
                filename = f"{file_base}_order_{position}_conf_{int(confidence * 100)}.png"
                cropped.save(os.path.join(output_folder, filename))
                print(f"Saved {filename}")
        
