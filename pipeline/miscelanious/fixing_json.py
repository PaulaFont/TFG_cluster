import json
base_directory = "" #TODO: add
json_path = os.path.join(base_directory , "llm_versions.json")
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

for key in data_dict.keys():
    filename = f"rsc37_rsc176_{key}_cut.txt"
    folder = "/data/users/pfont/out_harmonized_ocr"
    file_path = os.path.join(folder, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        file_text = f.read()

    data_dict[key]["original_ocr_cut"] = file_text
    
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=2)