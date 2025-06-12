import os
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
import time
import gradio as gr
import re
import pandas as pd

DOCUMENTS_PATH = "/data/users/pfont/final_documents"

def clean_text(text):
    """Clean corpus text"""
    text = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑàèìòùÀÈÌÒÙçÇ,.()\-\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_into_segments(words, min_words, max_words, overlap):
    segments = []
    i = 0
    while i < len(words):
        segment = words[i:i + max_words]
        if len(segment) < min_words:
            break
        segments.append(" ".join(segment))
        i += max_words - overlap
    return segments

def create_passages(document_directory=DOCUMENTS_PATH, min_words_per_paragraph = 50,
                    max_words_per_paragraph = 100, overlap = 30):
    section_headers = [
        "SENTENCIA", "RESULTANDO", "CONSIDERANDO", "VISTOS", "El Consejo falla", "sentencia", "considerando", "resultando", "probados"
    ]
    header_pattern = re.compile(r"^(SENTENCIA|RESULTANDO|CONSIDERANDO|VISTOS|El Consejo falla|sentencia|considerando|resultando|probados)\b", re.IGNORECASE)
      
    # Extract keys
    keys = ["all", "layout", "no_layout"]

    all_passages_data = [] # Will store dicts with text and metadata

    # We read all files within folder
    for filename in os.listdir(document_directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(document_directory, filename)

            #Gather information from filename
            name = filename.replace(".txt", "")
            key = filename.split("_")[-1] #key
            doc_id = filename.split("_")[2] #key

            with open(filepath, 'r', encoding='utf8') as file:
                content = file.read()
            
            content = clean_text(content)
            sections = re.split(r"(?=(" + "|".join(section_headers) + r"))", content)
            if sections[0] == "" and len(sections)>1 : sections = sections[1:] # handle if first split is empty
            current_section_type = "General" # Default
            
            processed_sections = []
            temp_section_content = ""
            for i, part in enumerate(sections):
                match = header_pattern.match(part)
                if match:
                    if temp_section_content: # Store previous section
                        processed_sections.append({"type": current_section_type, "content": temp_section_content.strip()})
                    current_section_type = match.group(1).capitalize()
                    temp_section_content = part[len(match.group(0)):].strip() # Content after header
                else:
                    temp_section_content += " " + part.strip()
            if temp_section_content: # Store the last section
                processed_sections.append({"type": current_section_type, "content": temp_section_content.strip()})

            for sec_data in processed_sections:
                section_content = sec_data["content"]
                section_type = sec_data["type"]
                words = section_content.split()
                
                if len(words) >= min_words_per_paragraph:
                    segmented_passages = split_into_segments(words, min_words_per_paragraph, max_words_per_paragraph, overlap)
                    for i, passage_text in enumerate(segmented_passages):
                        all_passages_data.append({
                            "text": passage_text,
                            "base_document_id": doc_id,
                            "processing_version": key,
                            "original_filename": filename, # Good to keep for reference
                            "section_type": section_type, # e.g., "Considerando"
                            "segment_index_in_section": i, # Order within its original section
                            "word_count": len(passage_text.split()),
                            "doc_id": doc_id
                        })
        
    df_passages = pd.DataFrame(all_passages_data)
    return df_passages


