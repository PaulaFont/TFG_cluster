import os
import yaml
from openai import OpenAI
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Servidor local del LLM
    api_key="token-abc123"
)

# Función para obtener los prompts
def get_prompts(ocr_text):

    #Llegir fitxer amb inici del prompt
    prompts = {}
    for filename in os.listdir("./data/prompt_texts/"):
        text_path = os.path.join("./data/prompt_texts/", filename)
        filename = filename.split(".")[0]
        prompt_text = open(text_path).read()
        prompts[filename] = prompt_text + ocr_text

    return prompts

# Función para llamar al LLM con diferentes prompts
def query_llm(model, prompt, temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        #temperature=temperature
    )
    return response.choices[0].message.content

# Función principal para evaluar combinaciones de modelos y prompts
def evaluate_model(model, ocr_text, temperature=0):
    prompts = get_prompts(ocr_text)
    results = {}
    times = []

    for key, prompt in prompts.items():
        start_time = time.time()
        results[key] = query_llm(model, prompt, temperature)
        query_time = round(time.time() - start_time, 4)
        times.append(query_time)
    return results, times
