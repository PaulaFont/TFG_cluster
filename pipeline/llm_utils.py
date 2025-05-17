import os
import requests
import time
import subprocess

HF_CACHE_DIR = "/data/users/pfont/models" 


def start_llm_server(model_name: str, port: int, chat_template: str = None):
    if not model_name:
        print("Error: You must give a model as an argument.")
        return False

    try:
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
        print(f"Ensured Hugging Face cache directory exists: {HF_CACHE_DIR}")
    except OSError as e:
        print(f"Error creating directory {HF_CACHE_DIR}: {e}")
        print("Please check permissions or choose a different directory.")
        return False

    os.environ['HF_HOME'] = HF_CACHE_DIR
    print(f"Set HF_HOME environment variable to: {os.environ['HF_HOME']}")
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(HF_CACHE_DIR, 'transformers')

    print(f"Starting server LLM with model: {model_name}...")
    command = [
        "bash", "-c",
        f"CUDA_VISIBLE_DEVICES=0,2 vllm serve {model_name} "
        "--dtype auto --api-key token-abc123 "
        f"--tensor-parallel-size 2 --enforce-eager --port {port}"
        + (f" --chat-template {chat_template}" if chat_template else "")
    ]

    # Launch inside screen
    subprocess.run(["screen", "-dmS", "llm_server"] + command)

    print("Waiting until server is ready...")
    while True:
        try:
            headers = {"Authorization": "Bearer token-abc123"}
            response = requests.get(f"http://localhost:{port}/v1/models", headers=headers)
            if response.status_code == 200:
                print("Server is ready!")
                break
        except requests.RequestException:
            pass

        screen_output = os.popen("screen -list").read()
        if "llm_server" not in screen_output:
            print("Error: The server process is not running!")
            return False

        print("Server not available yet, waiting 5 seconds...")
        time.sleep(5)
    return True

def get_prompts(ocr_text, key, prompt_dir="/home/pfont/files/LLM_eval/data/prompt_texts"):
    prompts = {}
    for filename in os.listdir(prompt_dir):
        filepath = os.path.join(prompt_dir, filename)
        prompt_key = filename.split(".")[0]
        prompt_text = open(filepath).read()
        prompts[prompt_key] = prompt_text + ocr_text
    return prompts[key]
    

def end_llm_server():
    print("Terminating server...")
    os.system("screen -XS llm_server quit")
    print("Experiment completed.")

def query_llm(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
