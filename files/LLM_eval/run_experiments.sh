#!/bin/bash
# Params
MODEL_NAME=$1  # Model name
CHAT_TEMPLATE=$2

if [ -z "$MODEL_NAME" ]; then
    echo "Error: You must give a model as an argument."
    exit 1
fi

echo "Starting server LLM with model: $MODEL_NAME..."
#CUDA_VISIBLE_DEVICES=1,2 vllm serve "$MODEL_NAME" --dtype auto --api-key token-abc123 \
#    --tensor-parallel-size 2 --enforce-eager --port 8000 &
#SERVER_PID=$! #Save server PID
chat=0

if [ -z "$CHAT_TEMPLATE" ]; then
    #Without chat template
    screen -dmS llm_server bash -c "CUDA_VISIBLE_DEVICES=1,2 vllm serve '$MODEL_NAME' --dtype auto --api-key token-abc123 --tensor-parallel-size 2 --enforce-eager --port 8000"
else 
    #With chat template
    echo "Using chat template $CHAT_TEMPLATE"
    chat=1
    screen -dmS llm_server bash -c "CUDA_VISIBLE_DEVICES=1,2 vllm serve $MODEL_NAME --chat-template $CHAT_TEMPLATE --dtype auto --api-key token-abc123 --tensor-parallel-size 2 --enforce-eager --port 8000"
fi

# Wait until server is ready
echo " Waiting until server is ready..."
until curl -s http://localhost:8000/v1/models > /dev/null; do
    if ! screen -list | grep -q "llm_server"; then
        echo "Error: The server process is not running!"
        exit 1
    fi
    echo "Server not available yet, waiting 5 seconds..."
    sleep 5
done

echo "Server ready. Starting experiment..."

python run_experiments.py --model "$MODEL_NAME" --chat $chat

echo "Terminating server..."
screen -XS llm_server quit # End server when finished

echo "Experiment completed with model: $MODEL_NAME"

