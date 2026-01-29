Start the IT Support model using vLLM:

python -m vllm.entrypoints.openai.api_server \
  --model it-support-lora-model \
  --port 8001
