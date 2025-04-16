import json
import requests
from logging_config import LoggerConfig

# Initialize logger
logger = LoggerConfig.get_logger("llama_extract_categories")

# vLLM server URL
VLLM_URL = "http://localhost:8000/v1/chat/completions"

# Model path registered with vLLM
# LLAMA_MODEL_PATH = "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
model = "llama-3.1-8b-instruct"
def extract_entities(sentence):
    # Construct request payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a named entity recognition assistant. Extract entities from text and return only a structured JSON object."
            },
            {
                "role": "user",
                "content": f"""
                Identify only the following categories in the sentence:
                - **cuisine**
                - **meals**
                - **pricing category**
                - **features**

                **Task**:
                - Extract the exact words/phrases from the sentence that match these categories.
                - If a category is not found, return an empty list `[]`.

                **Sentence**: "{sentence}"

                **Return only JSON. No explanations, no extra text.**
                """
            }
        ],
        "max_tokens": 128,
        "temperature": 0.1,
        "top_p": 0.9
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(VLLM_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        result = response.json()
        content = result['choices'][0]['message']['content'].strip()

        # Attempt to parse content as JSON
        try:
            parsed = json.loads(content)
            logger.debug(f"Extracted JSON: {parsed}")
            return parsed
        except json.JSONDecodeError:
            logger.warning(f"Model output was not valid JSON: {content}")
            return {
                "cuisine": [],
                "meals": [],
                "pricing category": [],
                "features": []
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Request failed: {e}")
        return {
            "cuisine": [],
            "meals": [],
            "pricing category": [],
            "features": []
        }
