import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# 1. Get credentials
wx_project_id = os.getenv("WX_PRODUCT_ID")
wx_api_key = os.getenv("WX_API_KEY")

credentials = Credentials(
    api_key=wx_api_key,
    url="https://us-south.ml.cloud.ibm.com"  # Change region if needed
)

# 2. Choose a foundation model and initialize Model
model_id = "ibm/granite-3-8b-instruct"  # Or any from the supported list

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=wx_project_id,
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 100,
        "min_new_tokens": 1,
        "repetition_penalty": 1.0
    }
)

# 3. Prepare your input and generate text
input_text = "What is the capital of France?"
generated_response = model.generate_text(prompt=input_text)

print(f"Prompt: {input_text}")
print(f"Generated text: {generated_response}")