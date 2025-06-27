import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM

load_dotenv()
if os.getenv("WATSONX_API_KEY") and not os.getenv("WATSONX_APIKEY"):
    os.environ["WATSONX_APIKEY"] = os.getenv("WATSONX_API_KEY")

analysis_llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct",
    url=os.getenv("WATSONX_ENDPOINT", "https://us-south.ml.cloud.ibm.com"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params={"max_new_tokens": 256, "temperature": 0.2, "top_p": 1},
) 