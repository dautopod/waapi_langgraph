import os
import yaml
from langchain_community.agent_toolkits.openapi import planner
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities.requests import RequestsWrapper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_yaml_spec(file_path:str):
    # Load Swagger Specification
    with open(file_path, "r", encoding='utf8') as f:
        raw_waapi_api_spec = yaml.load(f, Loader=yaml.Loader)

    # Add the host from environment variable
    waapi_host = os.getenv("WAAPI_HOST")
    if waapi_host:
        raw_waapi_api_spec["servers"] = [{"url": waapi_host}]

    # Reduce the OpenAPI spec
    return reduce_openapi_spec(raw_waapi_api_spec)


def get_open_api_toolkit():
    # Define the LangChain Agent
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, api_key=OPENAI_API_KEY)

    # Set headers for the requests
    headers = {"X-API-KEY": os.getenv("WAAPI_API_KEY")}
    requests_wrapper = RequestsWrapper(headers=headers)

    # NOTE: set allow_dangerous_requests manually for security concern https://python.langchain.com/docs/security
    waapi_agent = planner.create_openapi_agent(
        load_yaml_spec("waapi.yaml"),
        requests_wrapper,
        llm,
        allow_dangerous_requests=False,
    )

    return waapi_agent