import re
import yaml
from typing import Any, Dict, List, Literal, Sequence, cast

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, Tool
from pydantic import Field
from langchain_community.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain_community.utilities.requests import RequestsWrapper

Operation = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]

API_REQUEST_GENERATOR_TOOL_NAME = "request_generator"
API_REQUEST_GENERATOR_TOOL_DESCRIPTION = """
This tool generates API requests based on a plan of API calls and user metadata. 
Usage: {API_REQUEST_GENERATOR_TOOL_NAME}(plan, user_metadata)

- **Parameters**:
  1. **plan** (string): A comma-separated list of one or more API calls to be executed.
  2. **user_metadata** (dictionary): A dictionary containing user information, such as email, userid, phone number, etc.

Both parameters are required for the tool to function correctly. Ensure that you provide:
- A valid `plan` as a string.
- A valid `user_metadata` as a dictionary.

Example Usage:
{API_REQUEST_GENERATOR_TOOL_NAME}("api_call_1,api_call_2", {"email": "user@example.com", "userid": "12345", "phone": "123-456-7890"})
"""

API_CONTROLLER_DISPLAY_ONLY_PROMPT = """You are an agent that receives a sequence of API calls, their documentation and user metadata. Your task is to generate the API requests based on the provided information. You should produce the API request details, including the endpoint, method, headers, and body (if applicable), in a structured format.

Identify all of the relevant keys in the request body and replace them with the corresponding values from the user metadata. If the user metadata does not contain a key, you should replace it with a placeholder value so user can provide it.

Here is documentation on the API:
Base url: {api_url}
Endpoints:
{api_docs}

Your outputs should follow this format:

Request: the details of the API request (structured format)

Begin!

Plan: {plan}
UserMetadata: {user_metadata}
"""

def create_request_generation_tool(
    api_spec: ReducedOpenAPISpec,
    llm: BaseLanguageModel
) -> Any:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.chains.llm import LLMChain

    api_url = api_spec.servers[0]["url"]  # TODO: do better.
    def _create_and_run_request_generator_agent(plan_str: str, user_metadata_dict: dict) -> str:
        pattern = r"\b(GET|POST|PATCH|DELETE|PUT)\s+(/\S+)*"
        matches = re.findall(pattern, plan_str)
        endpoint_names = [
            "{method} {route}".format(method=method, route=route.split("?")[0])
            for method, route in matches
        ]
        docs_str = ""
        for endpoint_name in endpoint_names:
            found_match = False
            for name, _, docs in api_spec.endpoints:
                regex_name = re.compile(re.sub("\\{.*?\\}", ".*", name))
                if regex_name.match(endpoint_name):
                    found_match = True
                    docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n"
            if not found_match:
                raise ValueError(f"{endpoint_name} endpoint does not exist.")

        prompt = PromptTemplate(
            template=API_CONTROLLER_DISPLAY_ONLY_PROMPT,
            input_variables=["plan","user_metadata"],
            partial_variables={
                "api_url": api_url,
                "api_docs": docs_str
            },
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(plan_str, user_metadata_dict)
    
    return Tool(
        name=API_REQUEST_GENERATOR_TOOL_NAME,
        func=_create_and_run_request_generator_agent,
        description=API_REQUEST_GENERATOR_TOOL_DESCRIPTION,
    )