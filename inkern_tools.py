import re
import yaml
import json
from typing import Any, Dict, List, Literal, Sequence, cast

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, Tool
from pydantic import BaseModel, Field
from langchain_community.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain_community.utilities.requests import RequestsWrapper

Operation = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]

API_REQUEST_GENERATOR_TOOL_NAME = "request_generator"
API_REQUEST_GENERATOR_TOOL_DESCRIPTION = """
This tool generates API requests based on a plan of API calls and user metadata.
Usage: {API_REQUEST_GENERATOR_TOOL_NAME}({"plan": <plan>, "user_metadata": <user_metadata>})

- **Parameters**:
  1. **plan** (string): A comma-separated list of one or more API calls to be executed. 
     Each call should follow the format: "<HTTP_METHOD> <endpoint>" (e.g., "POST /api/sessions").
  2. **user_metadata** (dictionary): A dictionary containing user information, such as email, userid, phone number, etc.

Both parameters must be provided as part of a single dictionary.

Example Usage:
{API_REQUEST_GENERATOR_TOOL_NAME}({
    "plan": "POST /api/users",
    "user_metadata": {
        "email": "user@example.com",
        "userid": "12345",
        "phone": "123-456-7890"
    }
})
"""

API_CONTROLLER_DISPLAY_ONLY_PROMPT = """You are an agent that receives a sequence of API calls, their documentation and user metadata. Your task is to generate the API requests based on the provided information. You should produce the API request details, including the endpoint, method, headers, and body (if applicable), in a structured format.

Identify all of the relevant keys in the request body and replace them with the corresponding values from the user metadata. If the user metadata does not contain a key, you should replace it with a placeholder value so user can provide it.

Once you have generated the request, list the keys and the values that you have replaced from user metadata and ask the user to confirm if the values need to be replaced.

Here is documentation on the API:
Base url: {api_url}
Endpoints:
{api_docs}

Your outputs should follow this format:

Request: the details of the API request (structured format). This should be a complete url conisting of the base url and the endpoint.

Values: the keys and values that you have replaced from user metadata.

Begin!

Plan: {plan}
UserMetadata: {user_metadata}
"""

API_EXECUTOR_TOOL_NAME = "api_executor"
API_EXECUTOR_TOOL_DESCRIPTION = f"Can be used to execute an API call, like {API_EXECUTOR_TOOL_NAME}(request)."


from langchain.tools import StructuredTool
from langchain.prompts import PromptTemplate
import re
import yaml

import os
import yaml
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.tools import BaseRequestsTool,BaseTool
from langchain_community.agent_toolkits.openapi.planner_prompt import (
    PARSING_DELETE_PROMPT,
    PARSING_GET_PROMPT,
    PARSING_PATCH_PROMPT,
    PARSING_POST_PROMPT,
    PARSING_PUT_PROMPT,
    REQUESTS_GET_TOOL_DESCRIPTION
)
from langchain_community.agent_toolkits.openapi.planner import (
    RequestsPostToolWithParsing,
    RequestsPutToolWithParsing,
    RequestsDeleteToolWithParsing,
    RequestsPatchToolWithParsing,
    MAX_RESPONSE_LENGTH,
    _get_default_llm_chain_factory
)

class RequestsGetToolWithParsing(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Requests GET tool with LLM-instructed extraction of truncated responses."""

    name: str = "requests_get"
    """Tool name."""
    description: str = REQUESTS_GET_TOOL_DESCRIPTION
    """Tool description."""
    response_length: int = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: Any = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_GET_PROMPT)
    )
    """LLMChain used to extract the response."""

    def _run(self, text: str) -> str:
        from langchain.output_parsers.json import parse_json_markdown

        try:
            data = parse_json_markdown(text)
        except json.JSONDecodeError as e:
            raise e
        data_params = data.get("params")
        response: str = cast(
            str, self.requests_wrapper.get(data["url"], params=data_params)
        )
        response = response[: self.response_length]
        return self.llm_chain.invoke(
            {"response":response, "instructions":data["output_instructions"]}
        )

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()
    
# Load environment variables
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

def create_request_generation_tool(
    api_spec: ReducedOpenAPISpec,
    llm: BaseLanguageModel
) -> Any:
    from langchain.chains.llm import LLMChain

    api_url = api_spec.servers[0]["url"]

    # Define the input schema using pydantic
    class RequestGeneratorInputs(BaseModel):
        plan: str
        user_metadata: dict

    def _create_and_run_request_generator_agent(**kwargs) -> str:
        inputs = RequestGeneratorInputs(**kwargs)  # Unpack keyword arguments into the schema

        plan_str = inputs.plan
        user_metadata_dict = inputs.user_metadata

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
            input_variables=["plan", "user_metadata"],
            partial_variables={
                "api_url": api_url,
                "api_docs": docs_str
            },
        )
        chain = prompt | llm
        return chain.invoke({"plan": plan_str, "user_metadata": user_metadata_dict})

    return StructuredTool(
        name=API_REQUEST_GENERATOR_TOOL_NAME,
        func=_create_and_run_request_generator_agent,
        description=API_REQUEST_GENERATOR_TOOL_DESCRIPTION,
        args_schema=RequestGeneratorInputs  # Use the pydantic schema
    )

def create_api_executor_tools(
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
    allow_dangerous_requests: bool,
    allowed_operations: Sequence[Operation],
) -> Any:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.chains.llm import LLMChain

    tools: List[BaseTool] = []
    if "GET" in allowed_operations:
        get_llm_chain = PARSING_GET_PROMPT | llm
        tools.append(
            RequestsGetToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=get_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "POST" in allowed_operations:
        post_llm_chain = PARSING_POST_PROMPT | llm
        tools.append(
            RequestsPostToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=post_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "PUT" in allowed_operations:
        put_llm_chain = PARSING_PUT_PROMPT | llm
        tools.append(
            RequestsPutToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=put_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "DELETE" in allowed_operations:
        delete_llm_chain = PARSING_DELETE_PROMPT | llm
        tools.append(
            RequestsDeleteToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=delete_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "PATCH" in allowed_operations:
        patch_llm_chain = PARSING_PATCH_PROMPT | llm
        tools.append(
            RequestsPatchToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=patch_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    return tools