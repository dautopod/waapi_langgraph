# %%
import os
import yaml
from langchain_community.agent_toolkits.openapi import planner
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities.requests import RequestsWrapper
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
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

api_spec=load_yaml_spec("waapi.yaml")

llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, api_key=OPENAI_API_KEY)
api_planner_tool = planner._create_api_planner_tool(api_spec, llm)

# Set headers for the requests
headers = {"X-API-KEY": os.getenv("WAAPI_API_KEY")}
requests_wrapper = RequestsWrapper(headers=headers)

api_controller_tool=planner._create_api_controller_tool(
            api_spec,
            requests_wrapper,
            llm,
            allow_dangerous_requests=True,
            allowed_operations=("GET", "POST")
        )

#llm_with_tools = llm.bind_tools([api_planner_tool, api_controller_tool])
llm_with_tools = llm.bind_tools([api_planner_tool])
#response=llm_with_tools.invoke("I want to sign up for this service. What APIs should I call?")

#tool_node = ToolNode([api_planner_tool, api_controller_tool])
tool_node = ToolNode([api_planner_tool])
#tool_node.invoke({"messages": [llm_with_tools.invoke("I want to sign up for this service. What APIs should I call?")]})

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()

config = {"configurable": {"thread_id": "1"}}

#I want to sign up for this service. Which APIs should I call?
user_input = input("Enter your message: ")

all_messages = [("human", user_input)]
while user_input:
    aiMessage=None
    for chunk in app.stream(
        {"messages": all_messages}, config, stream_mode="values"
    ):        
        #print(chunk["messages"][-1])
        if 'finish_reason' in chunk["messages"][-1].response_metadata and chunk["messages"][-1].response_metadata['finish_reason']=='stop':
            chunk["messages"][-1].pretty_print()
        if chunk["messages"][-1].content:
            aiMessage=chunk["messages"][-1].content
    all_messages.append(("ai", aiMessage))
    user_input = input("Enter your message: ")
    all_messages.append(("human", user_input))