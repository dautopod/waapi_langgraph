from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.utilities.requests import RequestsWrapper
from langgraph.prebuilt import ToolNode
from inkern_tools import create_request_generation_tool, create_api_executor_tools
from pydantic import BaseModel
from dotenv import load_dotenv
from inkern_tools import load_yaml_spec
import os
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
import sqlite3
import uuid

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

api_spec=load_yaml_spec("waapi.yaml")

#llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, api_key=OPENAI_API_KEY)

llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1", temperature=0.0)

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

api_executor_tools=create_api_executor_tools(requests_wrapper,llm,allow_dangerous_requests=True,allowed_operations=("GET","POST"))

# We are going "bind" all tools to the model
# We have the ACTUAL tools from above, but we also need a mock tool to ask a human
# Since `bind_tools` takes in tools but also just tool definitions,
# We can define a tool definition for `ask_human`
class AskHuman(BaseModel):
    """Ask the human a question. If any of the tools output specify that human input is required, this tool must be called to get human input."""
    question: str

request_generator_tool = create_request_generation_tool(api_spec, llm)

llm_with_tools = llm.bind_tools([api_planner_tool, request_generator_tool, AskHuman] + api_executor_tools)

tool_node = ToolNode([api_planner_tool, request_generator_tool] + api_executor_tools)

# Define the function that determines whether to continue or not
def should_continue(state:MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # If tool call is asking Human, we return that node
    # You could also add logic here to let some system know that there's something that requires Human input
    # For example, send a slack message, etc
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "tools"


# Define the function that calls the model
def call_model(state:MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# We define a fake node to ask the human
def ask_human(state:MessagesState):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    #location = interrupt("Please provide input:")
    location = input("Please provide input:")
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": location}]
    return {"messages": tool_message}

workflow = StateGraph(MessagesState)

# Define the three nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("ask_human", ask_human)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")

# After we get back the human response, we go back to the agent
workflow.add_edge("ask_human", "agent")

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)

checkpointer = SqliteSaver(conn)
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

user_input = input("Start from scratch? y/n: ")

if user_input == "y":
    config["configurable"]["thread_id"] = str(uuid.uuid4())
    print("Starting new graph with config", config)
    user_input = "Start by asking user their phone number and use it in all appropriate placeholders (e.g. name) in the API calls. Check if there is already a session with user's phone number. If yes, provide details of the session."
    for chunk in app.stream({"messages": [user_input]}, config, stream_mode="updates"):
        for k,v in chunk.items():
            if isinstance(v["messages"][0],dict):
                print(k,v)
            else:
                v["messages"][0].pretty_print()
else:
    thread_id = input("Enter thread id: ")
    config = {"configurable": {"thread_id": thread_id}}
    print(app.get_state(config).values["messages"][-1].pretty_print())
    user_input=input("Provide further input: ")
    for chunk in app.stream({"messages": [user_input]}, config, stream_mode="updates"):
        for k,v in chunk.items():
            if isinstance(v["messages"][0],dict):
                print(k,v)
            else:
                v["messages"][0].pretty_print()