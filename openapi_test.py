import os
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.json.tool import JsonSpec
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi import toolkit

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_api_summary(spec: dict) -> str:
    """Extract a summary of available endpoints from the OpenAPI spec"""
    paths = spec.get("paths", {})
    summary = []
    for path, methods in paths.items():
        for method, details in methods.items():
            summary.append(f"- {method.upper()} {path}: {details.get('summary', 'No description')}")
    return "\n".join(summary)

# Load and process the OpenAPI spec with proper encoding
try:
    with open("WaapiSwagger.json", "r", encoding="utf-8") as f:
        raw_spec = json.load(f)
except UnicodeDecodeError:
    try:
        with open("WaapiSwagger.json", "r", encoding="utf-8-sig") as f:
            raw_spec = json.load(f)
    except UnicodeDecodeError:
        with open("WaapiSwagger.json", "r", encoding="latin-1") as f:
            raw_spec = json.load(f)

# Get API summary
api_summary = get_api_summary(raw_spec)

# Define the LangChain Agent with an improved prompt template
SYSTEM_PROMPT = """You are a helpful AI assistant that helps users understand and use APIs. You have access to the following API endpoints:

{api_summary}

When helping users, you should:
1. Recommend the most appropriate endpoints for their needs
2. Explain the required parameters and how to use them
3. Provide example usage where appropriate

Available tools:
{tools}

When you need to explore specific endpoint details, you can use the json_explorer tool.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
])

# Define the LangChain Agent
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, api_key=OPENAI_API_KEY)

# Define the State class
class State(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    tools_str: str = ""
    api_summary: str = ""

    class Config:
        arbitrary_types_allowed = True

# Initialize the graph builder
graph_builder = StateGraph(State)

# Set headers for the requests
headers = {"X-API-KEY": os.getenv("WAAPI_API_KEY")}
requests_wrapper = RequestsWrapper(headers=headers)

# Create JsonSpec from the loaded spec
json_spec = JsonSpec(dict_=raw_spec)

# Create the OpenAPI toolkit
openapitoolkit = toolkit.OpenAPIToolkit.from_llm(
    llm=llm,
    json_spec=json_spec,
    requests_wrapper=requests_wrapper,
    verbose=True,
    allow_dangerous_requests=True
)

# Get tools
tools = openapitoolkit.get_tools()
tools_map = {tool.name: tool for tool in tools}
tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

# Bind the tools to the LLM with the specific prompt
llm_with_tools = prompt | llm.bind_tools(tools)

# Define the chatbot node
def chatbot(state: State):
    response = llm_with_tools.invoke({
        "messages": state.messages,
        "tools": state.tools_str,
        "api_summary": state.api_summary
    })
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)

# Define the tool node
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add conditional logic to the graph
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "continue": "chatbot",
        "tool": "tools",
        END: END
    }
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Setup memory and compile the graph
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)

# Create a new state with the current conversation history
config = {"configurable": {"thread_id": "1"}}

def run_conversation():
    # Initialize conversation state
    conversation_messages = []
    
    while True:
        if not conversation_messages:
            user_input = "I want to sign up for this service. What APIs should I call?"
        else:
            user_input = input("\nEnter your message (or 'quit' to exit): ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
        
        # Add user message to conversation history
        user_message = HumanMessage(content=user_input)
        conversation_messages.append(user_message)
        
        # Create initial state with all required fields
        initial_state = {
            "messages": conversation_messages,
            "tools_str": tools_str,
            "api_summary": api_summary
        }
        
        # Stream responses
        events = graph.stream(initial_state, config, stream_mode="values")
        
        # Process and print events
        received_messages = []
        for event in events:
            if len(event.get("messages", [])) > 0:
                last_message = event["messages"][-1]
                
                # Only print and store messages we haven't seen before
                if last_message not in received_messages:
                    received_messages.append(last_message)
                    
                    if isinstance(last_message, AIMessage):
                        print("\n================================== AI Message ==================================")
                        print(last_message.content)
                    elif isinstance(last_message, HumanMessage):
                        print("\n================================ Human Message =================================")
                        print(last_message.content)
        
        # Update conversation history with AI responses
        conversation_messages.extend([msg for msg in received_messages if msg not in conversation_messages])

if __name__ == "__main__":
    try:
        print("Starting conversation (type 'quit' to exit)...\n")
        run_conversation()
    except KeyboardInterrupt:
        print("\nConversation ended by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nConversation ended.")