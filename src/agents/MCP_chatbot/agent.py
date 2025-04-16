from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from src.settings import load_config
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.messages import SystemMessage

config = load_config()
openai_config = config["openai-llm"]
llm = AzureChatOpenAI(
            azure_deployment=openai_config["azure_deployment"],
            openai_api_version=openai_config["api_version"],
            azure_endpoint=openai_config["azure_endpoint"],
            api_key=openai_config["api_key"]
        )
    
@asynccontextmanager
async def make_graph():
    async with MultiServerMCPClient(
        {
            "Tools": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
        }
    ) as client:
        tools = client.get_tools()
        llm_with_tools = llm.bind_tools(tools)

        
        # System message
        sys_msg = SystemMessage(content="You are a helpful assistant that can use tools to answer questions.")

        # Creating Tool calling assistant node
        def assistant(state: MessagesState):
            return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
        

        workflow = StateGraph(MessagesState)

        workflow.add_node("assistant", assistant)
        workflow.add_node("tools", ToolNode(tools))

        workflow.add_edge(START, "assistant")
        workflow.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        workflow.add_edge("tools", "assistant")
        
        agent = workflow.compile()
        yield agent