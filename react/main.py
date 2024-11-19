from dotenv import load_dotenv

load_dotenv()

from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph

from nodes import execute_tools, run_agent_reasoning_engine
from state import AgentState

AGENT_REASON = "agent_reason"
ACT = "act"


def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT


flow = StateGraph(AgentState)

flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, execute_tools)


flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    # print("Hello ReAct with LangGraph")
    res = app.invoke(
        input={
            # "input": "Predict the temperature in the next hour",
            # "input": "Predict the temperature in the next 24 hours",
            # "input": "what is the temperature",
            # "input": "What is the current light intensity?",
            # "input": "How strong is the sensor's signal, considering both RSSI and SNR?",
            # "input": "Is there a risk of fog formation based on the dew point and temperature?",
            # "input": "What is the health of the IoT sensor?",
            # "input": "is it day or night?",
            # "input": "According latest temperature reading from our sensors compare to the temperature reported by OpenWeatherMap, Do you think it's raining?",
            # "input": "Taking into account the current lux level, can you calculate the consumption by the end of the month?",
            "input": "Please get current temperature, predict the temperature in the next hour and evaluate the model's performance and decide if better to use OpenWeatherMap",
        }
    )
    print(res["agent_outcome"].return_values["output"])
