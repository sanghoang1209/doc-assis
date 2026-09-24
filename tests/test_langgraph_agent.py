import pytest
from app.services.agent.graph.core import AgentGraph, create_agent_state_graph
from app.services.agent.legacy import RawAgentGraph

def test_langgraph_compilation():
    graph = create_agent_state_graph()
    assert graph is not None

def test_langgraph_agent_init():
    agent = AgentGraph(max_loops=5)
    assert agent.max_loops == 5
    assert agent.graph is not None

def test_legacy_raw_agent_import():
    raw_agent = RawAgentGraph(max_loops=5)
    assert raw_agent.max_loops == 5
