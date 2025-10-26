"""
MiniAgent - A simple, powerful agent framework
Inspired by HICA and BAML, focusing on simplicity and readability
"""

from .core import Agent, Thread, AgentConfig
from .tools import Tool, ToolRegistry, ToolResult
from .llm import LLMClient, RetryPolicy, RetryStrategy
from .events import Event, EventType, StreamCallback

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Thread", 
    "AgentConfig",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "LLMClient",
    "RetryPolicy",
    "RetryStrategy",
    "Event",
    "EventType",
    "StreamCallback"
]
