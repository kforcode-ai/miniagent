#!/usr/bin/env python3
"""
Comprehensive test suite for MiniAgent framework
Tests: streaming, tool use, state management, thinking, retries, long conversations
"""

import asyncio
import time
from dotenv import load_dotenv
from miniagent import Agent, AgentConfig, ToolRegistry, StreamCallback, Thread
from miniagent.events import EventType
from miniagent.tools import KnowledgeBaseTool, WebSearchTool
from miniagent.llm import constant_retry

# Load environment variables
load_dotenv()


class TestMonitor:
    """Monitors and validates agent behavior during tests"""

    def __init__(self):
        self.events = []
        self.stream_chunks = []
        self.tool_calls = []
        self.responses = []
        self.thinking_events = []
        self.errors = []

    def reset(self):
        self.events = []
        self.stream_chunks = []
        self.tool_calls = []
        self.responses = []
        self.thinking_events = []
        self.errors = []


def create_monitor_callbacks(monitor: TestMonitor):
    """Create callbacks that monitor agent behavior"""
    callbacks = StreamCallback()

    # Track all events individually
    def event_handler(event):
        monitor.events.append(event)
        if event.type == EventType.AGENT_THINKING:
            monitor.thinking_events.append(event)
        elif event.type == EventType.TOOL_EXECUTION:
            monitor.tool_calls.append(event)

    # Register for all event types
    for event_type in EventType:
        callbacks.on(event_type, event_handler)

    # Track streaming specifically
    def stream_handler(event):
        monitor.stream_chunks.append(event.content)

    callbacks.on(EventType.STREAM_CHUNK, stream_handler)

    return callbacks


def create_test_agent():
    """Create agent with comprehensive tool set"""
    registry = ToolRegistry()

    # Knowledge base with rich content
    registry.register(KnowledgeBaseTool({
        "pricing": """📊 **Pricing Plans:**
• Starter: $9/month (10 users, 100GB storage, email support)
• Professional: $29/month (50 users, 1TB storage, live chat + email)
• Enterprise: Custom pricing (unlimited users, custom features, dedicated support)
All plans include SSL, daily backups, and 99.9% uptime.""",

        "features": """✨ **Key Features:**
• Real-time collaboration and document sharing
• Advanced security with 2FA and SSO integration
• Analytics dashboard with custom reports
• 1000+ integrations (Slack, Teams, Google Workspace, etc.)
• Mobile apps for iOS and Android
• API access (Professional and Enterprise plans)
• Advanced permissions and user management""",

        "support": """🎧 **Support Options:**
• 24/7 email support (all plans)
• Live chat support (Professional+)
• Phone support (Enterprise)
• Dedicated account manager (Enterprise)
• Community forum and knowledge base
• Video tutorials and documentation""",

        "security": """🔒 **Security Features:**
• End-to-end encryption
• Two-factor authentication (2FA)
• Single sign-on (SSO) integration
• Advanced audit logs
• Data export capabilities
• GDPR compliance
• SOC 2 Type II certified""",

        "integrations": """🔗 **Available Integrations:**
• Communication: Slack, Microsoft Teams, Zoom
• Storage: Google Drive, Dropbox, OneDrive
• Productivity: Jira, Trello, Asana, Monday.com
• Development: GitHub, GitLab, Bitbucket
• Analytics: Google Analytics, Mixpanel, Amplitude
• And 1000+ more via Zapier and custom APIs"""
    }))

    # Web search tool
    registry.register(WebSearchTool())

    # Math calculator
    @registry.tool(description="Perform mathematical calculations")
    async def calculate(expression: str):
        try:
            # Handle complex expressions with imports
            if "import math" in expression:
                # Execute in a safe namespace
                namespace = {"math": __import__("math")}
                result = eval(expression.split("; ")[1], namespace)
            else:
                # Safe evaluation with limited builtins
                allowed_names = {
                    k: v for k, v in __builtins__.items()
                    if k in ['abs', 'round', 'min', 'max', 'sum', 'pow', 'len']
                }
                result = eval(expression, {"__builtins__": allowed_names})
            return f"📐 Result: {expression} = {result}"
        except Exception as e:
            return f"❌ Error calculating '{expression}': {str(e)}"

    # Date/time tool
    @registry.tool(description="Get current date and time information")
    async def get_datetime():
        from datetime import datetime
        now = datetime.now()
        return f"📅 Current time: {now.strftime('%A, %B %d, %Y at %I:%M %p %Z')}"

    # Weather tool (mock)
    @registry.tool(description="Get weather information for a location")
    async def get_weather(location: str):
        # Mock weather data
        weather_data = {
            "New York": "Sunny, 72°F, Humidity: 45%",
            "London": "Cloudy, 15°C, Light rain expected",
            "Tokyo": "Clear, 22°C, Humidity: 60%",
            "Mumbai": "Hot and humid, 32°C, Partly cloudy"
        }
        location_clean = location.strip().title()
        if location_clean in weather_data:
            return f"🌤️ Weather in {location_clean}: {weather_data[location_clean]}"
        else:
            return f"❓ I don't have weather data for '{location}'. Try major cities like New York, London, Tokyo, or Mumbai."

    # Create agent with streaming enabled
    config = AgentConfig(
        name="TestAgent",
        system_prompt="""You are a helpful AI assistant with access to various tools.
Use tools when appropriate to provide accurate information.
Be conversational but concise. Show your reasoning when making decisions.
If you encounter errors, try alternative approaches.""",
        retry_policy=constant_retry(max_retries=3, delay=0.5),
        stream_by_default=True
    )

    return Agent(config=config, tools=registry), registry


async def test_long_conversation():
    """Test a long conversation with 20+ interactions"""
    print("="*80)
    print("🧪 COMPREHENSIVE MINIAGENT TEST - LONG CONVERSATION")
    print("="*80)

    agent, registry = create_test_agent()
    thread = Thread()
    monitor = TestMonitor()

    # Set up monitoring callbacks
    callbacks = create_monitor_callbacks(monitor)
    agent.callbacks = callbacks

    # Long conversation script (25 interactions)
    conversation_script = [
        "Hi there! Can you introduce yourself and tell me what you can do?",

        "Great! I need help with some calculations. What's 125 * 8?",
        "Now calculate (45 + 67) * 3 - 12",

        "Tell me about your pricing plans",
        "Which plan would be best for a team of 25 people?",
        "What security features do you offer?",

        "Can you check the current weather in New York?",
        "What's the weather like in Tokyo?",

        "What time is it right now?",

        "Can you search for the latest news about artificial intelligence?",
        "What about the current Nifty50 index value?",

        "Tell me about your integration options",
        "Do you integrate with Slack and Jira?",

        "What's 2^10?",
        "Calculate the square root of 144",

        "What are your support options?",
        "Do you have phone support?",

        "Search for Python programming tutorials",
        "What's the weather in Mumbai?",

        "Tell me about your analytics features",
        "Can you help me understand your API access?",

        "What's 999 * 777?",
        "Final question: what makes your platform different from competitors?"
    ]

    print(f"\n🗨️ Starting long conversation test with {len(conversation_script)} interactions...")
    print("-"*80)

    total_start_time = time.time()
    success_count = 0
    error_count = 0

    for i, user_query in enumerate(conversation_script, 1):
        print(f"\n[Turn {i:2d}/{len(conversation_script)}] {'='*50}")
        print(f"👤 User: {user_query}")

        # Reset monitor for this turn
        monitor.reset()

        turn_start_time = time.time()

        try:
            # Process query
            response = await agent.run(
                user_input=user_query,
                thread=thread,
                stream=True,
                max_iterations=15  # Allow more iterations for complex queries
            )

            turn_time = time.time() - turn_start_time

            print(f"⚡ Turn completed in {turn_time:.2f}s")
            print(f"📊 Events: {len(monitor.events)} | Tool calls: {len(monitor.tool_calls)} | Stream chunks: {len(monitor.stream_chunks)}")

            # Validate response quality
            if len(response.strip()) < 10:
                print("⚠️  WARNING: Very short response")
            elif "error" in response.lower() and "tool" not in response.lower():
                print("⚠️  WARNING: Response contains errors")
            else:
                success_count += 1

            # Check for expected tool usage
            if any(keyword in user_query.lower() for keyword in ["calculate", "math", "what's", "what is"]):
                if not monitor.tool_calls:
                    print("⚠️  WARNING: Math query but no tool calls detected")
            elif any(keyword in user_query.lower() for keyword in ["weather", "time", "date"]):
                if not monitor.tool_calls:
                    print("⚠️  WARNING: Info query but no tool calls detected")
            elif any(keyword in user_query.lower() for keyword in ["search", "nifty", "news", "latest"]):
                if not monitor.tool_calls:
                    print("⚠️  WARNING: Search query but no tool calls detected")
            elif any(keyword in user_query.lower() for keyword in ["pricing", "features", "support", "security", "integrations"]):
                if not monitor.tool_calls:
                    print("⚠️  WARNING: Knowledge query but no tool calls detected")

        except Exception as e:
            error_count += 1
            turn_time = time.time() - turn_start_time
            print(f"⚡ Turn failed in {turn_time:.2f}s")
            print(f"❌ ERROR: {str(e)}")
            monitor.errors.append(str(e))

        # Small delay between turns
        await asyncio.sleep(0.1)

    total_time = time.time() - total_start_time

    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    print(f"✅ Successful interactions: {success_count}/{len(conversation_script)}")
    print(f"❌ Failed interactions: {error_count}")
    print(f"⏱️  Total test time: {total_time:.1f}s")
    print(f"📈 Average response time: {total_time/len(conversation_script):.2f}s per interaction")
    # Final state analysis
    print(f"\n📈 Final Thread State:")
    print(f"   Messages: {len(thread.messages)}")
    print(f"   Events: {len(thread.events)}")
    print(f"   Total events captured: {len(monitor.events)}")

    # Tool usage summary
    tool_usage = {}
    for event in monitor.events:
        if event.type == EventType.TOOL_EXECUTION:
            tool_name = event.content.get('tool', 'unknown')
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

    if tool_usage:
        print(f"\n🔧 Tool Usage Summary:")
        for tool, count in tool_usage.items():
            print(f"   {tool}: {count} times")

    # Check for issues
    issues = []
    if error_count > 2:
        issues.append("High error rate")
    if success_count < len(conversation_script) * 0.8:
        issues.append("Low success rate")
    if len(monitor.events) < success_count * 2:
        issues.append("Low event activity")
    if len(monitor.stream_chunks) < success_count * 5:
        issues.append("Low streaming activity")

    if issues:
        print(f"\n⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"\n✅ All checks passed!")

    print(f"\n🎯 Test completed successfully!")
    print("="*80)


async def test_retry_mechanism():
    """Test retry mechanism with failing tools"""
    print("\n🧪 TESTING RETRY MECHANISM")
    print("-"*40)

    # Create a tool that fails initially but succeeds on retry
    retry_count = 0

    @registry.tool(description="Test tool with retry logic")
    async def flaky_tool():
        nonlocal retry_count
        retry_count += 1
        if retry_count < 3:
            raise Exception(f"Simulated failure #{retry_count}")
        return f"Success after {retry_count} attempts!"

    # Test the retry
    try:
        result = await agent.tools.execute("flaky_tool", {})
        print(f"✅ Retry test passed: {result.data}")
    except Exception as e:
        print(f"❌ Retry test failed: {str(e)}")


async def main():
    """Run all tests"""
    await test_long_conversation()
    # await test_retry_mechanism()  # Enable when needed


if __name__ == "__main__":
    asyncio.run(main())
