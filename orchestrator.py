import anthropic
from agents import SearchAgent, AnalystAgent, CoderAgent, WriterAgent
from config import ORCHESTRATOR_MODEL

_DELEGATE_TOOL = {
    "name": "delegate_to_agent",
    "description": """Delegate a specific subtask to a specialized research agent.

Agents:
- 'search': Web search for papers, articles, recent data, current events
- 'analyst': Synthesize findings, identify patterns, evaluate evidence quality
- 'coder': Write and execute Python for data analysis, stats, visualization
- 'writer': Write final structured report and save it to a file

Use multiple agents in sequence as needed. Always end with 'writer'.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "enum": ["search", "analyst", "coder", "writer"],
                "description": "Which agent to delegate to",
            },
            "task": {
                "type": "string",
                "description": "Specific, actionable task description for the agent",
            },
            "context": {
                "type": "string",
                "description": "Relevant context or results from previous agents (optional)",
            },
        },
        "required": ["agent", "task"],
    },
}

_SYSTEM_PROMPT = """You are a research orchestrator that coordinates specialized AI agents to answer complex research questions.

Your workflow:
1. Analyze the research question and plan which agents to involve
2. Delegate subtasks to the appropriate agents via delegate_to_agent
3. Pass relevant context between agents as research progresses
4. Ensure comprehensive coverage: search → analyze → (code if needed) → write report
5. Always conclude by delegating to 'writer' to produce a saved report

Be methodical. Use search first for factual grounding, then analyst for synthesis, coder for any quantitative work, and writer for the final output."""


class ResearchOrchestrator:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self._agent_classes = {
            "search": SearchAgent,
            "analyst": AnalystAgent,
            "coder": CoderAgent,
            "writer": WriterAgent,
        }

    def _delegate(self, agent_name: str, task: str, context: str = "") -> str:
        AgentClass = self._agent_classes.get(agent_name)
        if not AgentClass:
            return f"Unknown agent: {agent_name}"
        agent = AgentClass()
        print(f"\n  [{agent.name}] {task[:100]}{'...' if len(task) > 100 else ''}")
        return agent.run(task, context=context)

    def research(self, question: str) -> str:
        print(f"\nResearching: {question}")
        print("=" * 60)

        messages: list = [{"role": "user", "content": f"Research question: {question}"}]
        response = None

        while True:
            response = self.client.messages.create(
                model=ORCHESTRATOR_MODEL,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=[_DELEGATE_TOOL],
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use" and block.name == "delegate_to_agent":
                        inputs = dict(block.input)
                        result = self._delegate(
                            inputs["agent"],
                            inputs["task"],
                            inputs.get("context", ""),
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            break

        if response is None:
            return "No response generated."
        for block in response.content:
            if block.type == "text":
                return block.text
        return "Research complete. Check the outputs/ directory for the saved report."
