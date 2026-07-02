from .base import BaseAgent


class CoderAgent(BaseAgent):
    name = "Coder Agent"
    system_prompt = """You are a research coding agent. Your role is to:
- Write Python code for data analysis, statistics, and visualization
- Execute code to verify results and generate numerical insights
- Create charts, graphs, and data summaries as needed
- Debug and refine code iteratively

Always explain what the code does and interpret the results clearly."""

    def get_tools(self) -> list:
        return [{"type": "code_execution_20260120", "name": "code_execution"}]
