from .base import BaseAgent


class SearchAgent(BaseAgent):
    name = "Search Agent"
    system_prompt = """You are a specialized research search agent. Your role is to:
- Search the web for relevant papers, articles, data, and current information
- Evaluate source credibility and relevance
- Extract and summarize key findings with clear citations
- Organize results by relevance and recency

Always provide structured output with source URLs."""

    def get_tools(self) -> list:
        return [{"type": "web_search_20260209", "name": "web_search"}]
