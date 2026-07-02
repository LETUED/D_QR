from datetime import datetime
from .base import BaseAgent
from config import OUTPUT_DIR


class WriterAgent(BaseAgent):
    name = "Writer Agent"
    system_prompt = """You are a research writer. Your role is to:
- Synthesize findings from multiple research agents into a coherent report
- Structure content clearly: Introduction, Findings, Analysis, Conclusions
- Write in clear, precise language appropriate for the topic
- Include citations and highlight key insights
- Save the final report to a file

Always produce a complete, well-organized report and save it."""

    def get_tools(self) -> list:
        return [
            {
                "name": "save_report",
                "description": "Save the research report as a markdown file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Short descriptive filename without extension (e.g. 'qr_code_security')",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full report content in markdown format",
                        },
                    },
                    "required": ["filename", "content"],
                },
            }
        ]

    def execute_tool(self, name: str, inputs: dict) -> str:
        if name == "save_report":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = OUTPUT_DIR / f"{inputs['filename']}_{timestamp}.md"
            filepath.write_text(inputs["content"], encoding="utf-8")
            print(f"\n  [Writer] Report saved → {filepath}")
            return f"Report saved to {filepath}"
        return super().execute_tool(name, inputs)
