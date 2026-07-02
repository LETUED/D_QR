from __future__ import annotations
import anthropic


class BaseAgent:
    name: str = "Base Agent"
    model: str = "claude-sonnet-4-6"
    system_prompt: str = "You are a helpful AI assistant."
    max_tokens: int = 8192

    def __init__(self):
        self.client = anthropic.Anthropic()

    def get_tools(self) -> list:
        return []

    def execute_tool(self, name: str, inputs: dict) -> str:
        return f"Tool '{name}' not implemented"

    def run(self, task: str, context: str = "") -> str:
        content = f"Context:\n{context}\n\n---\n\nTask: {task}" if context else task
        messages: list = [{"role": "user", "content": content}]
        tools = self.get_tools()
        response = None

        while True:
            kwargs: dict = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": [{
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools

            response = self.client.messages.create(**kwargs)

            if response.stop_reason == "pause_turn":
                # Server-side tool hit iteration limit — append and continue
                messages.append({"role": "assistant", "content": response.content})
                continue

            if response.stop_reason == "tool_use":
                # Client-side tool use
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self.execute_tool(block.name, dict(block.input))
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            break  # end_turn, max_tokens, refusal, etc.

        if response is None:
            return ""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
