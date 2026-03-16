"""Tool for managing custom slash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.commands import CommandsManager


class CommandTool(Tool):
    """Tool to manage custom slash commands (add, list, remove)."""

    def __init__(self, commands: CommandsManager) -> None:
        self._commands = commands

    @property
    def name(self) -> str:
        return "command"

    @property
    def description(self) -> str:
        return (
            "Manage custom slash commands. "
            "Commands can run scripts directly (script mode), "
            "inject prompts into the agent (agent mode), "
            "or run a script then pass output to the agent (mixed mode)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove"],
                    "description": "Action to perform",
                },
                "name": {
                    "type": "string",
                    "description": "Command name without / (e.g. 'camera')",
                },
                "mode": {
                    "type": "string",
                    "enum": ["script", "agent", "mixed"],
                    "description": (
                        "script: run script, send output directly (no LLM). "
                        "agent: inject prompt, agent processes. "
                        "mixed: run script, then agent processes the output."
                    ),
                },
                "script": {
                    "type": "string",
                    "description": "Script path relative to workspace (for script/mixed modes)",
                },
                "prompt": {
                    "type": "string",
                    "description": "Predefined prompt for the agent (for agent/mixed modes)",
                },
                "description": {
                    "type": "string",
                    "description": "Short description shown in /help and Telegram menu",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        name: str | None = None,
        mode: str | None = None,
        script: str | None = None,
        prompt: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "list":
            commands = self._commands.list_commands()
            if not commands:
                return "No custom commands defined."
            lines = []
            for cmd_name, cmd_def in commands.items():
                m = cmd_def.get("mode", "?")
                d = cmd_def.get("description", "")
                lines.append(f"/{cmd_name} — {d} [{m}]")
            return "\n".join(lines)

        if action == "add":
            if not name:
                return "Error: name is required for add"
            if not mode:
                return "Error: mode is required for add"
            return self._commands.add(
                name=name,
                mode=mode,
                description=description or "",
                script=script,
                prompt=prompt,
            )

        if action == "remove":
            if not name:
                return "Error: name is required for remove"
            return self._commands.remove(name)

        return f"Error: unknown action '{action}'"
