"""Custom slash commands manager."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

COMMANDS_FILE = "commands.json"

# Built-in command names that cannot be overridden
RESERVED_COMMANDS = frozenset({"start", "new", "stop", "restart", "help"})


class CommandsManager:
    """Load, save, and manage custom slash commands from workspace/commands.json."""

    def __init__(self, workspace: Path | str) -> None:
        self._path = Path(workspace) / COMMANDS_FILE
        self._commands: dict[str, dict[str, Any]] = {}
        self._mtime: float = 0.0
        self._load()

    def _load(self) -> None:
        """Load commands.json if it exists."""
        if not self._path.exists():
            self._commands = {}
            self._mtime = 0.0
            return
        try:
            mtime = os.path.getmtime(self._path)
            if mtime == self._mtime and self._commands:
                return  # already up to date
            with open(self._path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("commands.json is not a dict, ignoring")
                data = {}
            self._commands = data
            self._mtime = mtime
            logger.info("Loaded {} custom commands from {}", len(data), self._path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load commands.json: {}", e)
            self._commands = {}

    def _save(self) -> None:
        """Write commands.json to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._commands, f, indent=2, ensure_ascii=False)
        self._mtime = os.path.getmtime(self._path)

    def _reload_if_changed(self) -> None:
        """Reload from disk if file was modified externally."""
        if self._path.exists():
            mtime = os.path.getmtime(self._path)
            if mtime != self._mtime:
                self._load()

    def get(self, name: str) -> dict[str, Any] | None:
        """Get a single command definition by name."""
        self._reload_if_changed()
        return self._commands.get(name)

    def list_commands(self) -> dict[str, dict[str, Any]]:
        """Return all custom commands."""
        self._reload_if_changed()
        return dict(self._commands)

    def add(
        self,
        name: str,
        mode: str,
        description: str,
        script: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Add or update a custom command. Returns status message."""
        name = name.lower().strip().lstrip("/")
        if name in RESERVED_COMMANDS:
            return f"Error: /{name} is a reserved command"
        if mode not in ("script", "agent", "mixed"):
            return f"Error: mode must be script, agent, or mixed"
        if mode in ("script", "mixed") and not script:
            return f"Error: script path required for {mode} mode"
        if mode in ("agent", "mixed") and not prompt:
            return f"Error: prompt required for {mode} mode"

        cmd: dict[str, Any] = {"mode": mode, "description": description}
        if script:
            cmd["script"] = script
        if prompt:
            cmd["prompt"] = prompt

        is_update = name in self._commands
        self._commands[name] = cmd
        self._save()
        action = "Updated" if is_update else "Added"
        return f"{action} command /{name} ({mode} mode)"

    def remove(self, name: str) -> str:
        """Remove a custom command. Returns status message."""
        name = name.lower().strip().lstrip("/")
        if name not in self._commands:
            return f"Error: command /{name} not found"
        del self._commands[name]
        self._save()
        return f"Removed command /{name}"

    def list_for_menu(self) -> list[dict[str, str]]:
        """Return commands formatted for Telegram bot menu."""
        self._reload_if_changed()
        return [
            {"name": name, "description": cmd.get("description", "")}
            for name, cmd in self._commands.items()
        ]
