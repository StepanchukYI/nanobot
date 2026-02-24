"""Tests for named agent profiles (--agent flag for cron jobs)."""

import json
import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.config.schema import AgentProfile, AgentsConfig, Config
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule

runner = CliRunner()


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_agent_profile_schema():
    """AgentProfile stores system_prompt and optional model."""
    profile = AgentProfile(system_prompt="You are a trader.", model="google/gemini-2.0-flash-001")
    assert profile.system_prompt == "You are a trader."
    assert profile.model == "google/gemini-2.0-flash-001"


def test_agent_profile_no_model():
    """AgentProfile.model is optional."""
    profile = AgentProfile(system_prompt="You are a trader.")
    assert profile.model is None


def test_agents_config_profiles_default_empty():
    """profiles dict is empty by default."""
    cfg = AgentsConfig()
    assert cfg.profiles == {}


def test_agents_config_profiles_camelcase():
    """Config accepts camelCase keys (standard nanobot JSON format)."""
    cfg = AgentsConfig.model_validate({
        "profiles": {
            "gerchik-trader": {
                "systemPrompt": "You are a Gerchik trader.",
                "model": "google/gemini-2.0-flash-001",
            }
        }
    })
    assert "gerchik-trader" in cfg.profiles
    assert cfg.profiles["gerchik-trader"].system_prompt == "You are a Gerchik trader."


# ---------------------------------------------------------------------------
# CronService: add_job with agent
# ---------------------------------------------------------------------------


def test_add_job_stores_agent(tmp_path):
    """add_job persists the agent field to disk."""
    service = CronService(tmp_path / "cron" / "jobs.json")
    job = service.add_job(
        name="gerchik-15m",
        schedule=CronSchedule(kind="every", every_ms=900_000),
        message="Run trading cycle",
        agent="gerchik-trader",
    )
    assert job.payload.agent == "gerchik-trader"

    # Reload from disk and verify persistence
    service2 = CronService(tmp_path / "cron" / "jobs.json")
    jobs = service2.list_jobs(include_disabled=True)
    assert jobs[0].payload.agent == "gerchik-trader"


def test_add_job_agent_none_by_default(tmp_path):
    """Jobs without --agent default to agent=None."""
    service = CronService(tmp_path / "cron" / "jobs.json")
    job = service.add_job(
        name="no-agent",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
    )
    assert job.payload.agent is None


def test_add_job_roundtrip_json(tmp_path):
    """Serialised JSON contains the agent field."""
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path)
    service.add_job(
        name="test",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
        agent="swing-analyst",
    )
    data = json.loads(store_path.read_text())
    assert data["jobs"][0]["payload"]["agent"] == "swing-analyst"


# ---------------------------------------------------------------------------
# CLI: cron add --agent
# ---------------------------------------------------------------------------


def _make_config_with_profile(tmp_path) -> Config:
    """Return a Config that has a 'gerchik-trader' profile."""
    from nanobot.config.schema import AgentProfile
    cfg = Config()
    cfg.agents.profiles["gerchik-trader"] = AgentProfile(
        system_prompt="You are a Gerchik trader.",
        model="google/gemini-2.0-flash-001",
    )
    return cfg


def test_cron_add_with_agent_flag(monkeypatch, tmp_path):
    """cron add --agent stores the profile name in the job."""
    cfg = _make_config_with_profile(tmp_path)
    # load_config is imported locally inside cron_add, so patch at the loader level
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    result = runner.invoke(app, [
        "cron", "add",
        "--name", "gerchik-15m",
        "--message", "Run trading cycle",
        "--every", "900",
        "--agent", "gerchik-trader",
    ])

    assert result.exit_code == 0, result.stdout
    assert "gerchik-trader" in result.stdout
    assert "gerchik-15m" in result.stdout

    # Verify persisted
    store_path = tmp_path / "cron" / "jobs.json"
    data = json.loads(store_path.read_text())
    assert data["jobs"][0]["payload"]["agent"] == "gerchik-trader"


def test_cron_add_unknown_agent_exits(monkeypatch, tmp_path):
    """cron add --agent with unknown profile name exits with error."""
    cfg = Config()  # no profiles
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    result = runner.invoke(app, [
        "cron", "add",
        "--name", "bad",
        "--message", "hello",
        "--every", "60",
        "--agent", "nonexistent-profile",
    ])

    assert result.exit_code == 1
    assert "nonexistent-profile" in result.stdout
    assert not (tmp_path / "cron" / "jobs.json").exists()


def test_cron_add_without_agent_still_works(monkeypatch, tmp_path):
    """cron add without --agent works exactly as before."""
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)

    result = runner.invoke(app, [
        "cron", "add",
        "--name", "no-agent",
        "--message", "hello",
        "--every", "60",
    ])

    assert result.exit_code == 0, result.stdout
    data = json.loads((tmp_path / "cron" / "jobs.json").read_text())
    assert data["jobs"][0]["payload"]["agent"] is None


# ---------------------------------------------------------------------------
# Context: system_prompt_prefix
# ---------------------------------------------------------------------------


def test_build_messages_with_system_prompt_prefix(tmp_path):
    """system_prompt_prefix is prepended to the default system prompt."""
    from nanobot.agent.context import ContextBuilder

    ctx = ContextBuilder(tmp_path)
    messages = ctx.build_messages(
        history=[],
        current_message="Hello",
        system_prompt_prefix="## Agent: gerchik-trader\nYou are a Gerchik trader.",
    )

    system_msg = messages[0]
    assert system_msg["role"] == "system"
    assert system_msg["content"].startswith("## Agent: gerchik-trader\nYou are a Gerchik trader.")


def test_build_messages_without_prefix_unchanged(tmp_path):
    """build_messages without prefix behaves exactly as before."""
    from nanobot.agent.context import ContextBuilder

    ctx = ContextBuilder(tmp_path)
    messages_without = ctx.build_messages(history=[], current_message="Hi")
    messages_with_none = ctx.build_messages(history=[], current_message="Hi", system_prompt_prefix=None)

    assert messages_without[0]["content"] == messages_with_none[0]["content"]
