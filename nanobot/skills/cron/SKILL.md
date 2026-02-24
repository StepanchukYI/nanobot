---
name: cron
description: Schedule reminders and recurring tasks.
---

# Cron

Use the `cron` tool to schedule reminders or recurring tasks.

## Three Modes

1. **Reminder** - message is sent directly to user
2. **Task** - message is a task description, agent executes and sends result
3. **One-time** - runs once at a specific time, then auto-deletes

## Examples

Fixed reminder:
```
cron(action="add", message="Time to take a break!", every_seconds=1200)
```

Dynamic task (agent executes each time):
```
cron(action="add", message="Check HKUDS/nanobot GitHub stars and report", every_seconds=600)
```

One-time scheduled task (compute ISO datetime from current time):
```
cron(action="add", message="Remind me about the meeting", at="<ISO datetime>")
```

Timezone-aware cron:
```
cron(action="add", message="Morning standup", cron_expr="0 9 * * 1-5", tz="America/Vancouver")
```

With named agent profile:
```
cron(action="add", message="Analyze trading risk", cron_expr="0 9 * * *", tz="Europe/Kyiv", agent="gerchik-trader")
```

List/remove:
```
cron(action="list")
cron(action="remove", job_id="abc123")
```

## Time Expressions

| User says | Parameters |
|-----------|------------|
| every 20 minutes | every_seconds: 1200 |
| every hour | every_seconds: 3600 |
| every day at 8am | cron_expr: "0 8 * * *" |
| weekdays at 5pm | cron_expr: "0 17 * * 1-5" |
| 9am Vancouver time daily | cron_expr: "0 9 * * *", tz: "America/Vancouver" |
| at a specific time | at: ISO datetime string (compute from current time) |

## Timezone

Use `tz` with `cron_expr` to schedule in a specific IANA timezone. Without `tz`, the server's local timezone is used.

## Agent Profiles

A cron job can run under a named agent profile defined in `nanobot.yml` / `config.json`. The profile overrides the system prompt and optionally the model for that specific job.

Define in `nanobot.yml`:
```yaml
agents:
  profiles:
    gerchik-trader:
      system_prompt: "You are a trading expert focused on risk management."
      model: google/gemini-2.0-flash-001  # optional
```

Or in `~/.nanobot/config.json`:
```json
{
  "agents": {
    "profiles": {
      "gerchik-trader": {
        "systemPrompt": "You are a trading expert focused on risk management.",
        "model": "google/gemini-2.0-flash-001"
      }
    }
  }
}
```

The profile's `systemPrompt` is prepended to the default system prompt when the job runs.
If no named profile is found, the `default` profile is used (if defined), otherwise the global defaults apply.

## CLI Commands

```bash
# List active jobs
nanobot cron list

# List all jobs including disabled
nanobot cron list --all

# Add a job (every N seconds)
nanobot cron add -n "Job Name" -m "Task message" --every 3600

# Add a job (cron expression)
nanobot cron add -n "Daily Report" -m "Summarize news" --cron "0 9 * * *" --tz "Europe/Kyiv"

# Add a job with a named agent profile
nanobot cron add -n "Trading Check" -m "Analyze risk" --cron "0 9 * * *" -a gerchik-trader

# Add a one-time job
nanobot cron add -n "Reminder" -m "Call dentist" --at "2026-03-01T10:00:00"

# Remove a job
nanobot cron remove <job-id>

# Enable / disable a job
nanobot cron enable <job-id>
nanobot cron enable <job-id> --disable

# Manually trigger a job
nanobot cron run <job-id>

# Force-run even if disabled
nanobot cron run <job-id> --force
```
