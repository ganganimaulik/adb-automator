# Automation & Approvals Rule

All automations and phone control policies executed in Antigravity operate fully unattended without manual approval prompts:

1. **No Manual Approvals**:
   - `adbagent` runs with `--no-send-approval` and `--allow-destructive` (`check_sends: false`, `allow_destructive: true`).
   - Sends (messages, likes, replies) and irreversible operations authorised by policy proceed directly without requesting user confirmation via `approve`.
   - Never halt or prompt the user for manual approval during scheduled or automated runs.

2. **Policy Enforcement**:
   - Actions must strictly abide by the active policy instructions (e.g., greeting gate, caps, rate limits, navigation).
   - If a send is refused by the harness (e.g., quota limit reached or policy violation), do not retry; record and proceed.
