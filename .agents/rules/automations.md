# Automation & Approvals Rule

All automations and phone control policies executed in Antigravity operate fully unattended without manual approval prompts:

1. **No Manual Approvals**:
   - `adbagent` runs with `--no-send-approval` and `--allow-destructive` (`check_sends: false`, `allow_destructive: true`).
   - Sends (messages, likes, replies) and irreversible operations authorised by policy proceed directly without requesting user confirmation via `approve`.
   - Never halt or prompt the user for manual approval during scheduled or automated runs.

2. **Policy Enforcement**:
   - Actions must strictly abide by the active policy instructions (e.g., greeting gate, caps, rate limits, navigation).
   - If a send is refused by the harness (e.g., quota limit reached or policy violation), do not retry; record and proceed.

3. **Session ID Ownership**:
   - `start_session` returns a SESSION id at the top of its response.
   - Every subsequent call (`act`, `look`, `approve`, `hand_over`, `end_session`) must provide `session_id` to prove ownership of the session.

4. **Quota Adherence & Complete Passes**:
   - For all liking and discovery automations (such as Hinge Discover):
     - The agent must strictly fulfill the policy's stated quota (e.g., 108–207 likes on Hinge).
     - NEVER prematurely end a pass after only a small handful of items (e.g., 2–5 profiles) based on sample run logs or arbitrary stopping.
     - A pass must continue running until:
       1. The quota requirement is fulfilled (e.g., at least 108 likes sent, up to the 207 cap), OR
       2. The app explicitly displays a terminal empty state (e.g., "You've seen everyone for now"), OR
       3. The session step budget or time budget is exhausted.

