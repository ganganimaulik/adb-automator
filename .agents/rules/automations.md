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
     - **MANDATORY HARD COMPLETION GATE**: The agent is STRICTLY FORBIDDEN from calling `adbagent/end_session` or advancing to the next policy pass until ONE of the following 3 conditions is genuinely satisfied:
       1. The quota requirement is fulfilled (e.g., at least 108 likes sent, up to the 207 cap), OR
       2. The app explicitly displays a terminal empty state (e.g., "You've seen everyone for now"), OR
       3. The session step budget or time budget is within 20 steps of exhaustion (e.g., step >= 850 of 950).
     - NEVER prematurely end a pass after only a small handful of items (e.g., 2–5 profiles) to advance through multi-task queues or save turns. Any premature call to `end_session` before satisfying one of the 3 conditions above is a critical policy violation.

5. **Phone In Use (Busy Handling)**:
   - If `start_session` returns that the phone is in use by another session (e.g. "the phone is in use by another session..."), **IMMEDIATELY STOP AND EXIT THE CONVERSATION**.
   - Do NOT retry, do NOT sleep or poll, and do NOT inspect running processes, command lines, or system locks.
   - Conclude your response immediately stating that the phone is currently busy with an ongoing session and exit cleanly.
