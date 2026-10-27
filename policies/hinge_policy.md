---
goal: |
  Watch Hinge (co.hinge.app): reply to matches who have written back, notify the owner (`notify`) of every new message
  a match sends and every reply you send, then like new profiles on Discover.
  Check Matches first every pass — someone waiting on a reply matters more than another like. Then work
  Discover and like between 108 and 207 new profiles per pass (minimum 108, up to 207 cap), continuing until
  that quota is satisfied, Discover shows "You've seen everyone for now", or the step budget is exhausted.
  Never stop early after only a few likes. Leave the phone on the Discover or Matches list, never inside a profile or a chat.
title: Hinge
snooze: You've seen everyone for now = 15m
send_limits: like=207, rose=0
---

# Hinge Automation Policy

## step 1. Discover Quota & Profile Liking
- **MANDATORY PASS QUOTA (108–207 LIKES)**:
  - Continuously like new Discover profiles until reaching at least 108 likes (up to 207 cap per pass).
  - Never stop or report done early after only a few likes.
  - Stop conditions: quota reached (108-207), Discover displays "You've seen everyone for now", or step budget exhausted.
- **Liking Criteria**:
  - Like only the first photo of each new profile.
  - Comment text to send: `Hey 🔥`.
  - Process all new cards; do not skip profiles simply because of shared first names.
  - Never review skipped profiles.

## step 2. Matches & Incoming Messages (Reply Policy)
- Check Matches before liking. Only open conversations under "Your turn".
- **Reply Rules**:
  - Never send double replies under any circumstances.
  - Never send the same reply again to the same person.
  - Read the whole message history before sending anything.
  - If the match has not sent an incoming reply yet (only your initial `Hey 🔥` is present) or the newest message was sent by you, send nothing and leave.
- **The Greeting Gate**:
  - Authorized reply text: `up for movie at my place?`
  - May ONLY be sent when the match's newest incoming message is a greeting and nothing more.
    - Counts as a greeting: `hi`, `hii`, `hiii`, `hey`, `heyy`, `hello`, `helloo`, `yo`, `hey there`, `hi there` (any casing, repeated letters, punctuation, trailing emoji).
    - Does NOT count (send nothing): questions (`how are you?`, `what's up?`, `wyd`), statements, compliments, brush-offs (`ok`, `k`, `lol`, `nvm`), declines (`Maybe some other time!!`), emoji/stickers/photos/GIFs/voice notes.
  - When the gate fails or movie proposal was already sent: type nothing, clear any leftover draft, and leave the chat.

## Notifications (`notify`)
The owner wants a notification on their phone for every new message a match sends and for every reply you send:
- **New message**: In each chat opened where the match sent new message(s), notify every message the match sent after your most recent message, oldest first:
  `Hinge: new message from <name>: <their message>`
- **Reply sent**: Once your reply shows in the thread as sent:
  `Hinge: replied to <name>: <the reply you sent>`
- Nothing else is notified.

## Safety & Boundaries
- Never double reply.
- Never compose, rephrase, soften, or improvise messages: `up for movie at my place?` triggered by a greeting is the only authorized reply.
- Never use paid features (Boost, Roses, HingeX, Signals).
- Never share financial, personal contact, or banking details.
- Finish each pass on the main Discover or Matches list.
