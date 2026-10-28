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
- **Skip chats already checked**:
  - Before opening any "Your turn" chat, look up every row on screen with ONE `read_notified`, one line per row: `Hinge: new message from <name>`.
  - A row's preview (the text after the name, without "1 day ago. Reply?") is the match's newest message.
  - If that message is already among the lines found for that person AND it is not one of the reply triggers below (greeting, where, living situation), an earlier pass already read that chat and sent nothing on purpose: do not open it.
  - Open every other row: a newest message that was never notified, or one that is a trigger (its reply never went out). When unsure, open it.
  - If `act` rejects `read_notified` as not an allowed action, skip this check and open the rows as before.
- **Reply Rules**:
  - Never send double replies under any circumstances.
  - Never send the same reply again to the same person.
  - Read the whole message history before sending anything.
  - If the match has not sent an incoming reply yet (only your initial `Hey 🔥` is present) or the newest message was sent by you, send nothing and leave.
- **Authorized Replies & Triggers**:
  - **1. Greeting**:
    - Trigger: Match's newest incoming message is a greeting and nothing more (e.g. `hi`, `hii`, `hiii`, `hey`, `heyy`, `hello`, `helloo`, `yo`, `hey there`, `hi there`, trailing emoji or punctuation allowed).
    - Authorized reply: `up for movie at my place?`
    - Send only if this movie proposal was not already sent in the thread.
  - **2. Where / Location Question**:
    - Trigger: Match asks where (e.g. `where?`, `where`, `where do you stay?`, `where is your place?`, `where do you live?`, `where at?`, `where in ahmedabad?`).
    - Authorized reply: `Got flat in Bodakdev.`
    - Send only if this location reply was not already sent in the thread.
  - **3. Living Situation (Alone / Flatmates / Family Question)**:
    - Trigger: Match asks if you live alone, with family, or about flatmates/roommates (e.g. `do you live alone?`, `do you live with family?`, `live with family?`, `with parents?`, `you live alone?`, `alone?`, `live alone?`, `who do you stay with?`, `do you have flatmates?`, `roommates?`).
    - Authorized reply: `2bhk, personal room. 1 flatmate, wont bother us.`
    - Send only if this living situation reply was not already sent in the thread.
- **Unmatched Messages (Send Nothing)**:
  - If the match's message does not match one of the authorized triggers above (e.g. general questions like `how are you?`, `wyd`, statements, compliments, brush-offs like `ok`, `lol`, declines like `Maybe some other time!!`, emoji/stickers/photos/GIFs/voice notes), or if the applicable reply was already sent: type nothing, clear any leftover draft, and leave the chat.

## Notifications (`notify`)
The owner wants a notification on their phone for every new message a match sends and for every reply you send:
- **New message**: In each chat opened where the match sent new message(s), notify every message the match sent after your most recent message, oldest first:
  `Hinge: new message from <name>: <their message>`
- **Reply sent**: Once your reply shows in the thread as sent:
  `Hinge: replied to <name>: <the reply you sent>`
- Nothing else is notified.

## Safety & Boundaries
- Never double reply.
- Never compose, rephrase, soften, or improvise messages: only send the exact authorized replies (`up for movie at my place?`, `Got flat in Bodakdev.`, or `2bhk, personal room. 1 flatmate, wont bother us.`) strictly when their respective trigger condition is met.
- Never use paid features (Boost, Roses, HingeX, Signals).
- Never share financial, personal contact, or banking details.
- Finish each pass on the main Discover or Matches list.
