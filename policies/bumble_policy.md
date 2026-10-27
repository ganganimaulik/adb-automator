---
goal: |
  Watch Bumble (com.bumble.app) for messages only: send an opener to new matches, and answer matches who greet back.
  Never swipe, like, pass or SuperSwipe anyone -- there is no liking in this goal at all. Go straight to the Chats
  tab every pass and stay off the People, Discover and Liked You tabs. Send `Hey 🔥` to each new match under
  "Your matches" whose chat is still empty, then open each conversation marked "Your move" and reply only as the
  REPLY POLICY says. Notify the owner (`notify`) of every new message a match sends and every reply you send.
  Finish on the Chats list, never inside a chat.
title: Bumble
send_limits: like=0, superswipe=0, compliment=0
---

# Bumble Automation Policy

Messages only. Never like, pass, swipe, or SuperSwipe anyone.

## step 1. Opener on New Matches
- Under "Your matches", check each match whose chat is completely empty (no previous messages).
- Opener message to send: `Hey 🔥`.
- Send exactly ONE message per new match until they reply. Never send `Hey 🔥` to a chat that already contains any message.

## step 2. Conversations ("Your move" Rows)
- Open conversations marked "Your move". Skip rows without "Your move" and expired conversations.
- **Reply Rules**:
  - Never send double replies.
  - Never send the same reply again to the same person.
  - Read entire message history before deciding.
  - If the newest message in the thread is yours, send nothing and leave.
- **The Greeting Gate**:
  - Authorized reply text: `up for movie at my place?`
  - May ONLY be sent when the match's newest incoming message is a greeting and nothing more.
    - Counts as a greeting: `hi`, `hii`, `hiii`, `hey`, `heyy`, `hello`, `helloo`, `yo`, `hey there`, `hi there` (any casing, repeated letters, punctuation, trailing emoji, or addressed to you by name).
    - Does NOT count (send nothing): questions (`how are you?`, `what's up?`, `wyd`), statements, compliments, brush-offs (`ok`, `k`, `lol`, `nvm`), declines (`Maybe some other time!!`), emoji-only/stickers/photos/GIFs/voice notes/video messages.
  - When the gate fails or movie proposal was already sent: send nothing, clear any leftover draft, and leave the chat.

## Notifications (`notify`)
The owner wants notifications on their phone:
- **New message**: In each conversation whose newest message is theirs, notify every message the match sent after your most recent message:
  `Bumble: new message from <name>: <their message>`
- **Reply sent**: When an authorized reply is sent:
  `Bumble: replied to <name>: <the reply you sent>`
- **Opener sent**: When `Hey 🔥` is sent to a new match:
  `Bumble: opener to <name>: Hey 🔥`
- Nothing else is notified.

## Safety & Boundaries
- Never swipe, like, pass or SuperSwipe on People, Liked You, or Discover tabs.
- Never double reply.
- Never compose, rephrase, soften, or improvise messages: `up for movie at my place?` triggered by a greeting is the only authorized reply.
- Never purchase paid features (Spotlight, SuperSwipe, Rematch, Extend, Bumble Boost/Premium).
- Never share financial, personal contact, or banking details.
- Finish each pass on the Chats list.
