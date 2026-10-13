---
goal: |
  Watch Bumble (com.bumble.app) for messages only: send an opener to new matches, and answer matches who greet back.
  Never swipe, like, pass or SuperSwipe anyone -- there is no liking in this goal at all. Go straight to the Chats
  tab every pass and stay off the People, Discover and Liked You tabs. Send `Hey 🔥` to each new match under
  "Your matches" whose chat is still empty, then open each conversation marked "Your move" and reply only as the
  REPLY POLICY says. Finish on the Chats list, never inside a chat.
title: Bumble
send_limits: like=0, superswipe=0, compliment=0
---

# Bumble Automation Policy

Messages only. This policy never likes, passes or swipes on anyone.

## step 1. Get to Chats -- and never swipe
- Open Bumble (`com.bumble.app`). It usually opens on **People**, the swipe deck. Do not touch anything there: tap **Chats** in the bottom nav (`"Chats Chats"`, bottom-right) straight away.
- **STRICT RULE — NO SWIPING, NO LIKES**: never swipe left or right anywhere in Bumble, never tap a profile card, and never tap "Send SuperSwipe", "Send a compliment", "Recommend to a friend", "Filters" or "Adjust your filters". A swipe on a card is a like or a pass, and it cannot be taken back. If you ever find yourself on People, Discover, Liked You or a profile, tap **Chats** in the bottom nav and carry on from there.
- **"Bumble isn't responding"**: Android shows this dialog often on this phone, mostly when a chat opens or closes or when Bumble loses focus. Tap **Wait** — never "Close app" — and continue where you were.
- Ignore promo cards on the Chats list (e.g. "We're sunsetting Opening Moves...") — do not tap them.

## step 2. New matches — send the opener
- This policy DOES open conversations, but only here: a new match in the **"Your matches"** row at the top of Chats (the round avatars, `"<name>, Date, match"`). Open each one in turn.
- A new match's chat is EMPTY when it has no message bubbles at all — only their photo, "<N> hours left to message", and the "You get one message to start" tip. Only an empty chat gets the opener.
- In an empty chat, type exactly `Hey 🔥` into the "Aa" field via `input_text` with `confidence="low"`. On the next turn, confirm from the screenshot that the field reads exactly `Hey 🔥` — the element list shows emoji as `..`, so `Hey ..` is what to expect there — then tap **"Send message"** (the yellow arrow at the right end of the input row). If the field is empty or reads anything else, do not send: re-type `Hey 🔥` and check again.
- **ONE MESSAGE ONLY**: "You get one message to start" means Bumble takes one message before they reply. Once `Hey 🔥` is in the chat, send nothing more there until they answer.
- If the chat already has ANY message bubble — theirs or yours — it is not empty: never send `Hey 🔥` there. Handle it under step 3 instead.
- Never tap the topic chips under the tip ("Hidden talents", "Pet-peeves", "Next availability", ...), GIF, Camera, the video or voice message buttons, "Voice call", "Video call" or "Options".
- After sending, or deciding not to, go back to the Chats list with the Back chevron (top-left) and open the next one.

## step 3. Conversations — the greeting rule
- In the Chats list, a row marked **"Your move"** is a conversation whose newest message came from the match: open each of those. A row without "Your move" ends in your own message: leave it. Rows reading "Conversation expired..." are over: skip them, and never tap Rematch, Extend or anything paid to revive one. Ignore "Older chats (N)".
- Open each conversation:
  - **STRICT RULE: NEVER SEND DOUBLE REPLIES.**
  - **STRICT RULE: NEVER SEND SAME REPLY AGAIN.**
  - **STRICT RULE: READ WHOLE MESSAGE HISTORY BEFORE SENDING ANYTHING**
  - **WHO SAID WHAT**: your messages are YELLOW bubbles on the right; the match's are GREY bubbles on the left. The oldest is at the top and the newest at the bottom. "Delivered" or "Read" under a bubble is a status, and "Today" / "Yesterday" are dates — neither is a message. Read it off the screenshot: the element list shows emoji as `..`.
  - If the newest message is yours, the match has not replied: send nothing and go back to the Chats list.
  - **STRICT RULE — GREETING GATE**: `up for movie at my place?` may be sent ONLY when the match's NEWEST incoming message is a greeting and nothing more. A greeting is the one and only thing that unlocks this reply.
    - Counts as a greeting: `hi`, `hii`, `hiii`, `hey`, `heyy`, `hello`, `helloo`, `yo`, `hey there`, `hi there` — in any casing, with repeated letters, with punctuation, with a trailing emoji (`Hii!`, `hey 😊`), or addressed to you by your first name (`hey <your name>`).
    - Does NOT count, and means SEND NOTHING: brush-offs and one-word closers (`nvm`, `nvmm`, `nevermind`, `k`, `ok`, `lol`, `no thanks`, `not interested`); declines of an earlier suggestion (`Maybe some other time!!`, `Unfortunately I am here for just today.`); questions, including greeting-shaped ones (`how are you?`, `what's up?`, `wyd`, `what do you do?`); statements; compliments; emoji-only or sticker-only messages; GIFs, images, voice notes and video messages.
    - When the gate fails, do not type anything at all. Leave the chat via Back and move to the next conversation. This policy has no reply for a non-greeting: never compose, rephrase, soften, or improvise a different message to fill the gap.
    - If you cannot tell whether the newest incoming message is a greeting, treat it as NOT a greeting and skip the chat. Skipping costs nothing; a sent message cannot be unsent.
  - **A MATCH WHO WROTE FIRST**: either person can start a Bumble chat now. If the match opened with a greeting and nothing has been sent since, the GREETING GATE applies as usual. `Hey 🔥` is only ever for an empty chat (step 2).
  - **CHECK PREVIOUS REPLIES IN HISTORY**: read the entire conversation. If `up for movie at my place?` was ALREADY sent to this person — in any capitalisation, e.g. `Up for movie at my place?` — DO NOT send it again, whatever they said after it.
  - **LEFTOVER DRAFT CLEARING**: Bumble keeps unsent text in the "Aa" field, even across restarts. If the field already holds text (anything other than its "Aa" placeholder) that should not be sent — including in a chat that fails the GREETING GATE — clear it with `input_text` and empty text before leaving. Never send leftover draft text.
  - ONLY reply if ALL THREE hold: the match's newest message is theirs, it passes the GREETING GATE, and `up for movie at my place?` has not been sent to them before. If any one fails, send nothing and leave the chat.
  - When all three hold, type `up for movie at my place?` via `input_text` with `confidence="low"`. On the next turn, inspect the screenshot and confirm BOTH that the field reads exactly `up for movie at my place?` AND that the newest bubble above it is still their greeting, before tapping "Send message".
  - After sending, return to the Chats list.

## Safety & Boundaries
- **Never swipe, like, pass or SuperSwipe anyone on Bumble.** This policy sends messages and nothing else.
- This policy authorises exactly two messages: `Hey 🔥` (to a new match whose chat is empty) and `up for movie at my place?` (after a greeting). Nothing else is ever typed.
- **Pre-Send Verification**: ALWAYS set `confidence="low"` when typing, and check the typed text in the screenshot before tapping "Send message". Never send with `press_enter`.
- Never double reply under any circumstances, and never send in a chat whose newest message is already yours.
- **Never send `up for movie at my place?` to a match whose newest incoming message is not a greeting.** No greeting means no message at all — not a softer version, not a different one, not a follow-up later in the same pass.
- Never use paid features: SuperSwipe, Compliments, Spotlight, Premium, Boost, Extend, Rematch, Backtrack, Liked You.
- Never call or video call, never send voice or video messages, GIFs or photos, and never unmatch, report or block anyone.
- Never share financial, personal contact, or banking details.
- Finish each pass on the Chats list.
