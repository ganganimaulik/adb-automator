---
goal: |
  Watch Tinder -- the website tinder.com, logged in inside Chrome (com.android.chrome) -- for messages only: send an
  opener to new matches who pass the profile check, and answer matches who greet back. Never swipe, like, pass or
  Super Like anyone -- there is no liking in this goal at all. Go straight to Tinder's Messages tab every pass and
  stay off the Tinder (swipe deck), Explore and Likes You tabs. Send `Hey 🔥` to each new match under "New matches"
  whose chat is still empty, but ONLY after their profile shows a real person and a distance under 50 km; then open
  each conversation whose newest message is theirs and reply only as the REPLY POLICY says. Finish on the Messages
  screen, never inside a chat.
title: Tinder
send_limits: like=0
---

# Tinder Automation Policy

Messages only. This policy never likes, passes, swipes on or Super Likes anyone.

## step 1. Get to Messages -- and never swipe
- Tinder is a web page in Chrome, not an app. `open_app com.android.chrome`: the Tinder tab is normally still open. Tap **Messages** in Tinder's bottom nav (the speech bubble, fourth from the left) to reach `tinder.com/app/matches`. Only if no Tinder tab is open at all, use `open_url https://tinder.com/app/matches` -- each `open_url` opens a new Chrome tab, so never use it just to change page.
- If the element list shows only Chrome's toolbar and a bare "Web View", the page has not loaded into the element list yet: wait one step and look again.
- **STRICT RULE — NO SWIPING, NO LIKES**: never open the **Tinder** tab (flame icon, the swipe deck), **Explore**, **Likes You** or the **Get Likes** tile. Never tap a heart, X, star, Super Like or Boost button anywhere. If you ever land on the swipe deck (`tinder.com/app/recs`), touch nothing there and tap **Messages** in the bottom nav.
- **NEVER SWIPE A CONVERSATION ROW SIDEWAYS**: each row on the Messages list slides over to reveal **Report** and **Unmatch**. Scroll that list only up and down. If a row is showing Report / Unmatch, do not tap anywhere on that row: tap "Reload this page" in Chrome's toolbar to close it. The only sideways swipe allowed is on the **New matches** row of tiles, to see more new matches.
- Dismiss upsells and prompts (Tinder Gold / Platinum, "Turn on notifications", Chrome's "Add to Home screen" or Translate) with their close X / "No thanks" / "Maybe later" / "Not now".
- If Tinder shows a login or sign-up screen, the session has expired: do not try to log in. Stop the pass and report it.

## step 2. New matches — check the profile, then send the opener
- This policy DOES open conversations, but only here: a new match in the **"New matches"** row at the top of Messages (the photo tiles, `"<name> <name>"` in the element list). Skip the first tile, **Get Likes** -- it is a paid feature, not a match. Swipe that row left to see further tiles; open each new match in turn.
- A new match's chat is EMPTY when it has no messages at all -- only "You Matched with <name>", how long ago, and their round photo; no `You:` or `<name>:` line anywhere. Only an empty chat can get the opener. If the chat already has ANY message -- theirs or yours -- never send `Hey 🔥` there; handle it under step 3 instead.
- **STRICT RULE — PROFILE CHECK BEFORE `Hey 🔥`**: before typing anything in an empty chat, tap the match's name/photo in the chat header to open their profile, and check BOTH of these. Send the opener only if both pass.
  - **REAL PERSON**: look at EVERY photo -- tap the right half of the photo to move to the next one; the thin bars across the top of the photo show how many there are. Pass only if the photos are of a real human being and at least one of them shows that person's face clearly. FAIL, and send nothing, if every photo is any of: a cartoon, avatar, drawing, anime or AI-generated-looking image; a face hidden, covered, turned away, cropped off or blurred out (e.g. hands over the face, only hair or a back view); scenery, objects, pets, food, text, quotes or a screenshot with no person; a celebrity, model, stock or watermarked photo that looks lifted from elsewhere. If you cannot tell, it FAILS.
  - **DISTANCE UNDER 50 KM**: scroll the profile down to **Essentials** and read the distance line. Pass only if it reads `<N> kilometers away` with N below 50 (`27 kilometers away` passes, `50 kilometers away` and `120 kilometers away` fail), or `Less than a kilometer away`. If it is in miles, it must be below 31 miles. If the profile shows no distance at all, it FAILS.
  - Close the profile with its **Back** button (the round black down-arrow at top-right -- scroll back up if it is off screen). This returns to the chat.
  - If either check fails, send nothing: go back to Messages with the red Back chevron and move to the next new match. Do not message them later in the pass. They will be checked again next pass, with the same result.
- When both checks pass: tap the message field (the input over "Type a message") and type exactly `Hey 🔥` via `input_text` with `confidence="low"`. On the next turn, confirm from the screenshot that the field reads exactly `Hey 🔥` — the element list shows emoji as `..`, so `Hey ..` is what to expect there — then tap **"Send message"** (the round Send button at the bottom-right). If the field is empty or reads anything else, do not send: re-type `Hey 🔥` and check again.
- **ONE MESSAGE ONLY**: once `Hey 🔥` is in the chat, send nothing more there until they answer.
- Never tap **GIF**, the heart **Like** buttons beside messages, **Match options** (three dots), or the **Safety Toolkit** shield.
- After sending, or deciding not to, go back to Messages with the red **Back** chevron (top-left) and open the next one.

## step 3. Conversations — the greeting rule
- On the Messages list, the element list gives each conversation only as `<name> Report Unmatch`; the preview of its newest message is in the screenshot, under the name. A preview starting with the small curved reply arrow (↩) is YOUR message: the match has not answered, so leave that row. Open each row whose preview has NO arrow -- its newest message is the match's.
- Open each such conversation:
  - **STRICT RULE: NEVER SEND DOUBLE REPLIES.**
  - **STRICT RULE: NEVER SEND SAME REPLY AGAIN.**
  - **STRICT RULE: READ WHOLE MESSAGE HISTORY BEFORE SENDING ANYTHING** (scroll up in the thread if it does not all fit).
  - **WHO SAID WHAT**: the element list puts `You:` before each of your messages and `<name>:` before each of theirs, and that is the authority. On the screenshot your messages are dark maroon bubbles on the right; theirs are light grey bubbles on the left. Oldest at the top, newest at the bottom. "Sent at <time>", "Sent" and "You matched with <name> on <date>" are not messages. Read emoji off the screenshot: the element list shows them as `..`.
  - If the newest message is yours, the match has not replied: send nothing and go back to Messages.
  - **STRICT RULE — GREETING GATE**: `up for movie at my place?` may be sent ONLY when the match's NEWEST incoming message is a greeting and nothing more. A greeting is the one and only thing that unlocks this reply.
    - Counts as a greeting: `hi`, `hii`, `hiii`, `hey`, `heyy`, `hello`, `helloo`, `heloo`, `yo`, `hey there`, `hi there` — in any casing, with repeated letters, with punctuation, with a trailing emoji (`Hii!`, `hey 😊`), or addressed to you by your first name (`hey <your name>`).
    - Does NOT count, and means SEND NOTHING: brush-offs and one-word closers (`nvm`, `nvmm`, `nevermind`, `k`, `ok`, `okk`, `lol`, `no thanks`, `not interested`, `byee`); declines of an earlier suggestion (`Maybe some other time!!`, `Hmm maybe later`, `I'm not interested srry`); questions, including greeting-shaped ones (`how are you?`, `what's up?`, `wyd`, `what do you do?`); statements; compliments; emoji-only or sticker-only messages; GIFs and images.
    - When the gate fails, do not type anything at all. Leave the chat via Back and move to the next conversation. This policy has no reply for a non-greeting: never compose, rephrase, soften, or improvise a different message to fill the gap.
    - If you cannot tell whether the newest incoming message is a greeting, treat it as NOT a greeting and skip the chat. Skipping costs nothing; a sent message cannot be unsent.
  - **A MATCH WHO WROTE FIRST**: if the match opened with a greeting and nothing has been sent since, the GREETING GATE applies as usual. `Hey 🔥` is only ever for an empty chat that passed the profile check (step 2).
  - **CHECK PREVIOUS REPLIES IN HISTORY**: read the entire conversation. If `up for movie at my place?` was ALREADY sent to this person — in any capitalisation, e.g. `Up for movie at my place?` — DO NOT send it again, whatever they said after it.
  - **LEFTOVER DRAFT CLEARING**: if the message field already holds text (anything other than its "Type a message" placeholder) that should not be sent — including in a chat that fails the GREETING GATE — clear it with `input_text` and empty text before leaving. Never send leftover draft text.
  - ONLY reply if ALL THREE hold: the match's newest message is theirs, it passes the GREETING GATE, and `up for movie at my place?` has not been sent to them before. If any one fails, send nothing and leave the chat.
  - When all three hold, type `up for movie at my place?` via `input_text` with `confidence="low"`. On the next turn, inspect the screenshot and confirm BOTH that the field reads exactly `up for movie at my place?` AND that the newest message above it is still their greeting, before tapping "Send message".
  - After sending, return to Messages.

## Safety & Boundaries
- **Never swipe, like, pass or Super Like anyone on Tinder.** This policy sends messages and nothing else.
- **Never swipe a conversation row sideways, and never tap Report, Unmatch or Block.**
- This policy authorises exactly two messages: `Hey 🔥` (to a new match whose chat is empty AND whose profile passed the real-person and under-50-km checks) and `up for movie at my place?` (after a greeting). Nothing else is ever typed.
- **Pre-Send Verification**: ALWAYS set `confidence="low"` when typing, and check the typed text in the screenshot before tapping "Send message". Never send with `press_enter`.
- Never double reply under any circumstances, and never send in a chat whose newest message is already yours.
- **Never send `up for movie at my place?` to a match whose newest incoming message is not a greeting.** No greeting means no message at all — not a softer version, not a different one, not a follow-up later in the same pass.
- Never use paid features: Super Like, Boost, Tinder Gold / Platinum / Plus, Likes You, Get Likes, Rewind.
- Never react to messages with the heart Like, never send GIFs or photos, and never use Match options.
- Never log in, sign up, or enter any credentials.
- Never share financial, personal contact, or banking details.
- Finish each pass on the Messages screen.
