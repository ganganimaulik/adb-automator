---
goal: |
  Watch Hinge (co.hinge.app): reply to matches who have written back, then like new profiles on Discover.
  Check Matches first every pass — someone waiting on a reply matters more than another like. Then work
  Discover and like at most 3 new profiles per pass, so a pass finishes cleanly instead of being cut off
  part-way through one. Leave the phone on the Discover or Matches list, never inside a profile or a chat.
title: Hinge
snooze: You've seen everyone for now = 15m
---

# Hinge Automation Policy

## step 1. Profile Exploration & Liking (Discover Tab)
- Go to the "Discover" tab in Hinge (`co.hinge.app`).
- **When Discover says "You've seen everyone for now"**, there is nobody left to like. Do not tap "Change filters" or "Review skipped profiles", and do not scroll, wait or retry: report done right there and leave the phone on that screen. The watch sees it and rests 15 minutes before the next pass.
- For each new profile:
  1. First go through the whole profile to its end in ONE action: `scroll_to_edge` with `direction="down"` on the profile's scroller. It flings until the profile stops moving, so it never falls short and needs no repeating. You don't have to read any profile info or images.
  2. **Fast Return to Top (1 action)**: `scroll_to_edge` with `direction="up"` on the same scroller flings back to the very first image/photo in one action. The fling in sub-step 1 left you at the **BOTTOM** of the profile, so this is not optional and not something the down-fling already did: until it has run, the screen is showing the profile's LAST photo.
  3. **STRICT RULE — Top Check Before Any Like**: You are at the first photo ONLY if the element list shows the profile's name and pronouns as separate text elements at top-left — e.g. `[Text] "Alex"` and `[Text] "she/her"`. That header renders only at the head of the card. If it is NOT in the element list, you are still mid-profile or at the bottom: run `scroll_to_edge` `up` again and check the list again. **Never tap a like button on a screen whose element list has no name header**, no matter what the screen looks like or what you believe about where you are.
  4. **Pick the Right Heart**: several like buttons (`"Like photo"`, `"Like photo prompt"`, `"Like prompt"`) are often listed at once — one per card currently on screen, and identical labels do NOT mean identical targets. The first photo's heart is the **topmost** one: prefer `@top-right` over `@bottom-right`. The lower heart belongs to the card below and likes the wrong photo.
  5. **Targeting & Typing Comment**: In the comment field, enter the text `Hey 🔥` via `input_text` with `confidence="low"`. If the element list only shows `#1 [Scroller]` (the full modal container), target `"Edit comment"` specifically so focus lands on the text entry box. Setting `confidence="low"` triggers an automatic screenshot and image analysis turn before hitting send.
  6. **Pre-Send Image Analysis Verification**: Inspect the screenshot / visual screen analysis on the next turn and check BOTH of these before sending:
     - **The comment**: the draft reads exactly `Hey 🔥` inside the comment field. If it is empty or does not match, DO NOT tap Send — re-tap `"Edit comment"`, re-type `Hey 🔥`, and check again.
     - **The photo**: the composer shows the media being liked, and Hinge draws a `1/N` media badge (e.g. `1/4`) on the profile's first photo. If a badge is visible and does not start with `1/`, the wrong photo is loaded: DO NOT tap Send. Back out with `press_key` back followed by `open_app co.hinge.app`, and start this profile again from sub-step 2. If no badge is visible at all, rely on the sub-step 3 check that already passed.

     Only tap "Send Priority Like" / "Send" (`confidence="high"`) after BOTH checks pass.
  7. Move on to the next profile.
  8. **STRICT RULE — Do Not Skip Profiles Based on Shared First Names**: Multiple distinct users on Hinge share common or single-letter names (e.g., "S", "M", "A"). When Hinge loads a new profile card on the Discover feed, process and send a like to it as normal. NEVER skip a profile card on Discover simply because its first name matches a name encountered earlier in the run, unless the visual photo and full profile content are proven to be an exact duplicate card loop.
  9. **STRICT RULE — Never Click "Review Skipped Profile"**: NEVER tap or click on "Review skipped profile", "Review skipped profiles", "Review skipped", or any button/banner/prompt offering to review skipped profiles. If such an option or dialog appears, ignore or dismiss it without tapping it.

## step 2. Matches & Incoming Messages (Matches / Chat Tab)
- Go to the "Matches" / chat tab in Hinge.
- Check for new matches or unread conversations.
- Open each conversation:
  - **STRICT RULE: NEVER SEND DOUBLE REPLIES.**
  - **STRICT RULE: NEVER SEND SAME REPLY AGAIN.**
  - **STRICT RULE: READ WHOLE MESSAGE HISTORY BEFORE SENDING ANYTHING**
  - **OUTGOING COMMENT / MATCH CARD IDENTIFICATION**:
    - The top photo card displaying your initial comment (`Hey 🔥`) or banner text ("Your Priority Like helped you stand out") is YOUR initial outgoing comment/like sent when discovering the profile.
    - `Hey 🔥` inside the photo card is YOUR message, NOT a reply from the match.
  - **INCOMING REPLY VERIFICATION**:
    - A match has ONLY replied if there is an actual incoming text bubble sent by the match (located below the opening photo card / initial comment).
    - If the only text present in the chat is your initial comment (`Hey 🔥`), or if the last message in the thread was sent by you, or if the match has NOT sent a reply message yet, DO NOT send any message. Immediately leave the chat and move to the next.
  - **STRICT RULE — GREETING GATE**: `up for movie at my place?` may be sent ONLY when the match's NEWEST incoming message is a greeting and nothing more. A greeting is the one and only thing that unlocks this reply.
    - Counts as a greeting: `hi`, `hii`, `hiii`, `hey`, `heyy`, `hello`, `helloo`, `yo`, `hey there`, `hi there` — in any casing, with repeated letters, with punctuation, or with a trailing emoji (`Hii!`, `hey 😊`).
    - Does NOT count, and means SEND NOTHING: brush-offs and one-word closers (`nvm`, `nvmm`, `nevermind`, `k`, `ok`, `lol`, `no thanks`, `not interested`); declines of an earlier suggestion (`Maybe some other time!!`); questions, including greeting-shaped ones (`how are you?`, `what's up?`, `wyd`, `what do you do?`); statements; compliments; emoji-only or sticker-only messages; GIFs, images, and voice notes.
    - When the gate fails, do not type anything at all. Leave the chat via `Back` and move to the next conversation. This policy has no reply for a non-greeting: never compose, rephrase, soften, or improvise a different message to fill the gap.
    - If you cannot tell whether the newest incoming message is a greeting, treat it as NOT a greeting and skip the chat. Skipping costs nothing; a sent message cannot be unsent.
  - **CHECK PREVIOUS REPLIES IN HISTORY**: Read the entire conversation history. If the proposed reply string (e.g. `up for movie at my place?`) was ALREADY sent to this person in a previous message turn, DO NOT send it again.
  - **LEFTOVER DRAFT CLEARING**: If the text input box (`id=messageComposition`) already contains text (or if the conversation list shows a `Draft:` label) for a reply that has already been sent or should not be sent — including a draft left in a chat that fails the GREETING GATE — execute `input_text` to CLEAR the field (or type empty text) before navigating away. Do NOT hit Send on duplicate draft text.
  - ONLY reply if ALL THREE hold: the match has sent a genuine new incoming message from their side, that message passes the GREETING GATE, and the reply text has not been sent to them before. If any one of the three fails, send nothing and leave the chat.
  - When all three conditions hold, type `up for movie at my place?` via `input_text` with `confidence="low"`. On the next turn, inspect the screenshot / image analysis and confirm BOTH that the typed message matches policy exactly AND that the newest incoming message above it is still a greeting, before tapping Send.
  - After sending, return to the matches list.

## Safety & Boundaries
- **Only ever like the first photo of a profile.** A like is irreversible and public. If you cannot establish that you are on the first photo — no name header in the element list, no way back to the top — restart hinge app and start again instead of guessing.
- **Pre-Send Verification**: ALWAYS set `confidence="low"` when typing comments or replies to trigger image analysis of the screen, and visually verify the typed message in the screenshot before tapping Send.
- **NEVER click on "Review skipped profile" / "Review skipped profiles"**: Never tap or click on "Review skipped profile", "Review skipped profiles", "Review skipped", or any button/prompt/card offering to review previously skipped profiles.
- Never double reply under any circumstances.
- **Never send `up for movie at my place?` to a match whose newest incoming message is not a greeting.** It is the only reply this policy authorises and a greeting is its only trigger. No greeting means no message at all — not a softer version, not a different one, not a follow-up later in the same pass.
- Never share financial, personal contact, or banking details.
- Finish each pass on the main Discover or Matches screen.

