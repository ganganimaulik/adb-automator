---
goal: |
  Watch Tinder -- the website tinder.com, logged in inside Chrome (com.android.chrome) -- and notify the owner
  (`notify`) of new matches and new messages, read straight off the Messages screen. Notifications only: never open
  a match or a conversation, never type, send or react to anything, and never swipe, like, pass or Super Like anyone.
  Go straight to Tinder's Messages tab every pass and stay off the Tinder (swipe deck), Explore and Likes You tabs.
  Notify each tile under "New matches" that carries Tinder's red new-match dot, and each conversation whose photo
  carries the red unread dot, with the message its preview shows. Finish on the Messages screen.
title: Tinder
send_limits: like=0, send=0, enter=0
---

# Tinder Automation Policy

Notifications only. Sending messages is switched off for now: this policy never types, sends or reacts to anything,
and never likes, passes, swipes on or Super Likes anyone. The owner answers from their own phone.

Everything this policy needs is on the Messages screen. Tinder marks what is new with a small red dot, and opening a
match or a conversation marks it seen and clears that dot -- the owner's own sign of what is new. So never open one.

## step 1. Get to Messages -- and never swipe
- Tinder is a web page in Chrome, not an app. `open_app com.android.chrome`: the Tinder tab is normally still open. Tap **Messages** in Tinder's bottom nav (the speech bubble, fourth from the left) to reach `tinder.com/app/matches`. Only if no Tinder tab is open at all, use `open_url https://tinder.com/app/matches` -- each `open_url` opens a new Chrome tab, so never use it just to change page.
- If the element list shows only Chrome's toolbar and a bare "Web View", the page has not loaded into the element list yet: wait one step and look again. The list of conversations can redraw a moment after Chrome comes to the front: if the screenshot and the element list name different conversations, wait one step and look again.
- **STRICT RULE — NO SWIPING, NO LIKES**: never open the **Tinder** tab (flame icon, the swipe deck), **Explore**, **Likes You** or the **Get Likes** tile. Never tap a heart, X, star, Super Like or Boost button anywhere. If you ever land on the swipe deck (`tinder.com/app/recs`), touch nothing there and tap **Messages** in the bottom nav.
- **NEVER SWIPE A CONVERSATION ROW SIDEWAYS**: each row on the Messages list slides over to reveal **Report** and **Unmatch**. Scroll that list only up and down. If a row is showing Report / Unmatch, do not tap anywhere on that row: tap "Reload this page" in Chrome's toolbar to close it. The only sideways swipe allowed is on the **New matches** row of tiles, to see more new matches.
- Dismiss upsells and prompts (Tinder Gold / Platinum, "Turn on notifications", Chrome's "Add to Home screen" or Translate) with their close X / "No thanks" / "Maybe later" / "Not now".
- **Chrome's own sign-in and sync screens** ("Make Chrome your own", "Sign in to Chrome", "Continue as <name>", "Turn on sync", "Save time, type less" / "sync your history and tabs"): always choose **Stay signed out**, **No thanks** or **Use without an account**. Never tap "Continue as …", "Yes, I'm in", "Turn on sync" or anything else that signs Chrome in, turns on sync or accepts terms.
- If Tinder shows a login or sign-up screen, the session has expired: do not try to log in. Stop the pass and report it.

## step 2. New matches — only the tiles with a dot
- The **"New matches"** row at the top of Messages lists EVERY match you have not messaged yet, and most of them are old. A match is new only while Tinder marks its tile with a **small red dot**. A tile without the dot is a match already seen: never notify it, however it looks. With no dotted tile there is no new match, and this step notifies nothing.
- Skip the first tile, **Get Likes** -- it is a paid feature, not a match. The rest are `"<name> <name>"` in the element list, or `"<name> <name> Verified!"` for one with the blue verified tick. The element list does not show the dot: look for it on the screenshot. The blue verified tick beside a name is NOT the dot.
- Notify each tile that has the dot, one line each, exactly in this form:
  `Tinder: new match: <name>`
  Never open a tile: opening one marks it seen and clears its dot.
- Check every tile on screen. Newer matches come first, so swipe the row left (the one sideways swipe allowed) to look at the next tiles only while the last tile on screen still has a dot.

## step 3. New messages — the rows with a red dot, read off the list
- A conversation with a message the owner has not read has a **small red dot on the right edge of its round photo**. That dot is the only sign of a new message. A row without it has nothing new -- even when the match wrote last and nobody has answered -- so leave it.
- The element list gives each conversation only as `<name> Report Unmatch`, with neither the dot nor the message: read both off the screenshot. The preview under the name is the newest message. A preview starting with the small curved reply arrow (↩) is the owner's own message.
- Notify each row that has the red dot, exactly in this form:
  `Tinder: new message from <name>: <the preview>`
  Copy the preview exactly as the screenshot shows it, emoji included; if it is cut off with "…", copy what is shown. A photo or GIF is `[photo]` or `[GIF]`.
- **Never open a conversation**: opening one marks it read and clears its dot, and the preview already holds what to notify. Type nothing, send nothing.
- Check every row: a conversation with a new message moves to the top of the list, but an older unread one can sit below a newer one the owner has already read. Scroll the list down (never sideways) only if it continues below the screen. Leave a row that Tinder's bottom nav cuts off: it is the oldest conversation, and a tap near it lands on **Likes You**.

## Notifications (`notify`)
The owner wants a notification on their phone for every new match and every new message a match sends -- and for nothing else. Use the `notify` action: its `text` takes one notification per line. The harness delivers each line once and skips any line it has delivered before, on this pass or an earlier one -- so notify every time a rule here applies, even about a dot an earlier pass may have notified, and never try to remember or work out what was already sent.
- **New match**: `Tinder: new match: <name>` -- one line per tile under "New matches" that carries the red dot (step 2). Never for a tile without the dot.
- **New message**: `Tinder: new message from <name>: <the preview>` -- one line per conversation whose photo carries the red unread dot (step 3). Never for a row without the dot.
- All of a pass's lines can go in one `notify`, one per line. Nothing else is ever notified: not profiles, not upsells, not the pass summary.
- `notify` changes nothing on the phone: carry on from the same screen afterwards.

## Safety & Boundaries
- **Never open a match tile or a conversation.** This policy reads the Messages screen and nothing else.
- **Never send, type or react to anything on Tinder.** This policy authorises no message at all: not `Hey 🔥`, not `up for movie at my place?`, not anything else. Never tap "Send message", never press enter in a message field, never use the heart Like on a message.
- **Never swipe, like, pass or Super Like anyone on Tinder.**
- **Never swipe a conversation row sideways, and never tap Report, Unmatch or Block.**
- `notify` is only ever for the owner: never type a notification into a chat, and never notify anything **Notifications** does not list.
- Never use paid features: Super Like, Boost, Tinder Gold / Platinum / Plus, Likes You, Get Likes, Rewind.
- Never log in, sign up, or enter any credentials -- on Tinder or in Chrome. Never sign Chrome in to an account, turn on Chrome sync, or accept Chrome's or Google's terms: those screens are declined with Stay signed out / No thanks (step 1).
- Never share financial, personal contact, or banking details.
- Finish each pass on the Messages screen.
