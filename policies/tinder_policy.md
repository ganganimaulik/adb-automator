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

Notifications only. Sending messages is switched off: never type, send, react, like, pass, swipe, or Super Like.
Everything is read directly from the Messages screen without opening chats (opening clears the owner's red dots).

## step 1. New Matches Notifications (Only Tiles with Red Dot)
- Check tiles under "New matches" on the Messages screen.
- Only notify tiles that carry Tinder's small red dot on the screenshot. Skip "Get Likes" and tiles without a dot.
- Notification format:
  `Tinder: new match: <name>`
- NEVER open or tap a match tile.

## step 2. New Messages Notifications (Only Rows with Red Dot)
- Check conversation rows under "Messages".
- Only notify rows whose photo carries Tinder's small red unread dot on the screenshot.
- Read preview text under the name off the screenshot.
- Notification format:
  `Tinder: new message from <name>: <the preview>`
- Copy preview text exactly as shown (emojis included, photos/GIFs as `[photo]` or `[GIF]`).
- NEVER open or tap a conversation row.

## Notifications (`notify`)
The owner wants notifications for every new match and new message:
- `Tinder: new match: <name>`
- `Tinder: new message from <name>: <the preview>`
- Batch notifications together where applicable. Nothing else is notified.

## Safety & Boundaries
- Never open any match tile or conversation.
- Never send, type, or react to anything on Tinder.
- Never swipe, like, pass, or Super Like on the Tinder swipe deck (tinder.com/app/recs), Explore, or Likes You.
- Never swipe conversation rows horizontally.
- Never sign Chrome in or enable Chrome sync; choose Stay signed out / No thanks.
- Never purchase paid features (Tinder Gold/Platinum/Plus, Boost, Super Like).
- Finish each pass on the Messages screen.
