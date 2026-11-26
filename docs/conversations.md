# How ySupport handles Discord conversations

## Automatic ticket replies

The bot normally collects messages for two seconds. The classifier checks the
request before Codex runs. Each turn gets the current request, response schema,
and tool map.

Ticket owners can add text or images to the active turn through the Codex app
server's `turn/steer`. Classifier and intake checks apply to the new input. The
bot removes it from the queue after Codex confirms receipt. If receipt is late
or uncertain, the bot drops the draft and skips the next collection delay.

An edit or deletion of active input invalidates the draft before delivery starts.
The bot retries with the revised input. Each transaction hash keeps the chain
identified by its explorer link or an explicit statement. A correction can
update the chain for one receipt.

After restart, the next ticket interaction restores unfinished input in a fresh
Codex session. The restart itself sends no reply.

## Staff takeover and private handoffs

A staff message pauses automatic Discord replies. A substantive response to an
open Telegram handoff sends one Discord update and closes the handoff. Automatic
replies stay paused.

**Dismiss** in Telegram retires the handoff. **Stop Bot** in Discord also
cancels queued or active delivery and releases the execution slot. For one bot
reply, staff can use `y:`. It reads the message staff reply to and up to four
recent user images from ticket history. Staff images carry a separate label.

The bot saves an accepted reply before acknowledging it. Independent workers
keep polling and dismissal available. Notice edits and deletion stay ordered
after canceled HTTP calls. Each notice names the channel, summarizes the bot's
answer, and gives the remaining team action.

The team update uses a fresh Codex session, recent text and images, and older
messages fetched by ID. Recorded screenshots take priority. The bot redrafts
once if context changes and saves a fingerprint before sending. Recovery and
staff follow-ups keep older history. Missing context and uncertain delivery go
to the team for review. Telegram notice cleanup retries independently.

## Public concerns

An authorized contributor starts or continues a public turn by replying with a
configured trigger.

Each public handoff links to the original Discord message and saves its own
question, findings, and attachments. The **Preparing** notice can accept one
substantive reply once the bot saves the context and sends its first Discord
reply. Registered early replies wait for delivery. The bot leaves earlier
updates unacknowledged so Telegram can retry them. Recovery skips the initial
Discord reply.

The bot checks the latest 50 messages for replies linked to the concern. It
uses current screenshots and redrafts once if the source or linked replies
change. Refreshing an image URL alone leaves the draft valid. If the team
already answered in Discord, the bot can close the handoff without another post.

The bot saves a content hash and the latest message ID before sending. Recovery
checks up to 50 later messages for an exact reply from the bot to the original
message. An uncertain send goes to the team for review. A stable Discord nonce
deduplicates HTTP retries.

Delivery, **Dismiss**, or confirmed deletion of the source message or channel
clears the retained conversation. A temporary or permission error keeps it
available for review. Notice cleanup retries independently, including on a
fresh trigger after restart. A confirmed missing notice completes cleanup.
Use **Dismiss** to retire a pending handoff from Telegram. Older alert-only
notices require a response in Discord.

See the [runtime and tool reference](reference.md) and [privacy policy](../PRIVACY.md).
