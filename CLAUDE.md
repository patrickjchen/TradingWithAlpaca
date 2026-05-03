# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`alpaca_codegen.py` is an AG2 (autogen) multi-agent script that **generates** the files under `generated/`. The generated files (`subscriber.py`, `trader.py`, `cli.py`, `trade_stream.py`, `persister.py`, `webgui.py`, `web/index.html`) are the actual Alpaca crypto trading pipeline; they are committed but are LLM output, regenerable via the codegen. Most stages emit Python (verified by `py_compile`+import in a temp dir); the `webgui_html` stage emits a standalone HTML/CSS/JS file (verified by a sanity check — has `<!DOCTYPE`, closes `</html>`, non-trivial size).

When fixing a bug in the runtime pipeline, you generally have two options:
1. **Edit `generated/<file>.py` directly** — fastest, fine for one-off fixes.
2. **Edit the relevant `*_SYS` prompt in `alpaca_codegen.py` and regenerate** — appropriate when the issue is a pattern the LLM keeps regressing on. The "Durable rules" block in `CODER_PREAMBLE` exists specifically to capture lessons learned (e.g. `CryptoDataStream.run()` is sync, no `CryptoExchange` enum, Redis HSET rejects None). Add to it rather than re-fixing the same bug each regeneration.

## Running the codegen

```bash
python alpaca_codegen.py                       # generate any missing files in generated/
python alpaca_codegen.py --stage all           # regenerate everything, overwriting
python alpaca_codegen.py --stage subscriber    # regenerate one stage
python alpaca_codegen.py --stage trader BTC/USD ETH/USD   # custom symbol list
python alpaca_codegen.py --force-research      # also redo RESEARCH.md
python alpaca_codegen.py --fix err.txt --stage cli        # feed a runtime traceback back to the Coder for one stage
```

Stages are ordered (`STAGES` list in `alpaca_codegen.py`); later stages receive earlier ones as `### EXISTING:` context. Each stage retries up to `MAX_RETRIES=5` times, feeding compile/import errors back to the Coder. On final failure the last attempt is still written to disk for inspection.

The codegen uses Gemini via OpenRouter. Key resolution order: `/home/mark/KAGGLE/AIMO3/openrouter-keys` file → `$OPENROUTER_API_KEY`. Override the model with `GEMINI_MODEL=...` (default `google/gemini-3.1-pro-preview`).

## Running the generated pipeline

The pipeline is split into independent long-running workers; each can be restarted independently because Redis buffers events between them.

```bash
pip install -r requirements.txt
export ALPACA_API_KEY_ID=...  ALPACA_API_SECRET_KEY=...
# start redis (e.g. docker run -p 6379:6379 redis)

python generated/subscriber.py        # crypto market data websocket -> Redis (HSET/LPUSH/SADD)
python generated/trade_stream.py      # trading websocket -> Redis Stream alpaca:trade_updates
python generated/persister.py         # consumes the Redis Stream -> SQLite (alpaca.db)
python generated/webgui.py            # Flask UI on http://127.0.0.1:5000 (WEBGUI_HOST/WEBGUI_PORT to override)

python generated/cli.py symbols
python generated/cli.py latest BTC/USD
python generated/cli.py account
python generated/cli.py buy BTC/USD 0.001    # paper by default; --live to opt in
```

`cli.py` must be run from inside `generated/` (or with that as cwd) because it does `import trader`.

## Architecture: the two halves

**Half 1 — market data (read path).** `subscriber.py` opens `CryptoDataStream` and subscribes to trades, quotes, AND L2 orderbooks. It pushes each tick into Redis under several keys per symbol: `alpaca:latest:<SYM>` (HSET, top-of-book snapshot), `alpaca:recent:<SYM>:<trade|quote>` (LPUSH+LTRIM, last 100), `alpaca:orderbook:<SYM>` (HSET with JSON-encoded `bids`/`asks` arrays — the full L2 book each tick, since Alpaca delivers snapshots not deltas), `alpaca:symbols` (SADD, the universe). `cli.py`'s `symbols`/`latest`/`recent` subcommands read directly from Redis — no Alpaca call.

**Half 2 — trading (write path) and audit.** `trader.py` is a thin synchronous wrapper over `TradingClient` used by `cli.py`'s trading subcommands (`account`, `buy`, `sell`, `cancel`, ...). Independently, `trade_stream.py` subscribes to Alpaca's TradingStream websocket and `XADD`s every `TradeUpdate` to the Redis Stream `alpaca:trade_updates`. `persister.py` is a separate consumer-group reader that drains that stream into SQLite (`orders` upsert, `fills` insert-or-ignore on `execution_id`, `trade_events` append). The `trade_stream → Redis Stream → persister` split is deliberate: persister can crash and resume from the last acked id without losing events.

**UI — `webgui.py` + `web/index.html`.** Two separately-generated artifacts. `webgui.py` is a thin Flask **backend**: serves `web/index.html` as a static file at `/` and exposes JSON endpoints (`/api/symbols`, `/api/orderbook/<sym>`, `/api/orders`, `/api/fills`, `/api/summary`, `POST /api/order`, `DELETE /api/order/<id>`). It contains NO inline HTML — that lives in `web/index.html`, a self-contained vanilla-JS frontend (no React/CDN). The page has three tabs: (1) **Escalator** — per-symbol L2 depth ladder with exactly 10 ask rows above 10 bid rows (one price per row, padded with empties to keep the layout stable), polled from `/api/orderbook/<sym>` every 1s; clicking an ask row opens a BUY modal at that price, clicking a bid row opens a SELL modal. (2) **Activity** — orders + fills from SQLite via `/api/orders` and `/api/fills`. (3) **Summary** — account/positions/realized-PnL via `/api/summary`. The split was deliberate: combining backend + 500-line HTML in one LLM emission was timing out the codegen. The UI must NOT 500 when Redis is empty or `alpaca.db` is missing — those are normal startup states (`/api/orderbook` falls back to top-of-book from `alpaca:latest:<SYM>`, then to `{bids:[], asks:[], fallback:"no_data"}`).

## Paper-vs-live invariant

Paper trading is the default everywhere. `live=True` / `--live` is the explicit opt-in. `cli.py` prints a stderr warning before any `--live` action. Same `ALPACA_API_KEY_ID`/`ALPACA_API_SECRET_KEY` env vars work for both paper and live (paper-only keys also work for paper).

For crypto symbols use `TimeInForce.GTC` (or `IOC`) — `DAY` is rejected. Use `qty=` not `notional=`.

## Redis key/stream conventions

| Key/Stream                       | Type   | Writer            | Reader              |
|----------------------------------|--------|-------------------|---------------------|
| `alpaca:latest:<SYM>`            | Hash   | subscriber.py     | cli.py `latest`, webgui `/api/orderbook` fallback |
| `alpaca:recent:<SYM>:trade`      | List   | subscriber.py     | cli.py `recent`     |
| `alpaca:recent:<SYM>:quote`      | List   | subscriber.py     | cli.py `recent`     |
| `alpaca:orderbook:<SYM>`         | Hash   | subscriber.py     | webgui `/api/orderbook/<sym>` (metadata only) |
| `alpaca:ob:bids:<SYM>`           | Hash   | subscriber.py     | webgui `/api/orderbook/<sym>` |
| `alpaca:ob:asks:<SYM>`           | Hash   | subscriber.py     | webgui `/api/orderbook/<sym>` |
| `alpaca:symbols`                 | Set    | subscriber.py     | cli.py `symbols`, webgui `/api/symbols` |
| `alpaca:trade_updates`           | Stream | trade_stream.py   | persister.py        |
| `alpaca:recent:trade_updates`    | List   | trade_stream.py   | (debug only)        |

`alpaca:orderbook:<SYM>` HASH holds metadata only: `symbol`, `timestamp`. The actual L2 book lives in `alpaca:ob:bids:<SYM>` and `alpaca:ob:asks:<SYM>`, each a HASH of `<canonical price str> -> <size str>`. Subscriber **merges** deltas (Alpaca's crypto orderbook websocket delivers level changes, NOT full snapshots): `size > 0` → HSET that level, `size == 0` → HDEL that level. Webgui's `/api/orderbook` HGETALLs both sides, sorts (bids DESC by price, asks ASC), and slices to the top 10 per side.

Persister consumer group defaults: group `persister`, consumer `persister-1`. Override via `$PERSISTER_GROUP` / `$PERSISTER_CONSUMER`.

`$REDIS_URL` (default `redis://localhost:6379/0`) and `$ALPACA_DB_PATH` (default `./alpaca.db`) are honored by all workers.

## Verifying without real creds

The codegen verifier dispatches by stage language: `verify_python_file` runs `python -m py_compile` then `import` on each Python artifact in a temp dir, and `verify_html_file` does a sanity check on `web/index.html` (non-trivial size, leading `<!DOCTYPE`, closing `</html>`, no leftover code fence). This catches the common LLM mistakes (bad imports, top-level errors, truncated emission) without needing real Alpaca creds or a Redis server. The generated modules are written to **never** make network calls at import time — env reads and connections happen inside `main()` / `get_client()` only. Preserve that property when editing.
