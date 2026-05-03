"""AG2-driven code generator for an Alpaca crypto -> Redis pipeline + trading CLI.

Three AG2 agents collaborate (Gemini 2.5 Flash via OpenRouter):

  1. Researcher (AssistantAgent)  -> writes `RESEARCH.md`: a brief on the
     Alpaca crypto websocket API AND the Alpaca trading REST API
     (TradingClient, MarketOrderRequest, OrderSide/TimeInForce, paper vs live).
  2. Coder     (AssistantAgent)  -> writes the pipeline files based on the brief:
        subscriber.py   -- subscribes to a few crypto symbols on Alpaca's
                           crypto websocket, publishes each tick into Redis
                           (latest snapshot per symbol + a capped recent list).
        trader.py       -- thin wrapper around `alpaca.trading.TradingClient`
                           exposing functions: account(), positions(),
                           orders(), buy_market(symbol, qty), sell_market(...),
                           buy_limit(...), sell_limit(...), cancel(order_id),
                           cancel_all(). Paper trading by default.
        cli.py          -- argparse CLI. Read-only Redis queries:
                           `latest`, `recent`, `symbols`. Trading subcommands
                           (proxy to trader.py): `account`, `positions`,
                           `orders`, `buy`, `sell`, `cancel`, `cancel-all`.
                           Trading subcommands accept `--live` to opt out of
                           paper mode (default is paper for safety).
        trade_stream.py -- subscribes to Alpaca's TRADING websocket and
                           publishes each TradeUpdate into the Redis Stream
                           `alpaca:trade_updates`. Decouples ingest from
                           durability so persister.py can be restarted.
        persister.py    -- consumer-group reader on the Redis Stream that
                           writes orders / fills / a full audit log into a
                           local SQLite db. No websocket of its own.
        webgui.py       -- Flask web UI: 3 tabs (Escalator with live
                           bid/ask + click-to-trade dialog; Orders & Fills
                           from SQLite; Summary with account/positions/PnL).
                           Reads Redis (subscriber's keys) and SQLite
                           (persister's tables); proxies orders via trader.
  3. Verifier  (UserProxyAgent + LocalCommandLineCodeExecutor) -> byte-compiles
     each generated file and imports it. We do NOT place live orders or open
     a websocket -- that requires real creds and a Redis server -- but a clean
     compile + import catches the common LLM mistakes.

Up to MAX_RETRIES rounds: on compile failure the error is fed back to the Coder.

Usage:
    python alpaca_codegen.py
        # writes RESEARCH.md + subscriber.py + trader.py + cli.py
        #        + trade_stream.py + persister.py + webgui.py into ./generated/

Then to actually use what was generated:
    pip install alpaca-py redis flask
    export ALPACA_API_KEY_ID=...  ALPACA_API_SECRET_KEY=...
    # start a redis server (e.g. `docker run -p 6379:6379 redis`)
    python generated/subscriber.py             # market data -> Redis
    python generated/trade_stream.py           # trade updates -> Redis Stream
    python generated/persister.py              # Redis Stream -> SQLite
    python generated/cli.py latest BTC/USD     # query market data
    python generated/cli.py account            # paper account balance
    python generated/cli.py buy BTC/USD 0.001  # paper market buy
    python generated/webgui.py                 # web UI on http://127.0.0.1:5000
"""

import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from autogen import AssistantAgent, LLMConfig, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "generated"

# Gemini via OpenRouter -- same pattern as ../gem.py.
# Override the model via $GEMINI_MODEL env var. Options on OpenRouter:
#   google/gemini-3.1-pro-preview   (default; strongest codegen)
#   google/gemini-2.5-pro           (older Pro; reliable)
#   google/gemini-2.5-flash         (cheap; struggles with subtle bugs)
OPENROUTER_KEY_FILE = Path("/home/mark/KAGGLE/AIMO3/openrouter-keys")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "google/gemini-3.1-pro-preview")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MAX_RETRIES = 5
SUBPROC_TIMEOUT = 30

# Default crypto symbols the generated subscriber should subscribe to.
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]


# ---------- LLM config ----------

def load_openrouter_key() -> str:
    if OPENROUTER_KEY_FILE.exists():
        return OPENROUTER_KEY_FILE.read_text().strip().splitlines()[0].strip()
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env.strip()
    raise RuntimeError(
        f"No OpenRouter key found at {OPENROUTER_KEY_FILE} or in $OPENROUTER_API_KEY"
    )


def gemini_llm_config(temperature: float | None = 0.2) -> dict:
    cfg = {
        "config_list": [{
            "model": GEMINI_MODEL,
            "api_key": load_openrouter_key(),
            "base_url": OPENROUTER_BASE,
            "api_type": "openai",
            # webgui.py is a Flask app with an inline HTML/CSS/JS template
            # and easily exceeds 8192 output tokens; truncated output has no
            # closing ``` fence and fails extract_block. Give plenty of room.
            "max_tokens": 32768,
        }],
    }
    if temperature is not None:
        cfg["temperature"] = temperature
    return cfg


# ---------- Agent prompts ----------

RESEARCHER_SYS = """You are a senior engineer documenting a third-party API for a
teammate who has never used it.

Produce a markdown brief covering BOTH halves of the Alpaca API that another
LLM coder will use as a spec. Two top-level sections:

## Part A -- Alpaca Crypto Market Data Websocket

  - The websocket URL for crypto market data (paper/live distinction, if any).
  - How auth is performed (the auth message shape with key/secret).
  - The subscribe message shape -- specifically how to subscribe to trades,
    quotes, AND L2 orderbooks for a list of symbols (e.g. `BTC/USD`). DO NOT
    cover bars -- we are intentionally NOT subscribing to bars.
  - The shape (field names + types) of the messages the server pushes for
    trades (`t`), quotes (`q`), and orderbooks (`o`). Include the symbol
    field name. For orderbooks, document that each message is a SNAPSHOT
    (the full book on each tick, not a delta) with `bids` and `asks` arrays
    of `{price, size}` levels. Skip bars.
  - The recommended Python SDK (`alpaca-py`) classes that wrap this:
    `CryptoDataStream` -- constructor args, `subscribe_trades` /
    `subscribe_quotes` / `subscribe_orderbooks`, async handler signatures
    (note the Orderbook model's `.bids` / `.asks` are lists of items with
    `.price` / `.size` attributes), `run()`.

## Part B -- Alpaca Trading REST API

  - The REST endpoints (paper vs live base URLs).
  - The relevant `alpaca.trading` SDK surface:
        `from alpaca.trading.client import TradingClient`
        `from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest`
        `from alpaca.trading.enums import OrderSide, TimeInForce`
    Cover constructor (`TradingClient(api_key, secret_key, paper=True)`),
    plus methods: `submit_order(order_data)`, `get_account()`,
    `get_all_positions()`, `get_orders()`, `cancel_order_by_id(order_id)`,
    `cancel_orders()`.
  - For crypto symbols (e.g. `BTC/USD`), what `time_in_force` values are
    supported (hint: crypto requires `GTC` or `IOC`, NOT `DAY`).
  - Quantity vs notional: when to use `qty=` vs `notional=` on order requests.
  - Required env vars: `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`. Paper
    trading uses the SAME keys as data (paper-only keys also work).

Strict output: a single fenced ```markdown``` block, with the two `## Part`
headers above. No prose outside the block. Be accurate -- if you are unsure
about a field name, say so explicitly with a `# TODO verify` comment rather
than guessing."""

CODER_PREAMBLE = """You are a senior Python engineer. You will be given:
  (1) a research brief on the Alpaca crypto websocket API AND trading REST API,
  (2) a list of crypto symbols (where relevant),
  (3) optionally, the contents of OTHER already-generated files as context
      -- do NOT modify them; just use their public surface,
  (4) optionally, compile/runtime errors from a previous attempt.

Output format: emit EXACTLY ONE fenced code block containing the full
source of the requested file. The block's language tag (```python or
```html) is specified per stage. No prose, no FILE markers, no extra
fenced blocks. For Python files, use only the stdlib plus `alpaca-py`,
`redis`, `flask`. Type-hint where useful but don't over-engineer. Both
`python -m py_compile` and `import <module>` must succeed for Python
artifacts.

Durable rules (do NOT regress between iterations):
  - alpaca-py async/sync (subscriber):
      * `CryptoDataStream.run()` is SYNC -- it calls `asyncio.run(...)` itself.
        NEVER `await stream.run()` and NEVER call it from inside another
        `asyncio.run()` -- it deadlocks. Use this pattern instead:
            async def main():
                ...
                stream.subscribe_trades(trade_handler, *symbols)
                stream.subscribe_quotes(quote_handler, *symbols)
                stream.subscribe_orderbooks(orderbook_handler, *symbols)
                try:
                    await stream._run_forever()   # the async coroutine
                finally:
                    await stream.close()          # async; safe to await
            asyncio.run(main())
      * `stream.stop()` is SYNC -- never `await` it. Use `await stream.close()`.
  - alpaca-py enums: there is NO `CryptoExchange`. Treat `trade.exchange`,
    `quote.bid_exchange`, `quote.ask_exchange` as plain strings (no `.value`).
    `trade.conditions` may be None OR a list -- guard with `trade.conditions or []`.
    The `Bar` model lacks an `exchange` attribute -- but we don't handle bars.
  - We are NOT handling bars. No `subscribe_bars`, no `bar_handler`. Do not
    add bars even if the brief mentions them.
  - L2 orderbooks: alpaca-py's `CryptoDataStream.subscribe_orderbooks(handler,
    *symbols)` delivers an `Orderbook` model with attrs `symbol`, `timestamp`,
    `bids`, `asks`. Each side is a LIST of items with `.price` and `.size`
    (NOT dicts and NOT keyed by exchange). Despite Alpaca's docs claiming
    snapshots, the websocket actually delivers DELTAS -- each message contains
    only the levels that changed. The subscriber MUST merge these into
    accumulated state, NEVER overwrite the whole book. Use per-side Redis
    HASHes keyed by price for an idempotent merge:
        for level in book.bids:                       # repeat for asks
            field = format_price(level.price)         # canonical str, e.g. f"{p:.10g}"
            if level.size > 0:
                r.hset(f"alpaca:ob:bids:{symbol}", field, level.size)
            else:
                r.hdel(f"alpaca:ob:bids:{symbol}", field)
    A level with `size == 0` (or a level NOT present in a follow-up message
    while previously present) means "remove this price level". Format the
    price field consistently across writes so updates land on the same hash
    field instead of accumulating duplicates. Also write a small metadata
    HASH `alpaca:orderbook:<SYM>` with `symbol` + `timestamp` so consumers
    can show last-update info.
  - Redis HSET cannot accept `None` values. Crypto quotes routinely have
    `bid_exchange`/`ask_exchange` = None. Before every `r.hset(..., mapping=d)`
    call, filter Nones with a `_clean(d)` helper:
        def _clean(d): return {k: v for k, v in d.items() if v is not None}
  - Construct CryptoDataStream WITH credentials, never `CryptoDataStream()`
    with no args."""

SUBSCRIBER_SYS = CODER_PREAMBLE + """

You are generating: subscriber.py

REQUIREMENTS:
  - Use the `alpaca-py` SDK (`from alpaca.data.live.crypto import CryptoDataStream`).
  - Read `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` from env; exit with
    a helpful message if missing.
  - Connect to Redis via `redis.Redis.from_url(os.environ.get("REDIS_URL",
    "redis://localhost:6379/0"), decode_responses=True)`.
  - Subscribe to **trades, quotes, AND orderbooks** (NOT bars) for the
    symbols passed on the command line (default to the SYMBOLS constant if
    none given).
  - For each incoming TRADE or QUOTE message, write to Redis (apply
    `_clean(d)` first):
      * `HSET alpaca:latest:<SYMBOL> <field> <value>` -- merge latest fields
        across trades/quotes (price, bid_price, bid_size, ask_price, ask_size,
        timestamp, ...). Field names: `price` (last trade), `bid_price`,
        `bid_size`, `ask_price`, `ask_size`, `timestamp`.
      * `LPUSH alpaca:recent:<SYMBOL>:<KIND> <json>` then `LTRIM ... 0 99`
        where KIND is `trade` or `quote`.
      * `SADD alpaca:symbols <SYMBOL>` so the CLI / GUI can list them.
  - For each incoming ORDERBOOK message, MERGE the levels into per-side
    Redis HASH keys (one HASH per side per symbol). Alpaca's crypto
    orderbook websocket delivers DELTAS -- each message has only the
    changed levels, NOT a full snapshot. Overwriting the whole book on
    each tick (the obvious wrong implementation) would leave you with
    just the last delta in Redis. Apply per-level updates instead:
        BIDS_KEY = f"alpaca:ob:bids:{symbol}"
        ASKS_KEY = f"alpaca:ob:asks:{symbol}"
        def _pf(p): return f"{float(p):.10g}"  # canonical price field name

        # for each level in the incoming book.bids:
        for lv in book.bids:
            field = _pf(lv.price)
            size  = float(lv.size)
            if size > 0:
                r.hset(BIDS_KEY, field, size)
            else:
                r.hdel(BIDS_KEY, field)
        # ... and the same for book.asks against ASKS_KEY.

    After applying the deltas, write a small metadata HASH for the GUI:
        r.hset(f"alpaca:orderbook:{symbol}", mapping={
            "symbol":    symbol,
            "timestamp": iso_ts_string,
        })
        r.sadd("alpaca:symbols", symbol)
    DO NOT json-encode bids/asks into the metadata hash; the actual book
    levels live ONLY in the per-side HASHes. DO NOT clear the per-side
    HASHes before writing -- that defeats the merge.

    Note: alpaca-py's Orderbook model carries `.bids` and `.asks` as
    LISTS of items with `.price` and `.size` ATTRIBUTES (not dicts).
    Read them via `lv.price`, `lv.size`.
  - Convert datetimes to ISO strings before storing. Use `default=str` in
    json.dumps for safety.
  - Async handlers (one each for trade / quote / orderbook); follow the exact
    main() pattern in the durable rules (subscribe_trades + subscribe_quotes
    + subscribe_orderbooks, then `await stream._run_forever()`).
  - Handle KeyboardInterrupt cleanly."""

TRADER_SYS = CODER_PREAMBLE + """

You are generating: trader.py

REQUIREMENTS:
  - Thin, importable module wrapping `alpaca.trading.client.TradingClient`.
  - PAPER TRADING IS THE DEFAULT. `paper=True` unless `live=True` is passed.
  - Imports:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
  - Factory:
        def get_client(live: bool = False) -> TradingClient:
            key = os.environ["ALPACA_API_KEY_ID"]
            secret = os.environ["ALPACA_API_SECRET_KEY"]
            return TradingClient(key, secret, paper=not live)
  - Functions (each accepts `live: bool = False`):
        account(live=False)                              -> dict
        positions(live=False)                            -> list[dict]
        orders(live=False, status='open', limit=50)      -> list[dict]
        buy_market(symbol, qty, live=False)              -> dict
        sell_market(symbol, qty, live=False)             -> dict
        buy_limit(symbol, qty, limit_price, live=False)  -> dict
        sell_limit(symbol, qty, limit_price, live=False) -> dict
        cancel(order_id, live=False)                     -> None
        cancel_all(live=False)                           -> list
  - For crypto (e.g. "BTC/USD"), use `time_in_force=TimeInForce.GTC` --
    `DAY` is rejected by crypto. Use `qty=` (fractional supported), not notional.
  - Serialize SDK responses with a `_to_dict(obj)` helper that prefers
    `model_dump(mode='json')`, falls back to `dict()` then `__dict__`, so the
    CLI can json.dumps() without hitting UUIDs/datetimes.
  - The module must import cleanly (no top-level network calls). Reading env
    vars only happens inside `get_client()` -- not at import time."""

CLI_SYS = CODER_PREAMBLE + """

You are generating: cli.py

You will receive the contents of subscriber.py and trader.py as context.
DO NOT redefine or modify them; the CLI imports their public surface.

REQUIREMENTS:
  - argparse CLI with FLAT subcommands. ONE level of subparsers only -- do
    NOT create grouping subcommands like `data` or `trade`. Invocations look
    like `cli.py buy BTC/USD 0.001`, NOT `cli.py trade buy BTC/USD 0.001`.
    All subcommand names below sit at the top level next to each other:
      Read-only Redis queries (data written by subscriber.py):
        * `symbols`            -> SMEMBERS alpaca:symbols
        * `latest <SYMBOL>`    -> HGETALL alpaca:latest:<SYMBOL>
        * `recent <SYMBOL> [--kind trade|quote] [-n N]`
              -> LRANGE alpaca:recent:<SYMBOL>:<kind> 0 N-1, pretty-print json
      Trading subcommands (proxy to trader.py). Each accepts a `--live` flag
      (default: paper):
        * `account`                          -> trader.account(live)
        * `positions`                        -> trader.positions(live)
        * `orders [--status STATUS] [--limit N]` -> trader.orders(...)
        * `buy <SYMBOL> <QTY> [--limit PRICE]`
              -> trader.buy_market(...) or trader.buy_limit(...) if --limit given
        * `sell <SYMBOL> <QTY> [--limit PRICE]`
              -> trader.sell_market(...) or trader.sell_limit(...)
        * `cancel <ORDER_ID>`                -> trader.cancel(order_id, live)
        * `cancel-all`                       -> trader.cancel_all(live)
    Implementation: a single `subparsers = parser.add_subparsers(...)`, then
    one `subparsers.add_parser(<name>)` per command above -- NOT nested.
  - Same Redis URL convention as subscriber.py for the read-only commands.
  - Pretty-print trading results as `json.dumps(result, indent=2, default=str)`.
  - Exit non-zero with a clear message when a Redis query has no data.
  - When `--live` is passed, print to stderr:
        "WARNING: --live mode -- this places real orders"
    before executing.
  - Import trader as `import trader` (same directory)."""

TRADE_STREAM_SYS = CODER_PREAMBLE + """

You are generating: trade_stream.py

Long-running worker that subscribes to Alpaca's TRADING websocket and
publishes every TradeUpdate event into a Redis Stream. Persistence to
SQLite is the responsibility of a SEPARATE worker (persister.py) that
consumes from this Redis Stream. This split decouples ingest from
durability: persister.py can be restarted without losing events because
the Redis Stream buffers them.

REQUIREMENTS:
  - Use `from alpaca.trading.stream import TradingStream` and
    `from alpaca.trading.models import TradeUpdate`.
  - PAPER TRADING IS THE DEFAULT. `TradingStream(key, secret, paper=True)`.
    Allow opt-in to live with `--live` (or env `ALPACA_LIVE=1`).
  - Read `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` from env at startup;
    exit with a helpful message if missing.
  - Connect to Redis:
        r = redis.Redis.from_url(os.environ.get("REDIS_URL",
            "redis://localhost:6379/0"), decode_responses=True)
  - Subscribe to trade updates:
        stream.subscribe_trade_updates(on_trade_update)
    where `on_trade_update(update: TradeUpdate)` is `async def`.
  - For each TradeUpdate:
      1. Serialize via the same `_to_dict(obj)` helper used in trader.py
         (prefer `model_dump(mode='json')`, fallback to `dict()` then
         `__dict__`). Cast UUIDs/datetimes to str. Recurse into the
         nested `order` field so the serialized payload is a pure dict.
      2. Build a payload dict:
           {"event": ..., "execution_id": ..., "order_id": ...,
            "symbol": ..., "side": ..., "price": ..., "qty": ...,
            "ts": ..., "order": <serialized order dict>}
         Missing fields default to None / "".
      3. XADD to the Redis Stream `alpaca:trade_updates` with one field:
           r.xadd("alpaca:trade_updates",
                  {"json": json.dumps(payload, default=str)},
                  maxlen=100000, approximate=True)
      4. Also `LPUSH alpaca:recent:trade_updates <json>` then
         `LTRIM alpaca:recent:trade_updates 0 99` for at-a-glance debug
         (mirror what subscriber.py does for market data).
      5. Print a one-line status to stdout:
           f"[{ts}] {event} order={order_id[:8]} {symbol} {side} {qty}"
  - Async runtime pattern (same as subscriber.py -- use the durable rules):
        async def main():
            stream.subscribe_trade_updates(on_trade_update)
            try:
                await stream._run_forever()
            finally:
                await stream.close()
        asyncio.run(main())
    `stream.run()` is SYNC -- never await it. `stream.stop()` is SYNC --
    use `await stream.close()`.
  - Handle KeyboardInterrupt cleanly. Module imports must NOT touch the
    network -- only `main()` does."""

PERSISTER_SYS = CODER_PREAMBLE + """

You are generating: persister.py

Long-running worker that READS trade updates from Redis and persists every
order + fill event into a local SQLite database. It does NOT open the
Alpaca websocket -- a separate worker (trade_stream.py) handles that and
publishes events to the Redis Stream `alpaca:trade_updates`. This split
keeps ingest decoupled from persistence: persister.py can be restarted
without losing events (the stream buffers them; the consumer group resumes
from the last acknowledged id).

REQUIREMENTS:
  - DO NOT import or use `TradingStream`. No websocket, no alpaca-py
    network calls in this file. The data source is Redis.
  - Connect to Redis:
        r = redis.Redis.from_url(os.environ.get("REDIS_URL",
            "redis://localhost:6379/0"), decode_responses=True)
  - Source stream: `alpaca:trade_updates` (Redis Stream).
    Consumer group: env $PERSISTER_GROUP, default `persister`.
    Consumer name:  env $PERSISTER_CONSUMER, default `persister-1`.
    Idempotently create the group on startup:
        try:
            r.xgroup_create("alpaca:trade_updates", group,
                            id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
    Use `id="0"` so a fresh group replays from the start of the stream;
    this is safe because we use INSERT OR IGNORE on `execution_id`.
  - SQLite path comes from `os.environ.get("ALPACA_DB_PATH", "./alpaca.db")`.
    Use stdlib `sqlite3` only. Connect once at startup; pass the connection
    to the handler via a closure or module global.
  - Schema (CREATE TABLE IF NOT EXISTS at startup):
        orders(
          order_id        TEXT PRIMARY KEY,    -- the alpaca order UUID as text
          client_order_id TEXT,
          symbol          TEXT,
          side            TEXT,
          type            TEXT,
          qty             REAL,
          filled_qty      REAL,
          filled_avg_price REAL,
          status          TEXT,
          submitted_at    TEXT,
          updated_at      TEXT,
          created_at      TEXT,
          raw_json        TEXT
        )
        fills(
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          execution_id    TEXT UNIQUE,         -- payload['execution_id']; UNIQUE for idempotence
          order_id        TEXT,
          symbol          TEXT,
          side            TEXT,
          price           REAL,
          qty             REAL,
          event           TEXT,                -- 'fill' or 'partial_fill'
          ts              TEXT,                -- payload['ts']
          raw_json        TEXT,
          FOREIGN KEY(order_id) REFERENCES orders(order_id)
        )
        trade_events(
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          event           TEXT,                -- payload['event']
          order_id        TEXT,
          ts              TEXT,
          raw_json        TEXT
        )
    Add indexes: `CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)`,
    `idx_fills_order ON fills(order_id)`, `idx_events_ts ON trade_events(ts)`.
  - Main loop is plain synchronous Python -- no asyncio:
        while True:
            resp = r.xreadgroup(group, consumer,
                                {"alpaca:trade_updates": ">"},
                                count=64, block=5000)
            for _stream, entries in resp or []:
                for entry_id, fields in entries:
                    try:
                        payload = json.loads(fields["json"])
                        handle(payload)         # writes to sqlite
                        conn.commit()
                    except Exception as e:
                        # Don't ack on failure -- let the entry stay
                        # pending so it can be re-processed. Log + continue.
                        print(f"  ! error on {entry_id}: {e}", file=sys.stderr)
                        continue
                    r.xack("alpaca:trade_updates", group, entry_id)
  - Handle one payload (a dict with keys produced by trade_stream.py):
      event        = payload.get("event")
      execution_id = payload.get("execution_id")
      order_id     = payload.get("order_id")
      symbol       = payload.get("symbol")
      side         = payload.get("side")
      price        = payload.get("price")
      qty          = payload.get("qty")
      ts           = payload.get("ts")
      order        = payload.get("order") or {}   # nested order dict
    Then:
      1. Always INSERT into `trade_events(event, order_id, ts, raw_json)`
         with raw_json = json.dumps(payload, default=str).
      2. Upsert the order (use values from the nested `order` dict, falling
         back to top-level fields where order is empty):
           INSERT INTO orders(order_id, client_order_id, symbol, side, type,
             qty, filled_qty, filled_avg_price, status,
             submitted_at, updated_at, created_at, raw_json)
           VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(order_id) DO UPDATE SET
             filled_qty=excluded.filled_qty,
             filled_avg_price=excluded.filled_avg_price,
             status=excluded.status,
             updated_at=excluded.updated_at,
             raw_json=excluded.raw_json;
         Skip the upsert if there is no order_id.
      3. If event in ('fill','partial_fill'), INSERT OR IGNORE into
         `fills(execution_id, order_id, symbol, side, price, qty, event,
                ts, raw_json)`. Idempotent on `execution_id`.
      4. `conn.commit()` after each payload (the loop already does this
         once per entry; just make sure the handler doesn't commit early).
  - Use `.get()` defensively everywhere -- some payload fields may be
    missing or None depending on the event type. Cast numeric fields with
    `float(...)` only when not None.
  - Print a one-line status to stdout for each handled payload:
        f"[{ts}] {event} order={(order_id or '')[:8]} {symbol} {side} {qty}"
  - On startup print: the consumer group, consumer name, db path, and
    redis url so the operator can see the wiring.
  - Handle KeyboardInterrupt cleanly: close db + redis, then exit. Module
    imports must NOT touch the network -- only `main()` does."""

WEBGUI_SYS = CODER_PREAMBLE + """

You are generating: webgui.py

Thin Flask BACKEND for the trading UI. ALL HTML/CSS/JS lives in a separate
file (`generated/web/index.html`) generated by a different stage; this
file MUST NOT contain any inline HTML template -- no `render_template_string`,
no embedded `<html>` strings. Serve `web/index.html` as a static file.
The backend's job: serve that static page + expose JSON endpoints over
Redis / SQLite / trader.

You will receive subscriber.py, trader.py, persister.py, and
web/index.html as context -- do NOT redefine them; just use their public
surface, the Redis key conventions, the SQLite schema, and the endpoint
shapes the HTML's JS calls.

REQUIREMENTS:
  - Use Flask. Module-level work allowed (`app = Flask(__name__,
    static_folder=str(WEB_DIR), static_url_path='/static')`, route
    decorators) but do NOT call `app.run()` at import time -- only inside
    `if __name__ == "__main__":` so the verifier (which imports the
    module) doesn't bind a port.
  - Locate the web dir relative to THIS file:
        from pathlib import Path
        WEB_DIR = Path(__file__).parent / "web"
  - `GET /` returns `send_from_directory(str(WEB_DIR), "index.html")`.
    If `web/index.html` is missing, return a 200 with a small inline
    string saying "web/index.html not generated yet -- run `python
    alpaca_codegen.py --stage webgui_html`" so the operator gets a clear
    message instead of a 404.
  - Read REDIS_URL the same way subscriber.py does
    (`os.environ.get("REDIS_URL", "redis://localhost:6379/0")`,
    `decode_responses=True`).
  - Read ALPACA_DB_PATH the same way persister.py does
    (`os.environ.get("ALPACA_DB_PATH", "./alpaca.db")`).
  - Import trader as `import trader` (same directory) for order placement.
  - Default bind: host=os.environ.get("WEBGUI_HOST", "127.0.0.1"),
    port=int(os.environ.get("WEBGUI_PORT", "5000")). Print the URL on
    startup. `debug=False`, `threaded=True`.
  - Silence Werkzeug's per-request access log (the Escalator polls
    /api/orderbook every 1s -- that's 60 lines/min of noise per browser
    tab):
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
    Replace it with a throttled summary: an `@app.after_request` hook
    that counts requests by path under a `threading.Lock`, and once
    every 60 seconds prints ONE line:
        f"[webgui] {total} requests in last 60s ({top_paths_with_counts})"
    Then resets the counter. Use `time.monotonic()` for the interval
    timer. The summary print should `flush=True` so it appears promptly.

  JSON API endpoints (each returns JSON via flask.jsonify). ALL DATA FOR
  THE LADDER COMES FROM REDIS -- never call Alpaca for market data here.

    GET  /api/symbols
        -> sorted list from SMEMBERS alpaca:symbols. Empty list (not 500)
           if the set is missing.

    GET  /api/orderbook/<path:symbol>
        -> Build a top-N-levels snapshot for ONE symbol strictly from
           Redis. The book is stored as TWO per-side HASHes maintained by
           subscriber.py via merge-of-deltas:
               alpaca:ob:bids:<SYM>   field = price (str)  value = size (str)
               alpaca:ob:asks:<SYM>   field = price (str)  value = size (str)
           Resolution order:
             1. HGETALL alpaca:ob:bids:<SYM> + HGETALL alpaca:ob:asks:<SYM>.
                If EITHER side has at least one entry, build:
                  bids = [{"price": float(p), "size": float(s)}
                          for p, s in bid_hash.items() if float(s) > 0]
                  bids.sort(key=lambda x: -x["price"])     # best (highest) first
                  asks = [...]
                  asks.sort(key=lambda x:  x["price"])     # best (lowest)  first
                Return {"symbol": <SYM>,
                        "timestamp": HGET alpaca:orderbook:<SYM> timestamp,
                        "bids": bids[:10], "asks": asks[:10]}.
             2. Else fallback to HGETALL alpaca:latest:<SYM> and synthesize
                a 1-level book from `bid_price`/`bid_size`/`ask_price`/
                `ask_size`. Include `"fallback": "quote_only"`.
             3. Else return {"symbol": <SYM>, "bids": [], "asks": [],
                "fallback": "no_data"} -- NEVER 500.
           Coerce numeric fields to floats. Skip any field whose value
           parses to 0 or non-numeric. Do NOT read the metadata HASH's
           `bids`/`asks` keys -- they aren't there in the new schema.

    GET  /api/latest/<path:symbol>
        -> HGETALL alpaca:latest:<SYM>. {} if absent.

    GET  /api/orders?limit=50&status=...
        -> rows from SQLite orders ORDER BY submitted_at DESC LIMIT ?.
           [] if the db file or table is missing.

    GET  /api/fills?limit=50
        -> rows from SQLite fills ORDER BY ts DESC LIMIT ?. [] if missing.

    GET  /api/summary?live=0
        -> {"account": {...}, "positions": [...],
            "realized_pnl_by_symbol": {SYM: float}}.
           account + positions come from trader.account(live) /
           trader.positions(live). realized_pnl is aggregated from the
           SQLite fills table: per-symbol sum of `(price*qty) * (+1 if
           side=='sell' else -1)`. `live` query arg in {"1","true","yes"}
           enables live; anything else is paper. If creds / DB missing,
           return whatever blocks succeeded and `null` for the rest --
           never 500.

    POST /api/order
        -> JSON body:
            {"symbol": str, "qty": float, "side": "buy"|"sell",
             "type": "market"|"limit",
             "limit_price": float (required if type=="limit"),
             "live": bool (default false)}
           Validate side, type, required fields. On bad input return
           ({"error": "..."}, 400). Otherwise dispatch to
           trader.buy_market / trader.sell_market / trader.buy_limit /
           trader.sell_limit and return its result dict. On trader
           exception, return ({"error": str(e), "type":
           e.__class__.__name__}, 400).

    DELETE /api/order/<order_id>
        -> trader.cancel(order_id, live=...). `live` from `?live=1` query
           arg. Return {"cancelled": <id>} on success, ({"error":...},
           400) on failure.

  - SQLite reads: open a fresh connection per request
    (`sqlite3.connect(db_path, check_same_thread=False)`). Wrap each
    query in `try/except sqlite3.OperationalError` (tables may not exist
    before persister.py first runs) and return [] on that error. Use
    `conn.row_factory = sqlite3.Row` and convert rows to dicts so the
    JSON response has named fields.
  - Defensive rendering: NO endpoint may 500 on missing data -- empty
    Redis returns []/{}; missing alpaca.db returns []; missing creds in
    /api/summary returns null/empty blocks.
  - The trader module reads ALPACA_API_KEY_ID/SECRET only inside
    get_client(); routes that DON'T need trading (everything except
    /api/order, /api/order/<id> DELETE, and /api/summary) MUST NOT call
    any trader.* function, so the page renders even without Alpaca creds.
  - Handle KeyboardInterrupt cleanly. Module imports must NOT touch the
    network. Keep the file SHORT -- no inline templates, no fancy
    helpers. Aim well under 300 lines."""

INDEX_HTML_SYS = """You are a senior frontend engineer. Generate a SINGLE
self-contained HTML page (no build step, no external JS/CSS dependencies)
that drives the Alpaca crypto trading UI by calling the JSON endpoints
exposed by `webgui.py`.

Output format: emit EXACTLY ONE fenced ```html``` block containing the
complete HTML document. No prose, no FILE markers, no extra fenced blocks.
The first line of the block must be `<!DOCTYPE html>`.

Constraints:
  - Vanilla JS only -- no React/Vue/jQuery/imports/CDN <script src=...>.
  - Inline `<style>` and inline `<script>`. One file, no externals.
  - Use `fetch()` for REST and `setInterval()` for polling. (No SSE -- the
    backend is REST-only.)
  - Pages MUST work even if Redis is empty / SQLite is missing. Show
    placeholders ("(no data)" / "(no symbols streaming)") rather than
    blowing up on null/empty responses.

LAYOUT:
  - A small header strip with the page title and a Refresh button.
    The Symbol selector is NOT global -- it lives inside the Escalator
    tab (it is meaningful only there).
  - Three TABS, switched client-side via plain JS:

  Tab 1: "Escalator" -- per-symbol depth ladder.
    THE ESCALATOR'S DATA SOURCE IS REDIS (read by webgui.py, served via
    /api/orderbook/<symbol>). The page itself never talks to Alpaca.
    - The TAB ITSELF owns a Symbol <select> at its top (populated from
      /api/symbols). The selector must not appear in the page header --
      only the Escalator tab uses it.
    - One ladder for the symbol CURRENTLY SELECTED in this tab's
      dropdown. Switching the dropdown re-renders the ladder.
    - The ladder is a single table with EXACTLY 21 rows:
        * 10 ASK rows on top, ordered HIGHEST price at the very top down
          to BEST (lowest) ask just above the spread divider.
        * 1 spread divider row in the middle, showing
          "spread <px>  (<bps> bps)" computed from the best bid/ask.
        * 10 BID rows below, ordered BEST (highest) bid just under the
          divider down to lowest at the bottom.
      ONE PRICE PER ROW. Columns per row: side label, price, size.
    - If fewer than 10 distinct levels exist on a side, PAD the missing
      slots with EMPTY rows so the ladder is always 21 rows tall. Do
      NOT render literal "undefined" or "null" in any cell -- render an
      empty string instead.
    - Visually distinguish ask rows (red-tinted text + hover) from bid
      rows (green-tinted text + hover).
    - Click handlers:
        * Clicking an ASK row opens the order modal as a BUY at that
          row's price (the user is lifting the offer).
        * Clicking a BID row opens the order modal as a SELL at that
          row's price (the user is hitting the bid).
        * Empty rows are not clickable.
    - Polling: GET /api/orderbook/<currentSymbol> every 1000ms; replace
      the table body in place.
    - Below the ladder, show a small meta line: symbol, timestamp, and
      any `fallback` field present in the response (e.g. "[quote_only]"
      or "[no_data]") so the operator can see the data source.

  Tab 2: "Activity" -- recent orders + fills from SQLite.
    - Manual Refresh button + a Status filter <select> (all / open /
      filled / canceled). No auto-poll (this changes slowly).
    - Orders table: GET /api/orders?limit=50[&status=...]. Columns:
      submitted_at, symbol, side, type, qty, filled_qty, status,
      order_id (truncate to 8 chars; full id in `title` attribute).
      For active statuses (new/accepted/pending_new/partially_filled),
      include a "cancel" button that DELETEs /api/order/<id>.
    - Fills table below it: GET /api/fills?limit=50. Columns: ts,
      order_id (truncated), symbol, side, qty, price, event.

  Tab 3: "Summary" -- account + positions + realized PnL.
    - Live toggle (checkbox, default OFF=paper). Manual Refresh button.
    - One GET /api/summary?live=0|1 returns {account, positions,
      realized_pnl_by_symbol}. Render:
        * Account block: equity, cash, buying_power, portfolio_value
          (key/value grid; coerce numeric strings to floats for display
          when possible).
        * Positions table: symbol, qty, avg_entry_price, current_price,
          market_value, unrealized_pl, unrealized_plpc.
        * Realized PnL table: per-symbol realized $; bottom row total.

ORDER MODAL (used by Tab 1 row clicks):
  - Implemented as an absolutely-positioned `<div>` toggled via JS --
    NOT the native `<dialog>` element (browser compat).
  - Inputs:
      * Symbol (readonly, from the row's symbol).
      * Side  (readonly, "buy" for ask-row click / "sell" for bid-row).
      * Type  (radio: market / limit; default "limit"). When "limit",
        a Limit price input is shown, pre-filled with the clicked row's
        price.
      * Qty   (number, step="any", required, > 0).
      * Live  (checkbox, default OFF). When checked, before submit show
        a JS confirm() "This will place a REAL order. Continue?".
      * Submit + Close buttons. Esc closes the modal. Enter submits.
  - On submit: POST JSON to /api/order with {symbol, qty, side, type,
    limit_price (only if type==limit), live}. Show the response
    (success or error) as a small toast in the corner; close the modal
    on success.

API CONTRACT (match the keys webgui.py emits):
  - GET  /api/symbols                -> string[] (sorted symbols)
  - GET  /api/orderbook/<symbol>     -> {symbol, timestamp?, bids:[{price,size}],
                                          asks:[{price,size}], fallback?}
  - GET  /api/orders?limit=N[&status=S]
                                     -> object[] (rows from SQLite orders)
  - GET  /api/fills?limit=N          -> object[] (rows from SQLite fills)
  - GET  /api/summary?live=0|1       -> {account, positions, realized_pnl_by_symbol}
  - POST /api/order  body={symbol,qty,side,type,limit_price?,live}
                                     -> order dict on 200, {error,...} on 400
  - DELETE /api/order/<order_id>?live=0|1
                                     -> {cancelled} on 200, {error} on 400

  Defensive behavior: `/api/orderbook/<sym>` may return {bids:[],asks:[],
  fallback:"no_data"} -- the ladder must still render 21 padded rows.
  `/api/orders` and `/api/fills` may return [] -- show "(no rows)".

Keep the file under ~550 lines total. No emojis."""


# Stage registry: ordered tuples of (stage_name, target_path, lang,
# system_prompt, prior_files_to_pass).
#   stage_name: short id used by --stage and STAGE_BY_NAME (e.g. "webgui_html").
#   target_path: path RELATIVE to OUT_DIR (e.g. "web/index.html").
#   lang:        fenced-block language tag the LLM must emit ("python"|"html").
# Prior files are read from OUT_DIR and given to the Coder as `### EXISTING:`
# context. Their language is inferred from extension (.py -> python, .html ->
# html) so the Coder sees the right code-fence on each.
STAGES = [
    ("subscriber",   "subscriber.py",     "python", SUBSCRIBER_SYS,   []),
    ("trader",       "trader.py",         "python", TRADER_SYS,       []),
    ("cli",          "cli.py",            "python", CLI_SYS,
        ["subscriber.py", "trader.py"]),
    # trade_stream: subscribes to Alpaca TradingStream and publishes events
    # into a Redis Stream `alpaca:trade_updates`. trader.py is provided so
    # the LLM can crib the _to_dict serialization helper.
    ("trade_stream", "trade_stream.py",   "python", TRADE_STREAM_SYS, ["trader.py"]),
    # persister: reads from the Redis Stream and writes to SQLite. No
    # websocket. trade_stream.py is provided so the LLM knows the exact
    # shape of the JSON payloads it will be parsing.
    ("persister",    "persister.py",      "python", PERSISTER_SYS,    ["trade_stream.py"]),
    # webgui_html: standalone HTML/CSS/JS frontend. Self-contained -- it
    # talks to webgui.py over JSON only, so it doesn't need any Python prior.
    # Generated BEFORE webgui.py so the backend can be told exactly which
    # endpoints the frontend calls.
    ("webgui_html",  "web/index.html",    "html",   INDEX_HTML_SYS,   []),
    # webgui: thin Flask backend. Serves web/index.html as a static file +
    # exposes JSON endpoints. Reads Redis (subscriber's keys), SQLite
    # (persister's schema), places orders via trader. All four priors are
    # passed so the LLM matches field names / table columns / function
    # signatures / API contract exactly.
    ("webgui",       "webgui.py",         "python", WEBGUI_SYS,
        ["subscriber.py", "trader.py", "persister.py", "web/index.html"]),
]
STAGE_BY_NAME = {name: i for i, (name, _, _, _, _) in enumerate(STAGES)}


# ---------- Helpers ----------

def extract_block(text: str, lang: str) -> str | None:
    """Pull a fenced code block out of the LLM's response.

    Tried in order:
      1. ```<lang>\\n...\\n```            (the canonical case; case-insensitive
                                            on the lang tag because Gemini
                                            sometimes emits ```HTML).
      2. ```\\n...\\n```                  (no language tag at all).
      3. raw <!doctype...</html>         (HTML emitted with no fence -- the
                                            common Gemini failure mode for
                                            non-Python artifacts).
      4. ```<lang>\\n...                 (truncated -- hit max_tokens before
                                            the closing fence; let the
                                            verifier surface the real error).
      5. ```\\n...                       (truncated, no language tag).

    Greedy on the closing fence: the brief contains nested code-fence examples
    and a non-greedy match would stop at the first inner fence.
    """
    lang_re = re.escape(lang)
    # 1. Canonical fenced block with the right language tag.
    m = re.search(rf"```{lang_re}\b\s*\n(.*)\n```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2. Fenced block with no language tag (Gemini drops it sometimes).
    m = re.search(r"```\s*\n(.*)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 3. HTML-specific: bare <!DOCTYPE ... </html>, no fence at all.
    if lang == "html":
        m = re.search(r"(<!doctype\s+html[\s\S]*?</html\s*>)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # 4. Truncated: opening fence with the right lang, no closing fence.
    m = re.search(rf"```{lang_re}\b\s*\n(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 5. Truncated: opening fence, no language, no closing fence.
    m = re.search(r"```\s*\n(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else None


def chat_once(user: UserProxyAgent, agent: AssistantAgent, message: str) -> str:
    user.initiate_chat(agent, message=message, max_turns=1,
                       clear_history=True, silent=True)
    msg = agent.last_message()
    return msg["content"] if isinstance(msg, dict) else str(msg)


def chat_with_retry(user, agent, message, retries=3) -> str:
    last = None
    for i in range(retries):
        try:
            return chat_once(user, agent, message)
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(2.0 * (i + 1))
    raise last


_LANG_BY_EXT = {".py": "python", ".html": "html", ".md": "markdown"}


def _lang_for_path(path: str) -> str:
    """Infer the fenced-block language for a file, defaulting to its bare ext."""
    ext = Path(path).suffix.lower()
    return _LANG_BY_EXT.get(ext, ext.lstrip(".") or "text")


def verify_python_file(work_dir: str, target: str, target_code: str,
                       priors: dict[str, str]) -> tuple[bool, str]:
    """Write `target` (plus any `priors` so cross-file imports resolve) into
    work_dir, `python -m py_compile` the target, then `import` the target.

    Importing executes the module body. subscriber.py / cli.py / trader.py
    only do safe top-level work (lazy redis client, env-var reads) -- no
    network calls, so this is safe.

    Priors with non-.py extensions (e.g. web/index.html) are still written
    so they're discoverable from the module under test, but they're not
    imported.
    """
    for name, code in priors.items():
        p = Path(work_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
    target_p = Path(work_dir) / target
    target_p.parent.mkdir(parents=True, exist_ok=True)
    target_p.write_text(target_code)

    target_mod = target[:-3] if target.endswith(".py") else target
    code = (
        "import subprocess, sys, importlib\n"
        f"r = subprocess.run([sys.executable,'-m','py_compile',{target!r}], "
        "capture_output=True, text=True)\n"
        "print('COMPILE_STDOUT:', r.stdout)\n"
        "print('COMPILE_STDERR:', r.stderr)\n"
        "if r.returncode != 0: print('EXIT:', r.returncode); sys.exit(0)\n"
        "try:\n"
        f"    importlib.import_module({target_mod!r})\n"
        f"    print('IMPORT_OK: {target_mod}')\n"
        "    print('EXIT: 0')\n"
        "except Exception as e:\n"
        f"    print(f'IMPORT_FAIL: {target_mod}: ' + type(e).__name__ + ': ' + str(e))\n"
        "    print('EXIT: 1')\n"
    )
    executor = LocalCommandLineCodeExecutor(work_dir=work_dir,
                                            timeout=SUBPROC_TIMEOUT)
    from autogen.coding.base import CodeBlock
    res = executor.execute_code_blocks([CodeBlock(language="python", code=code)])
    out = res.output
    ok = "EXIT: 0" in out
    return ok, out.strip()[-3000:]


def verify_html_file(work_dir: str, target: str, target_code: str,
                     priors: dict[str, str]) -> tuple[bool, str]:
    """Sanity-check an HTML artifact: non-trivial size, declares a doctype,
    has a closing </html>. We can't run the page server-side; the goal is
    just to catch obvious truncation / wrong-language emission.
    """
    head = target_code.lstrip()[:200].lower()
    problems = []
    if len(target_code.strip()) < 200:
        problems.append(f"file is suspiciously short ({len(target_code)} chars)")
    if not head.startswith("<!doctype"):
        problems.append("missing leading <!DOCTYPE html> declaration")
    if "</html>" not in target_code.lower():
        problems.append("missing closing </html> tag (likely truncated output)")
    # Quick heuristic: the LLM sometimes emits a code-fence-of-fences. Bail.
    if target_code.lstrip().startswith("```"):
        problems.append("output still contains a leading fence; emit raw HTML")
    if problems:
        return False, "HTML verification failed:\n  - " + "\n  - ".join(problems)
    return True, f"HTML verify OK ({len(target_code)} chars)"


def verify(lang: str, work_dir: str, target: str, target_code: str,
           priors: dict[str, str]) -> tuple[bool, str]:
    """Dispatch to the per-language verifier."""
    if lang == "python":
        return verify_python_file(work_dir, target, target_code, priors)
    if lang == "html":
        return verify_html_file(work_dir, target, target_code, priors)
    # Unknown language: trust the LLM, just check non-empty.
    if not target_code.strip():
        return False, f"empty {lang} output"
    return True, f"no verifier for lang={lang!r}; accepted as-is"


def _read_priors(prior_names: list[str]) -> dict[str, str]:
    """Read prior-stage files from OUT_DIR (skipping any that don't exist).

    Keys are the relative paths exactly as listed in STAGES (e.g.
    `web/index.html`), so cross-file references in the rendered prompt
    line up with the on-disk layout the verifier reproduces.
    """
    out: dict[str, str] = {}
    for n in prior_names:
        p = OUT_DIR / n
        if p.exists():
            out[n] = p.read_text()
    return out


# ---------- Orchestration ----------

def ensure_research(force: bool = False) -> str:
    """Generate RESEARCH.md if missing (or `force=True`). Returns its content."""
    research_path = OUT_DIR / "RESEARCH.md"
    if research_path.exists() and not force:
        return research_path.read_text()

    cfg = gemini_llm_config(temperature=0.2)
    researcher = AssistantAgent("researcher", llm_config=cfg,
                                system_message=RESEARCHER_SYS,
                                human_input_mode="NEVER")
    user = UserProxyAgent("user", human_input_mode="NEVER",
                          code_execution_config=False,
                          max_consecutive_auto_reply=0,
                          default_auto_reply="")
    print(f"[research] drafting brief with {GEMINI_MODEL}...")
    raw = chat_with_retry(
        user, researcher,
        "Write the Alpaca brief as specified, covering both Part A "
        "(crypto websocket) and Part B (trading REST).",
    )
    brief_md = extract_block(raw, "markdown") or raw.strip()
    research_path.write_text(brief_md + "\n")
    print(f"          wrote {research_path} ({len(brief_md)} chars)")
    return brief_md


def run_stage(stage_idx: int, brief_md: str, symbols: list[str],
              extra_feedback: str = "") -> bool:
    """Generate one stage's file. Returns True on success.

    Looks up the stage from STAGES[stage_idx]. Reads any prior files from
    OUT_DIR (they're context, not regenerated). Loops up to MAX_RETRIES with
    compile/import (or html-sanity) errors fed back to the Coder.
    """
    stage_name, target_file, lang, system_prompt, prior_names = STAGES[stage_idx]
    target_path = OUT_DIR / target_file
    priors = _read_priors(prior_names)

    cfg = gemini_llm_config(temperature=0.2)
    coder = AssistantAgent("coder", llm_config=cfg,
                           system_message=system_prompt,
                           human_input_mode="NEVER")
    user = UserProxyAgent("user", human_input_mode="NEVER",
                          code_execution_config=False,
                          max_consecutive_auto_reply=0,
                          default_auto_reply="")

    priors_md = ""
    for name, code in priors.items():
        prior_lang = _lang_for_path(name)
        priors_md += (
            f"\n\n### EXISTING: {name} (already on disk; do NOT redefine, "
            f"use its public surface)\n```{prior_lang}\n{code}\n```"
        )

    # Sanitize the work-dir prefix: target_file may contain a "/" (web/index.html)
    safe_prefix = target_file.replace("/", "_").replace("\\", "_")
    work_dir = tempfile.mkdtemp(prefix=f"alpaca_stage_{safe_prefix}_")
    feedback = extra_feedback
    code = ""
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"  [stage:{target_file}] {GEMINI_MODEL} "
                  f"attempt {attempt}/{MAX_RETRIES}...")
            ask = (
                "Research brief:\n\n```markdown\n" + brief_md + "\n```"
                + priors_md
                + f"\n\nSYMBOLS (default for subscriber): {symbols}\n"
                  f"\nGenerate {target_file}. Output ONE fenced ```{lang}``` "
                  "block with the full file contents. No prose."
            )
            if feedback:
                ask += (
                    "\n\nPrevious attempt failed verification:\n"
                    f"```\n{feedback.strip()}\n```\n"
                    f"Fix the issue and re-emit {target_file} in full."
                )
            raw = chat_with_retry(user, coder, ask)
            code = extract_block(raw, lang) or ""
            if not code:
                # Show the head + tail of the raw response so the operator
                # can see what the model actually emitted (different fence,
                # leading prose, etc.) without dumping 30k tokens.
                head = raw[:400].replace("\n", "\\n")
                tail = raw[-200:].replace("\n", "\\n") if len(raw) > 600 else ""
                print(f"            ! no {lang} block extracted from "
                      f"{len(raw)}-char response; head={head!r}"
                      + (f" ... tail={tail!r}" if tail else ""))
                feedback = (
                    f"No ```{lang}``` fenced block found in your output. "
                    f"Output exactly one fenced ```{lang}``` block at the "
                    f"START of your message, no prose before or after. "
                    f"The opening fence MUST be three backticks immediately "
                    f"followed by the literal word `{lang}` then a newline."
                )
                continue
            ok, out = verify(lang, work_dir, target_file, code, priors)
            if ok:
                print("            verify OK")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(code + ("" if code.endswith("\n") else "\n"))
                print(f"            wrote {target_path} ({len(code)} chars)")
                return True
            print("            ! verify failed; feeding errors back")
            feedback = out
        # MAX_RETRIES exhausted -- write last attempt for inspection
        if code:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(code + ("" if code.endswith("\n") else "\n"))
            print(f"  [stage:{target_file}] FAILED after {MAX_RETRIES}; "
                  f"wrote last attempt to {target_path} for inspection")
        return False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_pipeline(symbols: list[str], stage_filter: str = "missing",
                 force_research: bool = False) -> None:
    """Top-level driver.

    stage_filter:
      - "missing" (default): run any stage whose output file is absent. If a
        file already exists, we skip it -- safe for preserving working code.
      - "all":               regenerate every stage, overwriting existing files.
      - "subscriber"/"trader"/"cli": run only that stage, regardless of state.
    """
    OUT_DIR.mkdir(exist_ok=True)

    print(f"=== alpaca_codegen | model={GEMINI_MODEL} | filter={stage_filter} ===")
    brief_md = ensure_research(force=force_research)

    def _target_of(idx: int) -> str:
        return STAGES[idx][1]

    if stage_filter == "missing":
        to_run = [
            i for i in range(len(STAGES))
            if not (OUT_DIR / _target_of(i)).exists()
        ]
        skipped = [
            _target_of(i) for i in range(len(STAGES))
            if (OUT_DIR / _target_of(i)).exists()
        ]
        if skipped:
            print(f"  skipping (already present): {skipped}")
    elif stage_filter == "all":
        to_run = list(range(len(STAGES)))
    elif stage_filter in STAGE_BY_NAME:
        to_run = [STAGE_BY_NAME[stage_filter]]
    else:
        raise SystemExit(
            f"Unknown stage {stage_filter!r}. "
            f"Use one of: missing, all, {', '.join(STAGE_BY_NAME)}"
        )

    if not to_run:
        print("  nothing to do (all stage files already present). "
              "Use --stage all to force regenerate.")
        return

    results: dict[str, bool] = {}
    for i in to_run:
        ok = run_stage(i, brief_md, symbols)
        results[_target_of(i)] = ok

    print("\n=== Summary ===")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED (last attempt written for inspection)"
        print(f"  {name}: {status}")

    print("\nNext steps (paper trading by default):")
    print("  pip install alpaca-py redis flask")
    print("  export ALPACA_API_KEY_ID=...  ALPACA_API_SECRET_KEY=...")
    print("  # start redis (e.g. docker run -p 6379:6379 redis)")
    print(f"  python {OUT_DIR/'subscriber.py'}            # market-data -> Redis")
    print(f"  python {OUT_DIR/'trade_stream.py'}          # trade updates -> Redis Stream")
    print(f"  python {OUT_DIR/'persister.py'}             # Redis Stream -> SQLite")
    print(f"  python {OUT_DIR/'webgui.py'}                # web UI on http://127.0.0.1:5000")
    print(f"  python {OUT_DIR/'cli.py'} symbols")
    print(f"  python {OUT_DIR/'cli.py'} latest {symbols[0]}")
    print(f"  python {OUT_DIR/'cli.py'} account")
    print(f"  python {OUT_DIR/'cli.py'} buy {symbols[0]} 0.001")
    print(f"  sqlite3 alpaca.db 'SELECT * FROM orders LIMIT 5'")


def run_fix(error_text: str, stage: str, symbols: list[str]) -> None:
    """Re-engage one stage's Coder with a runtime traceback as feedback.

    Reuses RESEARCH.md and any existing prior files in OUT_DIR.
    """
    research_path = OUT_DIR / "RESEARCH.md"
    if not research_path.exists():
        raise SystemExit(
            f"{research_path} not found. Run the full pipeline first."
        )
    if stage not in STAGE_BY_NAME:
        raise SystemExit(
            f"--fix requires a stage. Use one of: {', '.join(STAGE_BY_NAME)}"
        )
    brief_md = research_path.read_text()
    feedback = (
        "Previous attempt PASSES py_compile but CRASHES at run time with "
        "this traceback. Re-emit the file in full with the fix applied. "
        "Adhere to all durable rules in the system prompt.\n\n"
        f"Traceback:\n```\n{error_text.strip()}\n```"
    )
    ok = run_stage(STAGE_BY_NAME[stage], brief_md, symbols,
                   extra_feedback=feedback)
    print(f"\n[fix] {STAGES[STAGE_BY_NAME[stage]][1]}: "
          + ("OK" if ok else "FAILED"))


def _parse_args(argv: list[str]) -> dict:
    out = {"mode": "pipeline", "stage": "missing", "symbols": [],
           "error_file": None, "force_research": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--stage":
            i += 1
            out["stage"] = argv[i]
        elif a == "--fix":
            out["mode"] = "fix"
            i += 1
            out["error_file"] = argv[i]
        elif a == "--force-research":
            out["force_research"] = True
        elif a in ("-h", "--help"):
            out["mode"] = "help"
        else:
            out["symbols"].append(a)
        i += 1
    return out


HELP = """Usage:
  python alpaca_codegen.py [--stage <name>] [--force-research] [SYM ...]
      Run the pipeline. Stage filter:
        missing  (default)  -- generate only files absent from generated/
        all                 -- regenerate every stage, overwriting existing
        subscriber          -- regenerate only subscriber.py
        trader              -- regenerate only trader.py
        cli                 -- regenerate only cli.py
        trade_stream        -- regenerate only trade_stream.py
        persister           -- regenerate only persister.py
        webgui_html         -- regenerate only generated/web/index.html
        webgui              -- regenerate only webgui.py (Flask backend)

  python alpaca_codegen.py --fix <error_file|-> --stage <name> [SYM ...]
      Re-engage the Coder for one stage with a runtime traceback as feedback.

  Environment:
    GEMINI_MODEL  override the OpenRouter model id
                  (default: google/gemini-3.1-pro-preview)
"""


def main():
    parsed = _parse_args(sys.argv[1:])
    if parsed["mode"] == "help":
        print(HELP)
        return
    symbols = parsed["symbols"] or DEFAULT_SYMBOLS
    if parsed["mode"] == "fix":
        ef = parsed["error_file"]
        if ef == "-":
            error_text = sys.stdin.read()
        elif ef and Path(ef).exists():
            error_text = Path(ef).read_text()
        else:
            raise SystemExit(f"--fix: error file not found: {ef!r}")
        if parsed["stage"] == "missing":
            raise SystemExit(
                "--fix requires --stage <subscriber|trader|cli>."
            )
        run_fix(error_text, parsed["stage"], symbols)
        return
    run_pipeline(symbols, stage_filter=parsed["stage"],
                 force_research=parsed["force_research"])


if __name__ == "__main__":
    main()
