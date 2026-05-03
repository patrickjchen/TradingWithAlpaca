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
  3. Verifier  (UserProxyAgent + LocalCommandLineCodeExecutor) -> byte-compiles
     each generated file and imports it. We do NOT place live orders or open
     a websocket -- that requires real creds and a Redis server -- but a clean
     compile + import catches the common LLM mistakes.

Up to MAX_RETRIES rounds: on compile failure the error is fed back to the Coder.

Usage:
    python alpaca_codegen.py
        # writes RESEARCH.md + subscriber.py + trader.py + cli.py
        #        + trade_stream.py + persister.py into ./generated/

Then to actually use what was generated:
    pip install alpaca-py redis
    export ALPACA_API_KEY_ID=...  ALPACA_API_SECRET_KEY=...
    # start a redis server (e.g. `docker run -p 6379:6379 redis`)
    python generated/subscriber.py             # market data -> Redis
    python generated/trade_stream.py           # trade updates -> Redis Stream
    python generated/persister.py              # Redis Stream -> SQLite
    python generated/cli.py latest BTC/USD     # query market data
    python generated/cli.py account            # paper account balance
    python generated/cli.py buy BTC/USD 0.001  # paper market buy
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
            # OpenRouter default for Gemini 2.5 Flash truncated the brief at
            # ~400 chars; give the agents room to finish two full files.
            "max_tokens": 8192,
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
  - The subscribe message shape -- specifically how to subscribe to trades
    and quotes for a list of symbols (e.g. `BTC/USD`). DO NOT cover bars --
    we are intentionally NOT subscribing to bars.
  - The shape (field names + types) of the messages the server pushes for
    trades (`t`) and quotes (`q`). Include the symbol field name. Skip bars.
  - The recommended Python SDK (`alpaca-py`) classes that wrap this:
    `CryptoDataStream` -- constructor args, `subscribe_trades` /
    `subscribe_quotes`, async handler signatures, `run()`.

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

Output format: emit EXACTLY ONE fenced ```python``` block containing the
full source of the requested file. No prose, no FILE markers, no extra
fenced blocks. Use only the stdlib plus `alpaca-py`, `redis`. Type-hint
where useful but don't over-engineer. Both `python -m py_compile` and
`import <module>` must succeed.

Durable rules (do NOT regress between iterations):
  - alpaca-py async/sync (subscriber):
      * `CryptoDataStream.run()` is SYNC -- it calls `asyncio.run(...)` itself.
        NEVER `await stream.run()` and NEVER call it from inside another
        `asyncio.run()` -- it deadlocks. Use this pattern instead:
            async def main():
                ...
                stream.subscribe_trades(trade_handler, *symbols)
                stream.subscribe_quotes(quote_handler, *symbols)
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
  - Subscribe to **trades and quotes only** (NOT bars) for the symbols passed
    on the command line (default to the SYMBOLS constant if none given).
  - For each incoming message, write to Redis (apply `_clean(d)` first):
      * `HSET alpaca:latest:<SYMBOL> <field> <value>` -- merge latest fields
        across trades/quotes (price, bid, ask, ts, ...).
      * `LPUSH alpaca:recent:<SYMBOL>:<KIND> <json>` then `LTRIM ... 0 99`
        where KIND is `trade` or `quote`.
      * `SADD alpaca:symbols <SYMBOL>` so the CLI can list them.
  - Convert datetimes to ISO strings before storing. Use `default=str` in
    json.dumps for safety.
  - Async handlers; follow the exact main() pattern in the durable rules.
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

# Stage registry: ordered tuples of (target_filename, system_prompt, prior_files_to_pass).
# Prior files are read from OUT_DIR and given to the Coder as `### EXISTING:` context.
STAGES = [
    ("subscriber.py",   SUBSCRIBER_SYS,   []),
    ("trader.py",       TRADER_SYS,       []),
    ("cli.py",          CLI_SYS,          ["subscriber.py", "trader.py"]),
    # trade_stream: subscribes to Alpaca TradingStream and publishes events
    # into a Redis Stream `alpaca:trade_updates`. trader.py is provided so
    # the LLM can crib the _to_dict serialization helper.
    ("trade_stream.py", TRADE_STREAM_SYS, ["trader.py"]),
    # persister: reads from the Redis Stream and writes to SQLite. No
    # websocket. trade_stream.py is provided so the LLM knows the exact
    # shape of the JSON payloads it will be parsing.
    ("persister.py",    PERSISTER_SYS,    ["trade_stream.py"]),
]
STAGE_BY_NAME = {f[: -len(".py")]: i for i, (f, _, _) in enumerate(STAGES)}


# ---------- Helpers ----------

def extract_block(text: str, lang: str) -> str | None:
    # Greedy on the closing fence: the brief contains nested ```json/python```
    # examples, and a non-greedy match would stop at the first inner fence.
    m = re.search(rf"```{lang}\s*\n(.*)\n```", text, re.DOTALL)
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


def verify_one_file(work_dir: str, target: str, target_code: str,
                    priors: dict[str, str]) -> tuple[bool, str]:
    """Write `target` (plus any `priors` so cross-file imports resolve) into
    work_dir, `python -m py_compile` the target, then `import` the target.

    Importing executes the module body. subscriber.py / cli.py / trader.py
    only do safe top-level work (lazy redis client, env-var reads) -- no
    network calls, so this is safe.
    """
    for name, code in priors.items():
        (Path(work_dir) / name).write_text(code)
    (Path(work_dir) / target).write_text(target_code)

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


def _read_priors(prior_names: list[str]) -> dict[str, str]:
    """Read prior-stage files from OUT_DIR (skipping any that don't exist)."""
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
    compile/import errors fed back to the Coder.
    """
    target_file, system_prompt, prior_names = STAGES[stage_idx]
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
        priors_md += (
            f"\n\n### EXISTING: {name} (already on disk; do NOT redefine, "
            f"use its public surface)\n```python\n{code}\n```"
        )

    work_dir = tempfile.mkdtemp(prefix=f"alpaca_stage_{target_file}_")
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
                  f"\nGenerate {target_file}. Output ONE fenced ```python``` "
                  "block with the full file contents. No prose."
            )
            if feedback:
                ask += (
                    "\n\nPrevious attempt failed verification:\n"
                    f"```\n{feedback.strip()}\n```\n"
                    f"Fix the issue and re-emit {target_file} in full."
                )
            raw = chat_with_retry(user, coder, ask)
            code = extract_block(raw, "python") or ""
            if not code:
                feedback = (
                    "No ```python``` fenced block found in your output. "
                    "Output exactly one fenced ```python``` block, no prose."
                )
                print("            ! no python block; retrying")
                continue
            ok, out = verify_one_file(work_dir, target_file, code, priors)
            if ok:
                print("            verify OK")
                target_path.write_text(code + ("" if code.endswith("\n") else "\n"))
                print(f"            wrote {target_path} ({len(code)} chars)")
                return True
            print("            ! verify failed; feeding errors back")
            feedback = out
        # MAX_RETRIES exhausted -- write last attempt for inspection
        if code:
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

    if stage_filter == "missing":
        to_run = [
            i for i, (fname, _, _) in enumerate(STAGES)
            if not (OUT_DIR / fname).exists()
        ]
        skipped = [
            STAGES[i][0] for i in range(len(STAGES))
            if (OUT_DIR / STAGES[i][0]).exists()
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
        results[STAGES[i][0]] = ok

    print("\n=== Summary ===")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED (last attempt written for inspection)"
        print(f"  {name}: {status}")

    print("\nNext steps (paper trading by default):")
    print("  pip install alpaca-py redis")
    print("  export ALPACA_API_KEY_ID=...  ALPACA_API_SECRET_KEY=...")
    print("  # start redis (e.g. docker run -p 6379:6379 redis)")
    print(f"  python {OUT_DIR/'subscriber.py'}            # market-data -> Redis")
    print(f"  python {OUT_DIR/'trade_stream.py'}          # trade updates -> Redis Stream")
    print(f"  python {OUT_DIR/'persister.py'}             # Redis Stream -> SQLite")
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
    print(f"\n[fix] {STAGES[STAGE_BY_NAME[stage]][0]}: "
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
