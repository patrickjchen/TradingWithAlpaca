import os
import sys
import json
import sqlite3
import redis

def init_db(conn: sqlite3.Connection):
    """Initialize the SQLite schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders(
          order_id        TEXT PRIMARY KEY,
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
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fills(
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          execution_id    TEXT UNIQUE,
          order_id        TEXT,
          symbol          TEXT,
          side            TEXT,
          price           REAL,
          qty             REAL,
          event           TEXT,
          ts              TEXT,
          raw_json        TEXT,
          FOREIGN KEY(order_id) REFERENCES orders(order_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_events(
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          event           TEXT,
          order_id        TEXT,
          ts              TEXT,
          raw_json        TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON trade_events(ts)")
    conn.commit()

def handle_payload(conn: sqlite3.Connection, payload: dict):
    """Process a single trade update payload and persist it to SQLite."""
    event = payload.get("event")
    execution_id = payload.get("execution_id")
    order_id = payload.get("order_id")
    symbol = payload.get("symbol")
    side = payload.get("side")
    price = payload.get("price")
    qty = payload.get("qty")
    ts = payload.get("ts")
    order = payload.get("order") or {}

    raw_json = json.dumps(payload, default=str)

    # 1. Always INSERT into trade_events
    conn.execute(
        "INSERT INTO trade_events(event, order_id, ts, raw_json) VALUES(?, ?, ?, ?)",
        (event, order_id, ts, raw_json)
    )

    # 2. Upsert the order if order_id is present
    if order_id:
        client_order_id = order.get("client_order_id")
        o_symbol = order.get("symbol", symbol)
        o_side = order.get("side", side)
        o_type = order.get("type")
        
        def to_float(val):
            return float(val) if val is not None else None

        o_qty = to_float(order.get("qty", qty))
        o_filled_qty = to_float(order.get("filled_qty"))
        o_filled_avg_price = to_float(order.get("filled_avg_price"))
        o_status = order.get("status")
        o_submitted_at = order.get("submitted_at")
        o_updated_at = order.get("updated_at")
        o_created_at = order.get("created_at")

        conn.execute("""
            INSERT INTO orders(order_id, client_order_id, symbol, side, type,
              qty, filled_qty, filled_avg_price, status,
              submitted_at, updated_at, created_at, raw_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
              filled_qty=excluded.filled_qty,
              filled_avg_price=excluded.filled_avg_price,
              status=excluded.status,
              updated_at=excluded.updated_at,
              raw_json=excluded.raw_json
        """, (
            order_id, client_order_id, o_symbol, o_side, o_type,
            o_qty, o_filled_qty, o_filled_avg_price, o_status,
            o_submitted_at, o_updated_at, o_created_at, raw_json
        ))

    # 3. If event is a fill, INSERT OR IGNORE into fills
    if event in ('fill', 'partial_fill'):
        f_price = float(price) if price is not None else None
        f_qty = float(qty) if qty is not None else None
        conn.execute("""
            INSERT OR IGNORE INTO fills(execution_id, order_id, symbol, side, price, qty, event, ts, raw_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (execution_id, order_id, symbol, side, f_price, f_qty, event, ts, raw_json))

    # Print a one-line status
    order_id_str = (order_id or "")[:8]
    print(f"[{ts}] {event} order={order_id_str} {symbol} {side} {qty}")

def main():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    db_path = os.environ.get("ALPACA_DB_PATH", "./alpaca.db")
    group = os.environ.get("PERSISTER_GROUP", "persister")
    consumer = os.environ.get("PERSISTER_CONSUMER", "persister-1")
    stream_name = "alpaca:trade_updates"

    print("Starting Alpaca Trade Persister...")
    print(f"  Redis URL: {redis_url}")
    print(f"  DB Path:   {db_path}")
    print(f"  Group:     {group}")
    print(f"  Consumer:  {consumer}")

    r = redis.Redis.from_url(redis_url, decode_responses=True)

    # Idempotently create the consumer group
    try:
        r.xgroup_create(stream_name, group, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    conn = sqlite3.connect(db_path)
    init_db(conn)

    try:
        while True:
            resp = r.xreadgroup(
                group, 
                consumer,
                {stream_name: ">"},
                count=64, 
                block=5000
            )
            for _stream, entries in resp or []:
                for entry_id, fields in entries:
                    try:
                        payload = json.loads(fields["json"])
                        handle_payload(conn, payload)
                        conn.commit()
                    except Exception as e:
                        # Don't ack on failure -- let the entry stay pending
                        print(f"  ! error on {entry_id}: {e}", file=sys.stderr)
                        continue
                    
                    r.xack(stream_name, group, entry_id)
    except KeyboardInterrupt:
        print("\nExiting persister...")
    finally:
        conn.close()
        r.close()

if __name__ == "__main__":
    main()
