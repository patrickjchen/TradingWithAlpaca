import os
import sys
import json
import sqlite3
import redis

def setup_db(db_path: str) -> sqlite3.Connection:
    """Initialize the SQLite database and create tables/indexes if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders(
            order_id TEXT PRIMARY KEY,
            client_order_id TEXT,
            symbol TEXT,
            side TEXT,
            type TEXT,
            qty REAL,
            limit_price REAL,
            filled_qty REAL,
            filled_avg_price REAL,
            status TEXT,
            submitted_at TEXT,
            updated_at TEXT,
            created_at TEXT,
            raw_json TEXT
        )
    ''')

    # Migration: add limit_price column if upgrading from an older schema.
    cols = {row[1] for row in cursor.execute("PRAGMA table_info(orders)").fetchall()}
    if "limit_price" not in cols:
        cursor.execute("ALTER TABLE orders ADD COLUMN limit_price REAL")
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fills(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            execution_id TEXT UNIQUE,
            order_id TEXT,
            symbol TEXT,
            side TEXT,
            price REAL,
            qty REAL,
            event TEXT,
            ts TEXT,
            raw_json TEXT,
            FOREIGN KEY(order_id) REFERENCES orders(order_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT,
            order_id TEXT,
            ts TEXT,
            raw_json TEXT
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_ts ON trade_events(ts)')
    
    conn.commit()
    return conn

def handle_payload(conn: sqlite3.Connection, payload: dict):
    """Process a single trade update payload and write to SQLite."""
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
    cursor = conn.cursor()
    
    # 1. Always INSERT into trade_events
    cursor.execute(
        "INSERT INTO trade_events(event, order_id, ts, raw_json) VALUES(?, ?, ?, ?)",
        (event, order_id, ts, raw_json)
    )
    
    # 2. Upsert the order if order_id is present
    if order_id:
        o_client_order_id = order.get("client_order_id")
        o_symbol = order.get("symbol") or symbol
        o_side = order.get("side") or side
        o_type = order.get("type")
        
        o_qty_raw = order.get("qty") if order.get("qty") is not None else qty
        o_qty = float(o_qty_raw) if o_qty_raw is not None else None
        
        o_filled_qty_raw = order.get("filled_qty")
        o_filled_qty = float(o_filled_qty_raw) if o_filled_qty_raw is not None else None

        o_filled_avg_price_raw = order.get("filled_avg_price")
        o_filled_avg_price = float(o_filled_avg_price_raw) if o_filled_avg_price_raw is not None else None

        o_limit_price_raw = order.get("limit_price")
        o_limit_price = float(o_limit_price_raw) if o_limit_price_raw is not None else None

        o_status = order.get("status")
        o_submitted_at = order.get("submitted_at")
        o_updated_at = order.get("updated_at")
        o_created_at = order.get("created_at")

        cursor.execute('''
            INSERT INTO orders(order_id, client_order_id, symbol, side, type,
                qty, limit_price, filled_qty, filled_avg_price, status,
                submitted_at, updated_at, created_at, raw_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                limit_price=COALESCE(excluded.limit_price, orders.limit_price),
                filled_qty=excluded.filled_qty,
                filled_avg_price=excluded.filled_avg_price,
                status=excluded.status,
                updated_at=excluded.updated_at,
                raw_json=excluded.raw_json;
        ''', (
            order_id, o_client_order_id, o_symbol, o_side, o_type,
            o_qty, o_limit_price, o_filled_qty, o_filled_avg_price, o_status,
            o_submitted_at, o_updated_at, o_created_at, raw_json
        ))
        
    # 3. If event is a fill, INSERT OR IGNORE into fills
    if event in ('fill', 'partial_fill'):
        f_price = float(price) if price is not None else None
        f_qty = float(qty) if qty is not None else None
        
        cursor.execute('''
            INSERT OR IGNORE INTO fills(execution_id, order_id, symbol, side, price, qty, event, ts, raw_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            execution_id, order_id, symbol, side, f_price, f_qty, event, ts, raw_json
        ))

def main():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    db_path = os.environ.get("ALPACA_DB_PATH", "./alpaca.db")
    group = os.environ.get("PERSISTER_GROUP", "persister")
    consumer = os.environ.get("PERSISTER_CONSUMER", "persister-1")
    stream_name = "alpaca:trade_updates"
    
    print("Starting Alpaca Trade Persister...")
    print(f"  Consumer Group: {group}")
    print(f"  Consumer Name:  {consumer}")
    print(f"  DB Path:        {db_path}")
    print(f"  Redis URL:      {redis_url}")
    
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    conn = setup_db(db_path)
    
    # Idempotently create the consumer group
    try:
        r.xgroup_create(stream_name, group, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
            
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
                        
                        ts = payload.get("ts")
                        event = payload.get("event")
                        order_id = payload.get("order_id")
                        symbol = payload.get("symbol")
                        side = payload.get("side")
                        qty = payload.get("qty")
                        
                        print(f"[{ts}] {event} order={(order_id or '')[:8]} {symbol} {side} {qty}")
                        
                    except Exception as e:
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
