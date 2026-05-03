import os
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
import redis
import trader

WEB_DIR = Path(__file__).parent / "web"
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path='/static')

# Silence Werkzeug's per-request access log (the page polls /api/orderbook
# every 1s -- that's 60 lines/min of noise per browser tab). We emit our
# own throttled one-line summary every LOG_INTERVAL_SEC instead.
logging.getLogger('werkzeug').setLevel(logging.WARNING)

LOG_INTERVAL_SEC = 60
_log_lock = threading.Lock()
_log_counts: dict[str, int] = {}
_last_flush = time.monotonic()


@app.after_request
def _throttled_access_log(resp):
    global _last_flush
    summary = None
    with _log_lock:
        _log_counts[request.path] = _log_counts.get(request.path, 0) + 1
        now = time.monotonic()
        if now - _last_flush >= LOG_INTERVAL_SEC:
            counts = _log_counts.copy()
            _log_counts.clear()
            _last_flush = now
            total = sum(counts.values())
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
            summary = (total, ", ".join(f"{p}:{n}" for p, n in top))
    if summary:
        total, top_str = summary
        print(f"[webgui] {total} requests in last {LOG_INTERVAL_SEC}s ({top_str})", flush=True)
    return resp

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
DB_PATH = os.environ.get("ALPACA_DB_PATH", "./alpaca.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def is_live(req):
    return str(req.args.get('live', '0')).lower() in {"1", "true", "yes"}

@app.route('/')
def index():
    if not (WEB_DIR / "index.html").exists():
        return "web/index.html not generated yet -- run `python alpaca_codegen.py --stage webgui_html`", 200
    return send_from_directory(str(WEB_DIR), "index.html")

@app.route('/api/symbols')
def api_symbols():
    try:
        symbols = r.smembers("alpaca:symbols")
        return jsonify(sorted(list(symbols)))
    except Exception:
        return jsonify([])

@app.route('/api/orderbook/<path:symbol>')
def api_orderbook(symbol):
    try:
        # The book is stored as two per-side HASHes (price -> size) maintained
        # by subscriber.py via merge-of-deltas; the metadata HASH only holds
        # symbol + timestamp.
        bid_h = r.hgetall(f"alpaca:ob:bids:{symbol}") or {}
        ask_h = r.hgetall(f"alpaca:ob:asks:{symbol}") or {}

        def _levels(h):
            out = []
            for p_str, s_str in h.items():
                try:
                    p, s = float(p_str), float(s_str)
                except (TypeError, ValueError):
                    continue
                if s > 0:
                    out.append({"price": p, "size": s})
            return out

        bids = _levels(bid_h)
        asks = _levels(ask_h)
        if bids or asks:
            bids.sort(key=lambda x: -x["price"])  # best (highest) first
            asks.sort(key=lambda x:  x["price"])  # best (lowest)  first
            meta = r.hgetall(f"alpaca:orderbook:{symbol}") or {}
            return jsonify({
                "symbol": symbol,
                "timestamp": meta.get("timestamp"),
                "bids": bids[:10],
                "asks": asks[:10],
            })

        latest = r.hgetall(f"alpaca:latest:{symbol}")
        if latest and latest.get('bid_price') is not None and latest.get('ask_price') is not None:
            bids = [{"price": float(latest['bid_price']), "size": float(latest.get('bid_size', 0))}]
            asks = [{"price": float(latest['ask_price']), "size": float(latest.get('ask_size', 0))}]
            return jsonify({
                "symbol": symbol,
                "timestamp": latest.get("timestamp"),
                "bids": bids,
                "asks": asks,
                "fallback": "quote_only"
            })
    except Exception:
        pass

    return jsonify({"symbol": symbol, "bids": [], "asks": [], "fallback": "no_data"})

@app.route('/api/latest/<path:symbol>')
def api_latest(symbol):
    try:
        return jsonify(r.hgetall(f"alpaca:latest:{symbol}"))
    except Exception:
        return jsonify({})

@app.route('/api/orders')
def api_orders():
    limit = request.args.get('limit', 50, type=int)
    status = request.args.get('status')
    conn = get_db()
    try:
        if status:
            rows = conn.execute("SELECT * FROM orders WHERE status = ? ORDER BY submitted_at DESC LIMIT ?", (status, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM orders ORDER BY submitted_at DESC LIMIT ?", (limit,)).fetchall()
        return jsonify([dict(row) for row in rows])
    except sqlite3.OperationalError:
        return jsonify([])
    finally:
        conn.close()

@app.route('/api/fills')
def api_fills():
    limit = request.args.get('limit', 50, type=int)
    conn = get_db()
    try:
        rows = conn.execute("SELECT * FROM fills ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return jsonify([dict(row) for row in rows])
    except sqlite3.OperationalError:
        return jsonify([])
    finally:
        conn.close()

@app.route('/api/summary')
def api_summary():
    live = is_live(request)
    res = {"account": None, "positions": None, "realized_pnl_by_symbol": {}}
    
    try:
        res["account"] = trader.account(live)
    except Exception:
        pass
        
    try:
        res["positions"] = trader.positions(live)
    except Exception:
        pass
        
    conn = get_db()
    try:
        rows = conn.execute("SELECT symbol, side, price, qty FROM fills").fetchall()
        pnl = {}
        for row in rows:
            sym = row['symbol']
            side = row['side']
            price = row['price']
            qty = row['qty']
            if price is not None and qty is not None:
                val = float(price) * float(qty)
                mult = 1 if side == 'sell' else -1
                pnl[sym] = pnl.get(sym, 0.0) + (val * mult)
        res["realized_pnl_by_symbol"] = pnl
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()
        
    return jsonify(res)

@app.route('/api/order', methods=['POST'])
def api_order():
    data = request.get_json() or {}
    symbol = data.get('symbol')
    qty = data.get('qty')
    side = data.get('side')
    order_type = data.get('type')
    limit_price = data.get('limit_price')
    live = bool(data.get('live', False))
    
    if not symbol or not qty or side not in ('buy', 'sell') or order_type not in ('market', 'limit'):
        return jsonify({"error": "Invalid input"}), 400
        
    try:
        qty = float(qty)
        if order_type == 'limit':
            if limit_price is None:
                return jsonify({"error": "limit_price required for limit orders"}), 400
            limit_price = float(limit_price)
            
        if order_type == 'market':
            if side == 'buy':
                res = trader.buy_market(symbol, qty, live)
            else:
                res = trader.sell_market(symbol, qty, live)
        else:
            if side == 'buy':
                res = trader.buy_limit(symbol, qty, limit_price, live)
            else:
                res = trader.sell_limit(symbol, qty, limit_price, live)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e), "type": e.__class__.__name__}), 400

@app.route('/api/order/<order_id>', methods=['DELETE'])
def api_cancel_order(order_id):
    live = is_live(request)
    try:
        trader.cancel(order_id, live=live)
        return jsonify({"cancelled": order_id})
    except Exception as e:
        return jsonify({"error": str(e), "type": e.__class__.__name__}), 400

if __name__ == "__main__":
    host = os.environ.get("WEBGUI_HOST", "127.0.0.1")
    port = int(os.environ.get("WEBGUI_PORT", "5000"))
    print(f"Starting Web GUI at http://{host}:{port}")
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nExiting Web GUI...")
