import os
import sys
import json
import asyncio
import argparse
from typing import Any

import redis
from alpaca.trading.stream import TradingStream
from alpaca.trading.models import TradeUpdate

# Global Redis client, initialized in main() to avoid network calls on import
r = None


def _to_dict(obj: Any) -> Any:
    """
    Serialize SDK responses to dictionaries.
    Prefers model_dump(mode='json') to handle UUIDs/datetimes safely.
    """
    if obj is None:
        return None
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode='json')
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


async def on_trade_update(update: TradeUpdate):
    """
    Callback for trade updates from the Alpaca trading websocket.
    Serializes the event and publishes it to Redis.
    """
    global r
    
    order_obj = getattr(update, "order", None)
    order_dict = _to_dict(order_obj)
    safe_order = order_dict or {}
    
    event = getattr(update, "event", None)
    if hasattr(event, "value"):
        event = event.value
        
    execution_id = getattr(update, "execution_id", None)
    order_id = safe_order.get("id")
    symbol = safe_order.get("symbol")
    side = safe_order.get("side")
    price = getattr(update, "price", None)
    qty = getattr(update, "qty", None)
    ts = getattr(update, "timestamp", None)
    
    payload = {
        "event": str(event) if event is not None else "",
        "execution_id": str(execution_id) if execution_id is not None else "",
        "order_id": str(order_id) if order_id is not None else "",
        "symbol": str(symbol) if symbol is not None else "",
        "side": str(side) if side is not None else "",
        "price": price if price is not None else None,
        "qty": qty if qty is not None else None,
        "ts": str(ts) if ts is not None else "",
        "order": order_dict
    }
    
    payload_json = json.dumps(payload, default=str)
    
    if r is not None:
        # 1. Publish to the durable Redis Stream for persister.py
        r.xadd(
            "alpaca:trade_updates",
            {"json": payload_json},
            maxlen=100000,
            approximate=True
        )
        
        # 2. Maintain a recent list for at-a-glance debugging
        r.lpush("alpaca:recent:trade_updates", payload_json)
        r.ltrim("alpaca:recent:trade_updates", 0, 99)
    
    # 3. Print a one-line status to stdout
    print(f"[{payload['ts']}] {payload['event']} order={payload['order_id'][:8]} {payload['symbol']} {payload['side']} {payload['qty']}")


async def main():
    global r
    
    parser = argparse.ArgumentParser(description="Alpaca Trade Updates Stream")
    parser.add_argument("--live", action="store_true", help="Use live trading API")
    args = parser.parse_args()
    
    is_live = args.live or os.environ.get("ALPACA_LIVE") == "1"
    
    api_key = os.environ.get("ALPACA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_API_SECRET_KEY")
    
    if not api_key or not api_secret:
        print("Error: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables must be set.")
        sys.exit(1)
        
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    
    stream = TradingStream(api_key, api_secret, paper=not is_live)
    stream.subscribe_trade_updates(on_trade_update)
    
    print(f"Subscribing to Alpaca trading stream (paper={not is_live})...")
    try:
        await stream._run_forever()
    finally:
        await stream.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting trade stream...")
