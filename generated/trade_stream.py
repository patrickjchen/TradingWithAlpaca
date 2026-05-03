import os
import sys
import json
import asyncio
import argparse
from typing import Any

import redis
from alpaca.trading.stream import TradingStream
from alpaca.trading.models import TradeUpdate


def _to_dict(obj: Any) -> Any:
    """
    Serialize SDK responses to plain Python dicts/lists.
    Prefers model_dump(mode='json') for Pydantic v2, falls back to dict() or __dict__.
    """
    if obj is None:
        return None
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def make_trade_update_handler(r: redis.Redis):
    async def on_trade_update(update: TradeUpdate):
        order_dict = _to_dict(update.order) if update.order else {}
        
        event = update.event
        event_str = str(getattr(event, "value", event))
        
        execution_id = update.execution_id
        order_id = order_dict.get("id")
        symbol = order_dict.get("symbol")
        side = order_dict.get("side")
        price = update.price
        qty = update.qty
        ts = update.timestamp
        
        payload = {
            "event": event_str,
            "execution_id": execution_id,
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "price": price,
            "qty": qty,
            "ts": ts,
            "order": order_dict
        }
        
        payload_json = json.dumps(payload, default=str)
        
        # 1. Publish to the main stream for persister.py
        r.xadd(
            "alpaca:trade_updates",
            {"json": payload_json},
            maxlen=100000,
            approximate=True
        )
        
        # 2. Maintain a recent list for at-a-glance debug
        r.lpush("alpaca:recent:trade_updates", payload_json)
        r.ltrim("alpaca:recent:trade_updates", 0, 99)
        
        # 3. Print one-line status
        order_id_str = str(order_id)[:8] if order_id else ""
        print(f"[{ts}] {event_str} order={order_id_str} {symbol} {side} {qty}")
        
    return on_trade_update


async def main():
    parser = argparse.ArgumentParser(description="Alpaca Trade Updates Streamer")
    parser.add_argument("--live", action="store_true", help="Use live trading API")
    args = parser.parse_args()

    live = args.live or os.environ.get("ALPACA_LIVE") == "1"

    api_key = os.environ.get("ALPACA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_API_SECRET_KEY")

    if not api_key or not api_secret:
        print("Error: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables must be set.")
        sys.exit(1)

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    r = redis.Redis.from_url(redis_url, decode_responses=True)

    stream = TradingStream(api_key, api_secret, paper=not live)
    
    handler = make_trade_update_handler(r)
    stream.subscribe_trade_updates(handler)

    print(f"Starting trade stream (paper={not live})...")
    try:
        await stream._run_forever()
    finally:
        await stream.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting trade stream...")
