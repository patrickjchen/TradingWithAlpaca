import os
import sys
import json
import asyncio
import argparse
import redis
from alpaca.data.live.crypto import CryptoDataStream

DEFAULT_SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD']

def _clean(d):
    """Remove None values from a dictionary for Redis HSET compatibility."""
    return {k: v for k, v in d.items() if v is not None}

async def main():
    api_key = os.environ.get("ALPACA_API_KEY_ID")
    secret_key = os.environ.get("ALPACA_API_SECRET_KEY")
    
    if not api_key or not secret_key:
        sys.exit("Error: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables must be set.")

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    r = redis.Redis.from_url(redis_url, decode_responses=True)

    parser = argparse.ArgumentParser(description="Alpaca Crypto Market Data Subscriber")
    parser.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS, help="List of crypto symbols to subscribe to")
    args = parser.parse_args()
    symbols = args.symbols

    stream = CryptoDataStream(api_key=api_key, secret_key=secret_key)

    async def trade_handler(trade):
        symbol = trade.symbol
        timestamp_iso = trade.timestamp.isoformat() if trade.timestamp else None
        
        data = {
            "price": float(trade.price),
            "size": float(trade.size),
            "timestamp": timestamp_iso,
            "exchange": str(trade.exchange) if trade.exchange else None,
            "conditions": trade.conditions or [],
            "id": trade.id
        }
        
        # Update latest fields
        latest_mapping = _clean({
            "price": data["price"],
            "timestamp": data["timestamp"]
        })
        if latest_mapping:
            r.hset(f"alpaca:latest:{symbol}", mapping=latest_mapping)
            
        # Push to recent trades
        recent_key = f"alpaca:recent:{symbol}:trade"
        r.lpush(recent_key, json.dumps(data, default=str))
        r.ltrim(recent_key, 0, 99)
        
        # Add to known symbols
        r.sadd("alpaca:symbols", symbol)

    async def quote_handler(quote):
        symbol = quote.symbol
        timestamp_iso = quote.timestamp.isoformat() if quote.timestamp else None
        
        data = {
            "bid_price": float(quote.bid_price),
            "bid_size": float(quote.bid_size),
            "ask_price": float(quote.ask_price),
            "ask_size": float(quote.ask_size),
            "timestamp": timestamp_iso,
            "bid_exchange": str(quote.bid_exchange) if quote.bid_exchange else None,
            "ask_exchange": str(quote.ask_exchange) if quote.ask_exchange else None
        }
        
        # Update latest fields
        latest_mapping = _clean({
            "bid_price": data["bid_price"],
            "bid_size": data["bid_size"],
            "ask_price": data["ask_price"],
            "ask_size": data["ask_size"],
            "timestamp": data["timestamp"]
        })
        if latest_mapping:
            r.hset(f"alpaca:latest:{symbol}", mapping=latest_mapping)
            
        # Push to recent quotes
        recent_key = f"alpaca:recent:{symbol}:quote"
        r.lpush(recent_key, json.dumps(data, default=str))
        r.ltrim(recent_key, 0, 99)
        
        # Add to known symbols
        r.sadd("alpaca:symbols", symbol)

    async def orderbook_handler(orderbook):
        # Alpaca's crypto orderbook websocket delivers DELTAS, not full
        # snapshots: each message contains only the levels that changed.
        # Merge into per-side HASHes (HSET on size>0, HDEL on size==0)
        # so the accumulated book lives in Redis. NEVER overwrite the
        # whole book or you lose all but the most recent delta.
        symbol = orderbook.symbol
        timestamp_iso = orderbook.timestamp.isoformat() if orderbook.timestamp else None

        bids_key = f"alpaca:ob:bids:{symbol}"
        asks_key = f"alpaca:ob:asks:{symbol}"

        def _pf(p):  # canonical price field-name so updates land idempotently
            return f"{float(p):.10g}"

        pipe = r.pipeline()
        for lv in (orderbook.bids or []):
            field = _pf(lv.price)
            size = float(lv.size)
            if size > 0:
                pipe.hset(bids_key, field, size)
            else:
                pipe.hdel(bids_key, field)
        for lv in (orderbook.asks or []):
            field = _pf(lv.price)
            size = float(lv.size)
            if size > 0:
                pipe.hset(asks_key, field, size)
            else:
                pipe.hdel(asks_key, field)

        # Metadata hash (book itself lives in the per-side hashes above).
        meta = _clean({"symbol": symbol, "timestamp": timestamp_iso})
        if meta:
            pipe.hset(f"alpaca:orderbook:{symbol}", mapping=meta)

        pipe.sadd("alpaca:symbols", symbol)
        pipe.execute()

    print(f"Subscribing to trades, quotes, and orderbooks for: {', '.join(symbols)}")
    stream.subscribe_trades(trade_handler, *symbols)
    stream.subscribe_quotes(quote_handler, *symbols)
    stream.subscribe_orderbooks(orderbook_handler, *symbols)

    try:
        await stream._run_forever()
    finally:
        await stream.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting subscriber...")
