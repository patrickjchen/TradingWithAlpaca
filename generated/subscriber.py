import asyncio
import os
import redis
import json
from datetime import datetime
import sys

from alpaca.data.live.crypto import CryptoDataStream
from alpaca.data.models import Trade, Quote

# Default symbols to subscribe to
SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD']

# Redis connection
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

def _clean(d: dict) -> dict:
    """Removes None values from a dictionary."""
    return {k: v for k, v in d.items() if v is not None}

def _to_iso_string(dt: datetime) -> str:
    """Converts a datetime object to an ISO 8601 string."""
    return dt.isoformat() if dt else None

async def trade_handler(trade: Trade):
    """
    Handles incoming trade data, stores it in Redis.
    """
    symbol = trade.symbol
    # Store latest trade data
    latest_data = _clean({
        "price": trade.price,
        "size": trade.size,
        "timestamp": _to_iso_string(trade.timestamp),
        "exchange": trade.exchange,
        "id": trade.id,
        "conditions": ','.join(trade.conditions or []) # conditions can be None
    })
    if latest_data:
        r.hset(f"alpaca:latest:{symbol}", mapping=latest_data)

    # Store recent trades (LPUSH and LTRIM)
    trade_json = json.dumps(_clean({
        "symbol": symbol,
        "price": trade.price,
        "size": trade.size,
        "timestamp": _to_iso_string(trade.timestamp),
        "exchange": trade.exchange,
        "id": trade.id,
        "conditions": trade.conditions or []
    }), default=str)
    r.lpush(f"alpaca:recent:{symbol}:trade", trade_json)
    r.ltrim(f"alpaca:recent:{symbol}:trade", 0, 99)

    # Add symbol to the set of known symbols
    r.sadd("alpaca:symbols", symbol)

async def quote_handler(quote: Quote):
    """
    Handles incoming quote data, stores it in Redis.
    """
    symbol = quote.symbol
    # Store latest quote data
    latest_data = _clean({
        "bid_price": quote.bid_price,
        "bid_size": quote.bid_size,
        "ask_price": quote.ask_price,
        "ask_size": quote.ask_size,
        "timestamp": _to_iso_string(quote.timestamp),
        "bid_exchange": quote.bid_exchange,
        "ask_exchange": quote.ask_exchange,
    })
    if latest_data:
        r.hset(f"alpaca:latest:{symbol}", mapping=latest_data)

    # Store recent quotes (LPUSH and LTRIM)
    quote_json = json.dumps(_clean({
        "symbol": symbol,
        "bid_price": quote.bid_price,
        "bid_size": quote.bid_size,
        "ask_price": quote.ask_price,
        "ask_size": quote.ask_size,
        "timestamp": _to_iso_string(quote.timestamp),
        "bid_exchange": quote.bid_exchange,
        "ask_exchange": quote.ask_exchange,
    }), default=str)
    r.lpush(f"alpaca:recent:{symbol}:quote", quote_json)
    r.ltrim(f"alpaca:recent:{symbol}:quote", 0, 99)

    # Add symbol to the set of known symbols
    r.sadd("alpaca:symbols", symbol)

async def main(symbols_to_subscribe: list):
    api_key_id = os.environ.get("ALPACA_API_KEY_ID")
    api_secret_key = os.environ.get("ALPACA_API_SECRET_KEY")

    if not api_key_id or not api_secret_key:
        print("Error: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set as environment variables.", file=sys.stderr)
        sys.exit(1)

    stream = CryptoDataStream(api_key_id, api_secret_key)

    print(f"Subscribing to trades and quotes for: {', '.join(symbols_to_subscribe)}")
    stream.subscribe_trades(trade_handler, *symbols_to_subscribe)
    stream.subscribe_quotes(quote_handler, *symbols_to_subscribe)

    try:
        await stream._run_forever()
    finally:
        print("Closing stream...")
        await stream.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Alpaca Crypto Data Subscriber")
    parser.add_argument(
        "symbols",
        nargs="*",
        default=SYMBOLS,
        help=f"Crypto symbols to subscribe to (e.g., BTC/USD ETH/USD). Defaults to {', '.join(SYMBOLS)}."
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.symbols))
    except KeyboardInterrupt:
        print("Subscriber stopped by user.")
