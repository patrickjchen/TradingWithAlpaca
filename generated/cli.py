import argparse
import json
import os
import sys
import redis

import trader

# Redis connection
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def print_json(data):
    """Helper to pretty-print JSON data."""
    print(json.dumps(data, indent=2, default=str))


def check_live(args):
    """Print a warning to stderr if --live mode is enabled."""
    if getattr(args, "live", False):
        print("WARNING: --live mode -- this places real orders", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Alpaca Crypto CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Read-only Redis queries
    subparsers.add_parser("symbols", help="List all known symbols from Redis")

    p_latest = subparsers.add_parser("latest", help="Get latest data for a symbol")
    p_latest.add_argument("symbol", help="Crypto symbol (e.g., BTC/USD)")

    p_recent = subparsers.add_parser("recent", help="Get recent data for a symbol")
    p_recent.add_argument("symbol", help="Crypto symbol (e.g., BTC/USD)")
    p_recent.add_argument("--kind", choices=["trade", "quote"], default="trade", help="Data kind (trade or quote)")
    p_recent.add_argument("-n", type=int, default=10, help="Number of recent records to fetch")

    # Trading subcommands
    p_account = subparsers.add_parser("account", help="Get account information")
    p_account.add_argument("--live", action="store_true", help="Use live trading account")

    p_positions = subparsers.add_parser("positions", help="Get all open positions")
    p_positions.add_argument("--live", action="store_true", help="Use live trading account")

    p_orders = subparsers.add_parser("orders", help="Get a list of orders")
    p_orders.add_argument("--status", default="open", help="Order status filter (open, closed, all)")
    p_orders.add_argument("--limit", type=int, default=50, help="Limit number of returned orders")
    p_orders.add_argument("--live", action="store_true", help="Use live trading account")

    p_buy = subparsers.add_parser("buy", help="Submit a buy order")
    p_buy.add_argument("symbol", help="Crypto symbol (e.g., BTC/USD)")
    p_buy.add_argument("qty", type=float, help="Quantity to buy")
    p_buy.add_argument("--limit", type=float, help="Limit price (if omitted, places a market order)")
    p_buy.add_argument("--live", action="store_true", help="Use live trading account")

    p_sell = subparsers.add_parser("sell", help="Submit a sell order")
    p_sell.add_argument("symbol", help="Crypto symbol (e.g., BTC/USD)")
    p_sell.add_argument("qty", type=float, help="Quantity to sell")
    p_sell.add_argument("--limit", type=float, help="Limit price (if omitted, places a market order)")
    p_sell.add_argument("--live", action="store_true", help="Use live trading account")

    p_cancel = subparsers.add_parser("cancel", help="Cancel a specific order by ID")
    p_cancel.add_argument("order_id", help="The ID of the order to cancel")
    p_cancel.add_argument("--live", action="store_true", help="Use live trading account")

    p_cancel_all = subparsers.add_parser("cancel-all", help="Cancel all open orders")
    p_cancel_all.add_argument("--live", action="store_true", help="Use live trading account")

    args = parser.parse_args()

    # Execute Redis commands
    if args.command == "symbols":
        data = r.smembers("alpaca:symbols")
        if not data:
            sys.exit("Error: No symbols found in Redis.")
        print_json(list(data))

    elif args.command == "latest":
        data = r.hgetall(f"alpaca:latest:{args.symbol}")
        if not data:
            sys.exit(f"Error: No latest data found for {args.symbol}.")
        print_json(data)

    elif args.command == "recent":
        raw_data = r.lrange(f"alpaca:recent:{args.symbol}:{args.kind}", 0, args.n - 1)
        if not raw_data:
            sys.exit(f"Error: No recent {args.kind} data found for {args.symbol}.")
        parsed_data = [json.loads(item) for item in raw_data]
        print_json(parsed_data)

    # Execute Trading commands
    elif args.command == "account":
        check_live(args)
        print_json(trader.account(args.live))

    elif args.command == "positions":
        check_live(args)
        print_json(trader.positions(args.live))

    elif args.command == "orders":
        check_live(args)
        print_json(trader.orders(args.live, args.status, args.limit))

    elif args.command == "buy":
        check_live(args)
        if args.limit is not None:
            res = trader.buy_limit(args.symbol, args.qty, args.limit, args.live)
        else:
            res = trader.buy_market(args.symbol, args.qty, args.live)
        print_json(res)

    elif args.command == "sell":
        check_live(args)
        if args.limit is not None:
            res = trader.sell_limit(args.symbol, args.qty, args.limit, args.live)
        else:
            res = trader.sell_market(args.symbol, args.qty, args.live)
        print_json(res)

    elif args.command == "cancel":
        check_live(args)
        trader.cancel(args.order_id, args.live)
        print_json({"status": "cancelled", "order_id": args.order_id})

    elif args.command == "cancel-all":
        check_live(args)
        res = trader.cancel_all(args.live)
        print_json(res)


if __name__ == "__main__":
    main()
