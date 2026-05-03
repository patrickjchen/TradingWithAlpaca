import os
import sys
import json
import argparse
import redis
import trader

def main():
    parser = argparse.ArgumentParser(description="Alpaca Crypto CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Read-only Redis queries
    subparsers.add_parser("symbols", help="List all known symbols from Redis")

    p_latest = subparsers.add_parser("latest", help="Get latest data for a symbol")
    p_latest.add_argument("symbol", help="Crypto symbol (e.g., BTC/USD)")

    p_recent = subparsers.add_parser("recent", help="Get recent trades or quotes for a symbol")
    p_recent.add_argument("symbol", help="Crypto symbol (e.g., BTC/USD)")
    p_recent.add_argument("--kind", choices=["trade", "quote"], default="trade", help="Kind of data to retrieve")
    p_recent.add_argument("-n", type=int, default=10, help="Number of recent items to retrieve")

    # Trading subcommands
    p_account = subparsers.add_parser("account", help="Get account information")
    p_account.add_argument("--live", action="store_true", help="Use live trading account")

    p_positions = subparsers.add_parser("positions", help="Get all open positions")
    p_positions.add_argument("--live", action="store_true", help="Use live trading account")

    p_orders = subparsers.add_parser("orders", help="Get a list of orders")
    p_orders.add_argument("--status", default="open", help="Order status filter (open, closed, all)")
    p_orders.add_argument("--limit", type=int, default=50, help="Maximum number of orders to return")
    p_orders.add_argument("--live", action="store_true", help="Use live trading account")

    p_buy = subparsers.add_parser("buy", help="Place a buy order")
    p_buy.add_argument("symbol", help="Crypto symbol (e.g., BTC/USD)")
    p_buy.add_argument("qty", type=float, help="Quantity to buy")
    p_buy.add_argument("--limit", type=float, help="Limit price (if omitted, places a market order)")
    p_buy.add_argument("--live", action="store_true", help="Use live trading account")

    p_sell = subparsers.add_parser("sell", help="Place a sell order")
    p_sell.add_argument("symbol", help="Crypto symbol (e.g., BTC/USD)")
    p_sell.add_argument("qty", type=float, help="Quantity to sell")
    p_sell.add_argument("--limit", type=float, help="Limit price (if omitted, places a market order)")
    p_sell.add_argument("--live", action="store_true", help="Use live trading account")

    p_cancel = subparsers.add_parser("cancel", help="Cancel a specific order")
    p_cancel.add_argument("order_id", help="ID of the order to cancel")
    p_cancel.add_argument("--live", action="store_true", help="Use live trading account")

    p_cancel_all = subparsers.add_parser("cancel-all", help="Cancel all open orders")
    p_cancel_all.add_argument("--live", action="store_true", help="Use live trading account")

    args = parser.parse_args()

    # Handle Redis read-only commands
    if args.command in ["symbols", "latest", "recent"]:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = redis.Redis.from_url(redis_url, decode_responses=True)

        if args.command == "symbols":
            data = r.smembers("alpaca:symbols")
            if not data:
                sys.exit("Error: No symbols found in Redis.")
            print(json.dumps(list(data), indent=2, default=str))

        elif args.command == "latest":
            data = r.hgetall(f"alpaca:latest:{args.symbol}")
            if not data:
                sys.exit(f"Error: No latest data found for {args.symbol}.")
            print(json.dumps(data, indent=2, default=str))

        elif args.command == "recent":
            key = f"alpaca:recent:{args.symbol}:{args.kind}"
            items = r.lrange(key, 0, args.n - 1)
            if not items:
                sys.exit(f"Error: No recent {args.kind} data found for {args.symbol}.")
            parsed = [json.loads(item) for item in items]
            print(json.dumps(parsed, indent=2, default=str))

    # Handle Trading commands
    else:
        if getattr(args, "live", False):
            print("WARNING: --live mode -- this places real orders", file=sys.stderr)

        res = None
        if args.command == "account":
            res = trader.account(live=args.live)
        elif args.command == "positions":
            res = trader.positions(live=args.live)
        elif args.command == "orders":
            res = trader.orders(live=args.live, status=args.status, limit=args.limit)
        elif args.command == "buy":
            if args.limit is not None:
                res = trader.buy_limit(args.symbol, args.qty, args.limit, live=args.live)
            else:
                res = trader.buy_market(args.symbol, args.qty, live=args.live)
        elif args.command == "sell":
            if args.limit is not None:
                res = trader.sell_limit(args.symbol, args.qty, args.limit, live=args.live)
            else:
                res = trader.sell_market(args.symbol, args.qty, live=args.live)
        elif args.command == "cancel":
            trader.cancel(args.order_id, live=args.live)
            res = {"status": "cancelled", "order_id": args.order_id}
        elif args.command == "cancel-all":
            res = trader.cancel_all(live=args.live)

        print(json.dumps(res, indent=2, default=str))

if __name__ == "__main__":
    main()
