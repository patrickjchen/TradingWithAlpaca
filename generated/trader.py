import os
from typing import Any, Dict, List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus


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


def get_client(live: bool = False) -> TradingClient:
    """
    Factory to create and return an Alpaca TradingClient.
    Defaults to paper trading unless live=True.
    """
    key = os.environ["ALPACA_API_KEY_ID"]
    secret = os.environ["ALPACA_API_SECRET_KEY"]
    return TradingClient(key, secret, paper=not live)


def account(live: bool = False) -> dict:
    """Retrieve account information."""
    client = get_client(live)
    return _to_dict(client.get_account())


def positions(live: bool = False) -> list[dict]:
    """Retrieve all open positions."""
    client = get_client(live)
    return _to_dict(client.get_all_positions())


def orders(live: bool = False, status: str = 'open', limit: int = 50) -> list[dict]:
    """Retrieve a list of orders."""
    client = get_client(live)
    
    status_map = {
        'open': QueryOrderStatus.OPEN,
        'closed': QueryOrderStatus.CLOSED,
        'all': QueryOrderStatus.ALL
    }
    req_status = status_map.get(status.lower(), QueryOrderStatus.OPEN)
    req = GetOrdersRequest(status=req_status, limit=limit)
    
    return _to_dict(client.get_orders(filter=req))


def buy_market(symbol: str, qty: float, live: bool = False) -> dict:
    """Submit a market buy order."""
    client = get_client(live)
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC
    )
    return _to_dict(client.submit_order(req))


def sell_market(symbol: str, qty: float, live: bool = False) -> dict:
    """Submit a market sell order."""
    client = get_client(live)
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    return _to_dict(client.submit_order(req))


def buy_limit(symbol: str, qty: float, limit_price: float, live: bool = False) -> dict:
    """Submit a limit buy order."""
    client = get_client(live)
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        limit_price=limit_price,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC
    )
    return _to_dict(client.submit_order(req))


def sell_limit(symbol: str, qty: float, limit_price: float, live: bool = False) -> dict:
    """Submit a limit sell order."""
    client = get_client(live)
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        limit_price=limit_price,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    return _to_dict(client.submit_order(req))


def cancel(order_id: str, live: bool = False) -> None:
    """Cancel a specific order by its ID."""
    client = get_client(live)
    client.cancel_order_by_id(order_id)


def cancel_all(live: bool = False) -> list:
    """Cancel all open orders."""
    client = get_client(live)
    return _to_dict(client.cancel_orders())
