## Part A -- Alpaca Crypto Market Data Websocket

The Alpaca Crypto Market Data Websocket provides real-time streaming data for various cryptocurrencies.

**Websocket URL:**

*   **Live:** `wss://stream.data.alpaca.markets/v1beta3/crypto/us`
*   **Paper:** `wss://stream.data.alpaca.markets/v1beta3/crypto/us` (The distinction between paper and live for crypto data streams is handled by the API keys used for authentication, not different URLs.)

**Authentication:**

Authentication is performed by sending an `auth` message after establishing the websocket connection.

```json
{
    "action": "auth",
    "key": "YOUR_API_KEY_ID",
    "secret": "YOUR_SECRET_KEY"
}
```

**Subscribe Message Shape (Trades and Quotes):**

To subscribe to trades and quotes for a list of symbols (e.g., `BTC/USD`, `ETH/USD`), send a `subscribe` message.

```json
{
    "action": "subscribe",
    "trades": ["BTC/USD", "ETH/USD"],
    "quotes": ["BTC/USD", "ETH/USD"]
}
```

**Message Shape from Server:**

*   **Trades (`t`):**
    ```json
    {
        "T": "t",          // Type: Trade
        "S": "BTC/USD",    // Symbol
        "i": 123456789,    // Trade ID # TODO verify field name
        "x": "CBSE",       // Exchange # TODO verify field name
        "p": 30000.50,     // Price
        "s": 0.001,        // Size (quantity)
        "t": "2023-10-27T10:00:00.123456789Z", // Timestamp (RFC3339 nano)
        "c": ["@"]         // Conditions # TODO verify field name and possible values
    }
    ```

*   **Quotes (`q`):**
    ```json
    {
        "T": "q",          // Type: Quote
        "S": "BTC/USD",    // Symbol
        "x": "CBSE",       // Exchange # TODO verify field name
        "bp": 30000.00,    // Bid Price
        "bs": 0.002,       // Bid Size (quantity)
        "ap": 30001.00,    // Ask Price
        "as": 0.001,       // Ask Size (quantity)
        "t": "2023-10-27T10:00:00.123456789Z"  // Timestamp (RFC3339 nano)
    }
    ```

**Recommended Python SDK (`alpaca-py`) Classes:**

The `alpaca-py` library provides convenient wrappers for interacting with the crypto data websocket.

*   **`CryptoDataStream`:**
    *   **Constructor:**
        ```python
        from alpaca.data.live import CryptoDataStream

        # For paper trading (using paper API keys)
        crypto_stream = CryptoDataStream(
            api_key="YOUR_API_KEY_ID",
            secret_key="YOUR_SECRET_KEY",
            raw_data=False # Set to True to receive raw dicts instead of Pydantic models
        )
        ```
    *   **`subscribe_trades`:**
        ```python
        async def trade_handler(trade):
            print(f"Trade: {trade.symbol} - Price: {trade.price}, Size: {trade.size}")

        crypto_stream.subscribe_trades(trade_handler, "BTC/USD", "ETH/USD")
        ```
        *   **Async Handler Signature:** `async def handler(trade: Trade) -> None`
            (where `Trade` is a Pydantic model representing the trade data)
    *   **`subscribe_quotes`:**
        ```python
        async def quote_handler(quote):
            print(f"Quote: {quote.symbol} - Bid: {quote.bid_price}, Ask: {quote.ask_price}")

        crypto_stream.subscribe_quotes(quote_handler, "BTC/USD", "ETH/USD")
        ```
        *   **Async Handler Signature:** `async def handler(quote: Quote) -> None`
            (where `Quote` is a Pydantic model representing the quote data)
    *   **`run()`:**
        Starts the websocket connection and listens for messages. This is a blocking call.
        ```python
        # In your main async function
        await crypto_stream.run()
        ```

## Part B -- Alpaca Trading REST API

The Alpaca Trading REST API allows programmatic trading and account management.

**REST Endpoints:**

*   **Live:** `https://api.alpaca.markets/v2`
*   **Paper:** `https://paper-api.alpaca.markets/v2`

**Relevant `alpaca.trading` SDK Surface:**

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
```

*   **`TradingClient` Constructor:**
    ```python
    # For paper trading
    trading_client = TradingClient(
        api_key="YOUR_API_KEY_ID",
        secret_key="YOUR_SECRET_KEY",
        paper=True
    )

    # For live trading
    # trading_client = TradingClient(
    #     api_key="YOUR_API_KEY_ID",
    #     secret_key="YOUR_SECRET_KEY",
    #     paper=False
    # )
    ```

*   **`submit_order(order_data)`:** Submits a new order.
    *   **Example Market Order:**
        ```python
        market_order_data = MarketOrderRequest(
            symbol="BTC/USD",
            qty=0.001,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC # Or IOC for crypto
        )
        submitted_order = trading_client.submit_order(market_order_data)
        ```
    *   **Example Limit Order:**
        ```python
        limit_order_data = LimitOrderRequest(
            symbol="ETH/USD",
            qty=0.01,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.IOC, # Or GTC for crypto
            limit_price=2000.00
        )
        submitted_order = trading_client.submit_order(limit_order_data)
        ```

*   **`get_account()`:** Retrieves account information.
    ```python
    account = trading_client.get_account()
    print(f"Account equity: {account.equity}")
    ```

*   **`get_all_positions()`:** Retrieves all open positions.
    ```python
    positions = trading_client.get_all_positions()
    for position in positions:
        print(f"Symbol: {position.symbol}, Quantity: {position.qty}")
    ```

*   **`get_orders()`:** Retrieves a list of orders. Can be filtered by status, symbol, etc.
    ```python
    all_orders = trading_client.get_orders()
    # pending_orders = trading_client.get_orders(status='open') # Example filter
    ```

*   **`cancel_order_by_id(order_id)`:** Cancels a specific order by its ID.
    ```python
    # Assuming 'submitted_order' from submit_order example
    trading_client.cancel_order_by_id(submitted_order.id)
    ```

*   **`cancel_orders()`:** Cancels all open orders.
    ```python
    trading_client.cancel_orders()
    ```

**`time_in_force` for Crypto Symbols:**

For crypto symbols (e.g., `BTC/USD`), the supported `time_in_force` values are:
*   `TimeInForce.GTC` (Good 'Til Canceled)
*   `TimeInForce.IOC` (Immediate Or Cancel)
`TimeInForce.DAY` is **NOT** supported for crypto orders.

**Quantity vs Notional:**

*   **`qty=`:** Use `qty=` when you want to specify the exact amount of the base asset to buy or sell (e.g., `0.001` BTC). This is the most common approach for crypto.
*   **`notional=`:** Use `notional=` when you want to specify the total dollar amount (in the quote currency) you wish to spend or receive (e.g., `$100` worth of BTC). The system will calculate the corresponding quantity based on the current market price.

    ```python
    # Buy $100 worth of BTC
    notional_order_data = MarketOrderRequest(
        symbol="BTC/USD",
        notional=100.00,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC
    )
    ```
    You should use either `qty` or `notional`, but not both, in an order request.

**Required Environment Variables:**

The `alpaca-py` SDK automatically picks up API keys from environment variables if not provided directly to the `TradingClient` constructor.

*   `ALPACA_API_KEY_ID`
*   `ALPACA_API_SECRET_KEY`

Note that for paper trading, you use your paper API keys, which are the same format as live keys but are associated with your paper account. The `paper=True` flag in the `TradingClient` constructor directs the client to the paper trading endpoint.
