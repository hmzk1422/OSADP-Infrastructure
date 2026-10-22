"""Test Bybit V5 linear order acceptance using a dedicated testnet account.

Required environment:
  UETS_BYBIT_TESTNET_API_KEY
  UETS_BYBIT_TESTNET_API_SECRET
  UETS_BYBIT_TESTNET_SYMBOL
  UETS_BYBIT_TESTNET_QTY
  UETS_BYBIT_TESTNET_MAX_NOTIONAL_USDT
  UETS_BYBIT_TESTNET_CONFIRM=YES

Optional:
  UETS_BYBIT_TESTNET_POSITION_MODE=one-way|hedge

This script is hard-wired to Bybit testnet; it has no mainnet override.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any


BASE_URL = "https://api-testnet.bybit.com"
RECV_WINDOW = "10000"
POLL_INTERVAL_SECONDS = 0.25
POLL_ATTEMPTS = 20
LIVE_ORDER_STATUSES = {"New", "PartiallyFilled", "Untriggered", "Triggered"}


class BybitTestnetError(RuntimeError):
    pass


def _decimal(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as error:
        raise BybitTestnetError(f"Bybit returned invalid {name}: {value!r}") from error
    if not result.is_finite():
        raise BybitTestnetError(f"Bybit returned non-finite {name}: {value!r}")
    return result


def _format_decimal(value: Decimal) -> str:
    return format(value, "f")


class BybitTestnet:
    def __init__(self, api_key: str, api_secret: str) -> None:
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        local_before = int(time.time() * 1000)
        server_time = self._request("GET", "/v5/market/time")["time"]
        local_after = int(time.time() * 1000)
        self.time_offset_ms = int(server_time) - ((local_before + local_after) // 2)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        authenticated: bool = False,
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode(params or {})
        body_text = json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""
        url = f"{BASE_URL}{path}"
        if query:
            url = f"{url}?{query}"
        headers = {"Content-Type": "application/json"}
        if authenticated:
            timestamp = str(int(time.time() * 1000) + self.time_offset_ms)
            payload = timestamp + self.api_key + RECV_WINDOW
            payload += query if method == "GET" else body_text
            signature = hmac.new(
                self.api_secret, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            headers.update({
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": RECV_WINDOW,
                "X-BAPI-SIGN": signature,
                "X-BAPI-SIGN-TYPE": "2",
            })
        request = urllib.request.Request(
            url,
            data=body_text.encode("utf-8") if method == "POST" else None,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                response_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise BybitTestnetError(f"Bybit HTTP {error.code} for {path}: {detail}") from error
        except urllib.error.URLError as error:
            raise BybitTestnetError(f"Bybit request failed for {path}: {error.reason}") from error
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise BybitTestnetError(f"Bybit returned invalid JSON for {path}") from error
        if not isinstance(response_json, dict):
            raise BybitTestnetError(f"Bybit returned a non-object response for {path}")
        if response_json.get("retCode") != 0:
            raise BybitTestnetError(
                f"Bybit rejected {path}: code={response_json.get('retCode')} "
                f"message={response_json.get('retMsg')}"
            )
        return response_json

    def get(self, path: str, params: dict[str, str], *, authenticated: bool = False) -> dict[str, Any]:
        return self._request("GET", path, params=params, authenticated=authenticated)

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, body=body, authenticated=True)


def _result(response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result")
    if not isinstance(result, dict):
        raise BybitTestnetError("Bybit response is missing its result object")
    return result


def _order_list(response: dict[str, Any]) -> list[dict[str, Any]]:
    order_list = _result(response).get("list")
    if not isinstance(order_list, list):
        raise BybitTestnetError("Bybit order query is missing result.list")
    return [order for order in order_list if isinstance(order, dict)]


def _wait_for_order(
    client: BybitTestnet,
    symbol: str,
    order_id: str,
    *,
    terminal: bool = False,
) -> dict[str, Any]:
    for _ in range(POLL_ATTEMPTS):
        orders = _order_list(client.get(
            "/v5/order/realtime",
            {"category": "linear", "symbol": symbol, "orderId": order_id},
            authenticated=True,
        ))
        if orders:
            order = orders[0]
            status = str(order.get("orderStatus", ""))
            if not terminal or status not in LIVE_ORDER_STATUSES:
                return order
        time.sleep(POLL_INTERVAL_SECONDS)
    expectation = "terminal" if terminal else "visible"
    raise BybitTestnetError(f"order {order_id} did not become {expectation} in Bybit query")


def _assert_no_existing_activity(client: BybitTestnet, symbol: str) -> None:
    active_orders = _order_list(client.get(
        "/v5/order/realtime",
        {"category": "linear", "symbol": symbol},
        authenticated=True,
    ))
    if active_orders:
        raise BybitTestnetError(
            f"testnet symbol {symbol} already has active orders; use a dedicated clean account"
        )
    positions = _result(client.get(
        "/v5/position/list",
        {"category": "linear", "symbol": symbol},
        authenticated=True,
    )).get("list", [])
    if any(_decimal(position.get("size", "0"), "position size") != 0 for position in positions):
        raise BybitTestnetError(
            f"testnet symbol {symbol} already has an open position; use a dedicated clean account"
        )


def _load_instrument(
    client: BybitTestnet, symbol: str
) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    instruments = _result(client.get(
        "/v5/market/instruments-info",
        {"category": "linear", "symbol": symbol},
    )).get("list", [])
    if not instruments:
        raise BybitTestnetError(f"Bybit has no linear instrument for {symbol}")
    instrument = instruments[0]
    if instrument.get("status") != "Trading":
        raise BybitTestnetError(f"{symbol} is not tradable (status={instrument.get('status')})")
    lot = instrument.get("lotSizeFilter", {})
    price = instrument.get("priceFilter", {})
    max_order_qty = _decimal(lot.get("maxOrderQty"), "maxOrderQty")
    max_market_qty = _decimal(lot.get("maxMktOrderQty"), "maxMktOrderQty")
    return (
        _decimal(lot.get("qtyStep"), "qtyStep"),
        _decimal(lot.get("minOrderQty"), "minOrderQty"),
        min(max_order_qty, max_market_qty),
        _decimal(lot.get("minNotionalValue", "0"), "minNotionalValue"),
        _decimal(price.get("tickSize"), "tickSize"),
    )


def _tick_price(price: Decimal, tick: Decimal, rounding: str) -> Decimal:
    units = (price / tick).to_integral_value(
        rounding=ROUND_FLOOR if rounding == "down" else ROUND_CEILING
    )
    return units * tick


def _order_body(
    *,
    symbol: str,
    side: str,
    qty: Decimal,
    order_type: str,
    position_idx: int,
    order_link_id: str,
    price: Decimal | None = None,
    trigger_price: Decimal | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": order_type,
        "qty": _format_decimal(qty),
        "positionIdx": position_idx,
        "orderLinkId": order_link_id,
    }
    if price is not None:
        body.update({"price": _format_decimal(price), "timeInForce": "GTC"})
    if trigger_price is not None:
        body.update({
            "triggerPrice": _format_decimal(trigger_price),
            "triggerDirection": 1 if side == "Buy" else 2,
            "triggerBy": "MarkPrice",
        })
    return body


def _create_and_verify(
    client: BybitTestnet,
    body: dict[str, Any],
    *,
    symbol: str,
    expected_side: str,
    expected_type: str,
    pending_orders: list[tuple[str, str]],
    market_orders: list[tuple[str, int, str]],
    position_idx: int,
) -> dict[str, Any]:
    result = _result(client.post("/v5/order/create", body))
    order_id = str(result.get("orderId", ""))
    if not order_id:
        raise BybitTestnetError(f"Bybit accepted {expected_type} {expected_side} without an orderId")
    if expected_type != "Market" or body.get("triggerPrice") is not None:
        pending_orders.append((symbol, order_id))
    elif expected_type == "Market":
        market_orders.append((expected_side, position_idx, order_id))
    order = _wait_for_order(
        client,
        symbol,
        order_id,
        terminal=expected_type == "Market" and body.get("triggerPrice") is None,
    )
    expected_api_type = "Limit" if expected_type == "Limit" else "Market"
    if order.get("side") != expected_side:
        raise BybitTestnetError(f"order {order_id} query returned the wrong side")
    if order.get("orderType") != expected_api_type:
        raise BybitTestnetError(f"order {order_id} query returned the wrong order type")
    if expected_type == "Market" and body.get("triggerPrice") is None and order.get("orderStatus") != "Filled":
        raise BybitTestnetError(f"market order {order_id} was {order.get('orderStatus')}")
    if expected_type != "Market" and order.get("orderStatus") not in LIVE_ORDER_STATUSES:
        raise BybitTestnetError(
            f"{expected_type} order {order_id} was not left active "
            f"(status={order.get('orderStatus')})"
        )
    if expected_type == "Stop" and _decimal(
        order.get("triggerPrice", "0"), "queried trigger price"
    ) != _decimal(body["triggerPrice"], "submitted trigger price"):
        raise BybitTestnetError(f"stop order {order_id} query returned the wrong trigger price")
    if expected_type == "Limit" and _decimal(
        order.get("price", "0"), "queried limit price"
    ) != _decimal(body["price"], "submitted limit price"):
        raise BybitTestnetError(f"limit order {order_id} query returned the wrong price")
    print(
        f"SUCCESS | {expected_type.upper()} | "
        f"{'LONG' if expected_side == 'Buy' else 'SHORT'} ({expected_side}) | "
        f"Bybit testnet accepted and order query verified | orderId={order_id}",
        flush=True,
    )
    return order


def _cancel_pending(
    client: BybitTestnet, symbol: str, order_id: str
) -> Decimal:
    order = _wait_for_order(client, symbol, order_id)
    if order.get("orderStatus") in LIVE_ORDER_STATUSES:
        client.post("/v5/order/cancel", {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        })
        order = _wait_for_order(client, symbol, order_id, terminal=True)
    return _decimal(order.get("cumExecQty", "0"), "cumulative executed quantity")


def _close_test_exposure(
    client: BybitTestnet,
    *,
    symbol: str,
    side: str,
    qty: Decimal,
    position_idx: int,
) -> None:
    if qty <= 0:
        return
    close_side = "Sell" if side == "Buy" else "Buy"
    body = _order_body(
        symbol=symbol,
        side=close_side,
        qty=qty,
        order_type="Market",
        position_idx=position_idx,
        order_link_id=f"uets-clean-{uuid.uuid4().hex[:20]}",
    )
    body["reduceOnly"] = True
    result = _result(client.post("/v5/order/create", body))
    close_id = str(result.get("orderId", ""))
    if not close_id:
        raise BybitTestnetError("Bybit did not return an id for test-position cleanup")
    close_order = _wait_for_order(client, symbol, close_id)
    if close_order.get("orderStatus") != "Filled":
        raise BybitTestnetError(
            f"test-position cleanup {close_id} did not fill completely "
            f"(status={close_order.get('orderStatus')})"
        )
    print(f"CLEANUP | reduce-only {close_side} {qty} | position closed", flush=True)


def _run_testnet_smoke() -> None:
    required = (
        "UETS_BYBIT_TESTNET_API_KEY",
        "UETS_BYBIT_TESTNET_API_SECRET",
        "UETS_BYBIT_TESTNET_SYMBOL",
        "UETS_BYBIT_TESTNET_QTY",
        "UETS_BYBIT_TESTNET_MAX_NOTIONAL_USDT",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise BybitTestnetError(
            "missing required environment variables: " + ", ".join(missing)
        )
    if os.environ.get("UETS_BYBIT_TESTNET_CONFIRM") != "YES":
        raise BybitTestnetError(
            "set UETS_BYBIT_TESTNET_CONFIRM=YES after confirming the API key is testnet-only"
        )

    mode = os.environ.get("UETS_BYBIT_TESTNET_POSITION_MODE", "one-way").lower()
    if mode not in {"one-way", "hedge"}:
        raise BybitTestnetError("UETS_BYBIT_TESTNET_POSITION_MODE must be one-way or hedge")
    api_key = os.environ["UETS_BYBIT_TESTNET_API_KEY"]
    api_secret = os.environ["UETS_BYBIT_TESTNET_API_SECRET"]
    symbol = os.environ["UETS_BYBIT_TESTNET_SYMBOL"].upper()
    qty = _decimal(os.environ["UETS_BYBIT_TESTNET_QTY"], "configured quantity")
    max_notional = _decimal(
        os.environ["UETS_BYBIT_TESTNET_MAX_NOTIONAL_USDT"], "configured max notional"
    )
    if qty <= 0 or max_notional <= 0:
        raise BybitTestnetError("configured quantity and max notional must be positive")

    client = BybitTestnet(api_key, api_secret)
    _assert_no_existing_activity(client, symbol)
    qty_step, min_qty, max_qty, min_notional, tick = _load_instrument(client, symbol)
    if qty < min_qty or qty > max_qty or qty % qty_step != 0:
        raise BybitTestnetError(
            f"quantity {qty} violates {symbol} testnet filters "
            f"(min={min_qty}, max={max_qty}, step={qty_step})"
        )
    ticker = _result(client.get(
        "/v5/market/tickers", {"category": "linear", "symbol": symbol}
    )).get("list", [])
    if not ticker:
        raise BybitTestnetError(f"Bybit returned no ticker for {symbol}")
    mark = _decimal(ticker[0].get("markPrice"), "mark price")
    notional = qty * mark
    if notional < min_notional:
        raise BybitTestnetError(
            f"quantity {qty} is below {symbol} min notional {min_notional}"
        )
    if notional > max_notional:
        raise BybitTestnetError(
            f"estimated notional {notional} exceeds your explicit testnet cap {max_notional}"
        )

    pending_orders: list[tuple[str, str]] = []
    market_orders: list[tuple[str, int, str]] = []
    exposures: list[tuple[str, int, Decimal]] = []
    try:
        for side in ("Buy", "Sell"):
            position_idx = (
                0 if mode == "one-way" else (1 if side == "Buy" else 2)
            )
            order = _create_and_verify(
                client,
                _order_body(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="Market",
                    position_idx=position_idx,
                    order_link_id=f"uets-mkt-{uuid.uuid4().hex[:20]}",
                ),
                symbol=symbol,
                expected_side=side,
                expected_type="Market",
                pending_orders=pending_orders,
                market_orders=market_orders,
                position_idx=position_idx,
            )
            market_order = (side, position_idx, str(order["orderId"]))
            executed = _decimal(order.get("cumExecQty", "0"), "market executed quantity")
            if executed <= 0:
                raise BybitTestnetError(f"market order {market_order[2]} filled zero quantity")
            _close_test_exposure(
                client,
                symbol=symbol,
                side=side,
                qty=executed,
                position_idx=position_idx,
            )
            market_orders.remove(market_order)
        for side in ("Buy", "Sell"):
            position_idx = (
                0 if mode == "one-way" else (1 if side == "Buy" else 2)
            )
            limit_price = _tick_price(
                mark * (Decimal("0.99") if side == "Buy" else Decimal("1.01")),
                tick,
                "down" if side == "Buy" else "up",
            )
            _create_and_verify(
                client,
                _order_body(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="Limit",
                    position_idx=position_idx,
                    order_link_id=f"uets-lmt-{uuid.uuid4().hex[:20]}",
                    price=limit_price,
                ),
                symbol=symbol,
                expected_side=side,
                expected_type="Limit",
                pending_orders=pending_orders,
                market_orders=market_orders,
                position_idx=position_idx,
            )

        for side in ("Buy", "Sell"):
            position_idx = (
                0 if mode == "one-way" else (1 if side == "Buy" else 2)
            )
            trigger_price = _tick_price(
                mark * (Decimal("1.02") if side == "Buy" else Decimal("0.98")),
                tick,
                "up" if side == "Buy" else "down",
            )
            _create_and_verify(
                client,
                _order_body(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="Market",
                    position_idx=position_idx,
                    order_link_id=f"uets-stp-{uuid.uuid4().hex[:20]}",
                    trigger_price=trigger_price,
                ),
                symbol=symbol,
                expected_side=side,
                expected_type="Stop",
                pending_orders=pending_orders,
                market_orders=market_orders,
                position_idx=position_idx,
            )
    finally:
        cleanup_errors: list[str] = []
        for market_side, position_idx, order_id in market_orders:
            try:
                order = _wait_for_order(client, symbol, order_id, terminal=True)
                executed = _decimal(order.get("cumExecQty", "0"), "market executed quantity")
                if executed > 0:
                    exposures.append((market_side, position_idx, executed))
            except Exception as error:
                cleanup_errors.append(
                    f"could not determine fills for market order {order_id}: {error}"
                )
        for pending_symbol, order_id in reversed(pending_orders):
            try:
                executed = _cancel_pending(client, pending_symbol, order_id)
                if executed > 0:
                    order = _wait_for_order(client, pending_symbol, order_id)
                    side = str(order.get("side", ""))
                    idx = int(order.get("positionIdx", 0))
                    exposures.append((side, idx, executed))
                print(f"CLEANUP | cancelled test order | orderId={order_id}", flush=True)
            except Exception as error:
                cleanup_errors.append(f"could not safely cancel order {order_id}: {error}")
        if mode == "one-way":
            net = sum(
                (qty if side == "Buy" else -qty for side, _, qty in exposures),
                Decimal("0"),
            )
            if net:
                try:
                    _close_test_exposure(
                        client,
                        symbol=symbol,
                        side="Buy" if net > 0 else "Sell",
                        qty=abs(net),
                        position_idx=0,
                    )
                except Exception as error:
                    cleanup_errors.append(f"could not safely flatten test position: {error}")
        else:
            for side in ("Buy", "Sell"):
                total = sum(
                    (qty for exposure_side, _, qty in exposures if exposure_side == side),
                    Decimal("0"),
                )
                if total:
                    try:
                        _close_test_exposure(
                            client,
                            symbol=symbol,
                            side=side,
                            qty=total,
                            position_idx=1 if side == "Buy" else 2,
                        )
                    except Exception as error:
                        cleanup_errors.append(f"could not safely flatten {side} hedge: {error}")
        if cleanup_errors:
            raise BybitTestnetError("; ".join(cleanup_errors))
        _assert_no_existing_activity(client, symbol)


class BybitTestnetOrderSmokeTests(unittest.TestCase):
    def test_market_limit_and_conditional_stop_buy_sell(self) -> None:
        _run_testnet_smoke()


if __name__ == "__main__":
    unittest.main(verbosity=2)
