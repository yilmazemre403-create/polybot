import os, json, time, math
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Optional, Dict, Any, List, Tuple

import requests
from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

load_dotenv()

# =========================
# CONFIG
# =========================

SERIES_SLUG = "btc-up-or-down-5m"  # <- otomatik güncel market seçilecek

@dataclass
class Config:
    clob_host: str
    gamma_host: str
    chain_id: int

    private_key: str
    funder: Optional[str]
    signature_type: int

    start_balance: float

    risk_per_trade: float
    daily_loss_limit: float
    daily_profit_target: float
    max_trades_per_day: int
    max_open_positions: int

    tp_pct: float
    sl_pct: float

    min_liquidity_usd: float
    max_spread: float

    scan_interval_sec: int
    order_timeout_sec: int
    cooldown_after_trade_sec: int

    dry_run: bool

    @staticmethod
    def from_env() -> "Config":
        def req(name: str) -> str:
            v = os.getenv(name, "").strip()
            if not v:
                raise RuntimeError(f"Missing env var: {name}")
            return v

        return Config(
            clob_host=os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com").strip(),
            gamma_host=os.getenv("GAMMA_HOST", "https://gamma-api.polymarket.com").strip(),
            chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),

            private_key=req("POLYMARKET_PK"),
            funder=os.getenv("POLYMARKET_FUNDER", "").strip() or None,
            signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")),

            start_balance=float(os.getenv("START_BALANCE", "107")),

            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.08")),
            daily_loss_limit=float(os.getenv("DAILY_LOSS_LIMIT", "0.25")),
            daily_profit_target=float(os.getenv("DAILY_PROFIT_TARGET", "0.35")),
            max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "10")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "2")),

            tp_pct=float(os.getenv("TP_PCT", "0.06")),
            sl_pct=float(os.getenv("SL_PCT", "0.04")),

            min_liquidity_usd=float(os.getenv("MIN_LIQUIDITY_USD", "3000")),
            max_spread=float(os.getenv("MAX_SPREAD", "0.05")),

            scan_interval_sec=int(os.getenv("SCAN_INTERVAL_SEC", "20")),
            order_timeout_sec=int(os.getenv("ORDER_TIMEOUT_SEC", "45")),
            cooldown_after_trade_sec=int(os.getenv("COOLDOWN_AFTER_TRADE_SEC", "20")),

            dry_run=os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes", "y"),
        )

# =========================
# STATE
# =========================

STATE_FILE = "state.json"

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def load_state(cfg: Config) -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "day": str(date.today()),
        "balance": cfg.start_balance,
        "daily_pnl": 0.0,
        "trades_today": 0,
        "open_positions": [],  # max 2
        "last_action": now_iso(),
    }

def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)

def reset_day_if_needed(state: Dict[str, Any]) -> None:
    today = str(date.today())
    if state.get("day") != today:
        state["day"] = today
        state["daily_pnl"] = 0.0
        state["trades_today"] = 0

# =========================
# POLY HELPERS
# =========================

def make_client(cfg: Config) -> ClobClient:
    client = ClobClient(
        cfg.clob_host,
        key=cfg.private_key,
        chain_id=cfg.chain_id,
        signature_type=cfg.signature_type,
        funder=cfg.funder,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    return client

def get_order_book_top(client: ClobClient, token_id: str) -> Tuple[Optional[float], Optional[float]]:
    ob = client.get_order_book(token_id)
    bids = ob.bids or []
    asks = ob.asks or []
    best_bid = float(bids[0].price) if bids else None
    best_ask = float(asks[0].price) if asks else None
    return best_bid, best_ask

def spread_pct(bid: Optional[float], ask: Optional[float]) -> float:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return 1.0
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid if mid > 0 else 1.0

def round_down(x: float, decimals: int) -> float:
    p = 10 ** decimals
    return math.floor(x * p) / p

def safe_price(p: float) -> float:
    return max(0.01, min(0.99, p))

def post_gtc_order(client: ClobClient, args: OrderArgs) -> Dict[str, Any]:
    signed = client.create_order(args)
    return client.post_order(signed, OrderType.GTC)

def wait_for_order_fill(client: ClobClient, order_id: str, timeout_sec: int) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    last = None
    while time.time() < deadline:
        o = client.get_order(order_id)
        last = o
        status = str(o.get("status", "")).lower()
        if status in ("filled", "canceled", "cancelled", "expired", "rejected"):
            return o
        time.sleep(1.2)
    return last or {"status": "unknown"}

def cancel_order_safe(client: ClobClient, order_id: str) -> None:
    try:
        client.cancel(order_id)
    except Exception:
        pass

# =========================
# GAMMA: FIND LATEST ACTIVE MARKET IN SERIES
# =========================

def gamma_get_latest_series_market(cfg: Config, series_slug: str) -> Optional[Dict[str, Any]]:
    # Pull recent markets, filter by seriesSlug
    url = f"{cfg.gamma_host}/markets"
    params = {
        "limit": "200",
        "offset": "0",
        "order": "updatedAt",
        "ascending": "false",
        "closed": "false",
        "active": "true",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    markets = r.json() or []
    now_utc = datetime.now(timezone.utc)

    best = None
    for m in markets:
        if str(m.get("seriesSlug", "")) != series_slug:
            continue
        if not m.get("enableOrderBook", False):
            continue
        if m.get("closed", False) or (not m.get("active", True)):
            continue
        if not m.get("acceptingOrders", False):
            continue
        # endDate check
        end_str = m.get("endDate")
        try:
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        except Exception:
            continue
        if end_dt <= now_utc:
            continue
        liq = float(m.get("liquidityNum") or m.get("liquidity") or 0.0)
        if liq < cfg.min_liquidity_usd:
            continue
        # token ids
        ct = m.get("clobTokenIds")
        if isinstance(ct, str):
            try:
                ct = json.loads(ct)
            except Exception:
                ct = None
        if not isinstance(ct, list) or len(ct) < 2:
            continue

        # pick first valid
        best = {
            "slug": m.get("slug"),
            "title": m.get("question") or m.get("title") or "",
            "liquidity": liq,
            "endDate": end_str,
            "yes_token_id": str(ct[0]),
            "no_token_id": str(ct[1]),
        }
        break

    return best

# =========================
# SIGNAL: BINANCE MOMENTUM + RSI (NO KEY)
# =========================

def binance_klines(symbol="BTCUSDT", interval="1m", limit=200) -> List[Dict[str, Any]]:
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    raw = r.json()
    out = []
    for k in raw:
        out.append({"open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4])})
    return out

def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i-1]
        if diff >= 0:
            gains += diff
        else:
            losses += -diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))

def decide_signal_5m() -> Optional[str]:
    # Returns "LONG" (Up/YES) or "SHORT" (Down/NO) or None
    ks = binance_klines(limit=120)  # last 120 minutes
    closes = [k["close"] for k in ks]
    r = rsi(closes, 14)
    if r is None:
        return None

    # 5m momentum (close now vs 5 mins ago)
    if len(closes) < 6:
        return None
    mom = (closes[-1] - closes[-6]) / closes[-6]  # ~5m change

    # professional-ish, simple rules:
    # - follow momentum when not overbought/oversold extreme
    # - fade extremes slightly
    if mom > 0.0007 and r < 72:
        return "LONG"
    if mom < -0.0007 and r > 28:
        return "SHORT"
    # fade extreme RSI (mean reversion)
    if r >= 78:
        return "SHORT"
    if r <= 22:
        return "LONG"
    return None

# =========================
# RISK + TRADING
# =========================

def can_trade_today(cfg: Config, state: Dict[str, Any]) -> Tuple[bool, str]:
    reset_day_if_needed(state)
    bal = float(state["balance"])
    pnl = float(state["daily_pnl"])
    if int(state.get("trades_today", 0)) >= cfg.max_trades_per_day:
        return False, "max_trades_per_day"
    if pnl <= -bal * cfg.daily_loss_limit:
        return False, "daily_loss_limit"
    if pnl >= bal * cfg.daily_profit_target:
        return False, "daily_profit_target"
    if len(state.get("open_positions", [])) >= cfg.max_open_positions:
        return False, "max_open_positions"
    return True, "ok"

def open_position(client: ClobClient, cfg: Config, state: Dict[str, Any], token_id: str, label: str) -> bool:
    ok, reason = can_trade_today(cfg, state)
    if not ok:
        log(f"Blocked open: {reason}")
        return False

    bid, ask = get_order_book_top(client, token_id)
    if bid is None or ask is None:
        return False
    sp = spread_pct(bid, ask)
    if sp > cfg.max_spread:
        log(f"Spread too high: {sp*100:.2f}%")
        return False

    price = safe_price(float(ask))  # marketable BUY
    bal = float(state["balance"])
    stake = round_down(bal * cfg.risk_per_trade, 2)
    size = round_down(stake / price, 2)
    if size <= 0:
        return False

    log(f"OPEN {label} token={token_id[:10]}.. price={price:.4f} size={size:.2f} stake={stake:.2f} DRY_RUN={cfg.dry_run}")

    if cfg.dry_run:
        state["open_positions"].append({
            "token_id": token_id,
            "label": label,
            "entry_price": price,
            "size": size,
            "stake": stake,
            "opened_at": now_iso(),
            "buy_order_id": "DRY_RUN",
        })
        state["trades_today"] = int(state.get("trades_today", 0)) + 1
        save_state(state)
        return True

    try:
        args = OrderArgs(token_id=str(token_id), price=price, size=size, side=BUY)
        resp = post_gtc_order(client, args)
        oid = resp.get("orderID") or resp.get("id") or resp.get("order_id")
        if not oid:
            log(f"BUY missing order id: {resp}")
            return False
        filled = wait_for_order_fill(client, oid, cfg.order_timeout_sec)
        if str(filled.get("status", "")).lower() != "filled":
            cancel_order_safe(client, oid)
            log("BUY not filled -> canceled")
            return False

        state["open_positions"].append({
            "token_id": token_id,
            "label": label,
            "entry_price": price,
            "size": size,
            "stake": stake,
            "opened_at": now_iso(),
            "buy_order_id": oid,
        })
        state["trades_today"] = int(state.get("trades_today", 0)) + 1
        save_state(state)
        return True
    except Exception as e:
        log(f"BUY ERROR: {e}")
        return False

def close_position(client: ClobClient, cfg: Config, state: Dict[str, Any], idx: int, reason: str) -> None:
    pos = state["open_positions"][idx]
    token_id = str(pos["token_id"])
    entry = float(pos["entry_price"])
    size = float(pos["size"])

    bid, ask = get_order_book_top(client, token_id)
    if bid is None or ask is None:
        return
    sell_price = round_down(safe_price(float(bid)), 4)

    log(f"CLOSE {reason} {pos['label']} entry={entry:.4f} bid={bid:.4f} sell={sell_price:.4f} size={size:.2f} DRY_RUN={cfg.dry_run}")

    if cfg.dry_run:
        pnl = (sell_price - entry) * size
        state["daily_pnl"] = float(state["daily_pnl"]) + pnl
        state["balance"] = float(state["balance"]) + pnl
        state["open_positions"].pop(idx)
        save_state(state)
        return

    try:
        args = OrderArgs(token_id=token_id, price=sell_price, size=round_down(size, 2), side=SELL)
        resp = post_gtc_order(client, args)
        oid = resp.get("orderID") or resp.get("id") or resp.get("order_id")
        if not oid:
            log(f"SELL missing order id: {resp}")
            return
        filled = wait_for_order_fill(client, oid, cfg.order_timeout_sec)
        if str(filled.get("status", "")).lower() != "filled":
            cancel_order_safe(client, oid)
            log("SELL not filled -> canceled")
            return

        pnl = (sell_price - entry) * size
        state["daily_pnl"] = float(state["daily_pnl"]) + pnl
        state["balance"] = float(state["balance"]) + pnl
        state["open_positions"].pop(idx)
        save_state(state)
    except Exception as e:
        log(f"SELL ERROR: {e}")

def manage_tp_sl(client: ClobClient, cfg: Config, state: Dict[str, Any]) -> None:
    for i in range(len(state.get("open_positions", [])) - 1, -1, -1):
        pos = state["open_positions"][i]
        token_id = str(pos["token_id"])
        entry = float(pos["entry_price"])
        bid, ask = get_order_book_top(client, token_id)
        if bid is None or ask is None:
            continue
        mid = (bid + ask) / 2.0
        tp = entry * (1.0 + cfg.tp_pct)
        sl = entry * (1.0 - cfg.sl_pct)
        if mid >= tp:
            close_position(client, cfg, state, i, "TP")
            time.sleep(cfg.cooldown_after_trade_sec)
        elif mid <= sl:
            close_position(client, cfg, state, i, "SL")
            time.sleep(cfg.cooldown_after_trade_sec)

# =========================
# MAIN LOOP
# =========================

def main():
    cfg = Config.from_env()
    state = load_state(cfg)
    save_state(state)

    client = make_client(cfg)

    log(f"ONLINE DRY_RUN={cfg.dry_run} bal={state['balance']:.2f} risk={cfg.risk_per_trade*100:.1f}% maxOpen={cfg.max_open_positions}")

    while True:
        try:
            reset_day_if_needed(state)

            # 1) manage exits
            if state.get("open_positions"):
                manage_tp_sl(client, cfg, state)

            # 2) if can open new, decide signal
            ok, reason = can_trade_today(cfg, state)
            if not ok:
                log(f"PAUSED: {reason} dailyPnL={state['daily_pnl']:.2f} trades={state['trades_today']}")
                time.sleep(cfg.scan_interval_sec)
                continue

            sig = decide_signal_5m()
            if sig is None:
                time.sleep(cfg.scan_interval_sec)
                continue

            m = gamma_get_latest_series_market(cfg, SERIES_SLUG)
            if not m:
                log("No active series market found")
                time.sleep(cfg.scan_interval_sec)
                continue

            yes_id = m["yes_token_id"]
            no_id = m["no_token_id"]

            # LONG -> YES, SHORT -> NO
            if sig == "LONG":
                open_position(client, cfg, state, yes_id, "YES(UP)")
            else:
                open_position(client, cfg, state, no_id, "NO(DOWN)")

            time.sleep(2)

        except Exception as e:
            log(f"ERROR: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
