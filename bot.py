import os, json, time, math
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

load_dotenv()

STATE_FILE = "state.json"

# =========================
# CONFIG
# =========================
@dataclass
class Config:
    clob_host: str
    chain_id: int
    private_key: str
    signature_type: int
    funder: Optional[str]

    dry_run: bool

    trade_usd: float
    max_open_positions: int
    max_trades_per_day: int

    tp_pct: float
    sl_pct: float

    daily_loss_limit_usd: float
    daily_profit_target_usd: float

    max_spread: float
    scan_interval_sec: int
    order_timeout_sec: int
    cooldown_after_trade_sec: int

    mom_window_sec: int
    mom_threshold: float

    order_min_shares: float

    yes_token_id: str
    no_token_id: str

    @staticmethod
    def from_env() -> "Config":
        def req(name: str) -> str:
            v = os.getenv(name, "").strip()
            if not v:
                raise RuntimeError(f"Missing env var: {name}")
            return v

        return Config(
            clob_host=os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com").strip(),
            chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
            private_key=req("POLYMARKET_PK"),
            signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")),
            funder=os.getenv("POLYMARKET_FUNDER", "").strip() or None,

            dry_run=os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes", "y"),

            trade_usd=float(os.getenv("TRADE_USD", "6")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "2")),
            max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "10")),

            tp_pct=float(os.getenv("TP_PCT", "0.06")),
            sl_pct=float(os.getenv("SL_PCT", "0.04")),

            daily_loss_limit_usd=float(os.getenv("DAILY_LOSS_LIMIT_USD", "15")),
            daily_profit_target_usd=float(os.getenv("DAILY_PROFIT_TARGET_USD", "20")),

            max_spread=float(os.getenv("MAX_SPREAD", "0.15")),
            scan_interval_sec=int(os.getenv("SCAN_INTERVAL_SEC", "10")),
            order_timeout_sec=int(os.getenv("ORDER_TIMEOUT_SEC", "45")),
            cooldown_after_trade_sec=int(os.getenv("COOLDOWN_AFTER_TRADE_SEC", "20")),

            mom_window_sec=int(os.getenv("MOM_WINDOW_SEC", "60")),
            mom_threshold=float(os.getenv("MOM_THRESHOLD", "0.008")),

            order_min_shares=float(os.getenv("ORDER_MIN_SHARES", "5")),

            yes_token_id=req("YES_TOKEN_ID"),
            no_token_id=req("NO_TOKEN_ID"),
        )


# =========================
# LOG / HELPERS
# =========================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def round_down(x: float, decimals: int) -> float:
    p = 10 ** decimals
    return math.floor(x * p) / p

def safe_price(p: float) -> float:
    return max(0.01, min(0.99, p))

def spread_pct(bid: Optional[float], ask: Optional[float]) -> float:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return 1.0
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid if mid > 0 else 1.0

# =========================
# STATE
# =========================
def reset_day_if_needed(state: Dict[str, Any]) -> None:
    today = str(date.today())
    if state.get("day") != today:
        state["day"] = today
        state["daily_pnl_usd"] = 0.0
        state["trades_today"] = 0

def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
                s.setdefault("day", str(date.today()))
                s.setdefault("daily_pnl_usd", 0.0)
                s.setdefault("trades_today", 0)
                s.setdefault("open_positions", [])
                s.setdefault("hist", {"YES": [], "NO": []})
                return s
        except Exception:
            pass
    return {
        "day": str(date.today()),
        "daily_pnl_usd": 0.0,
        "trades_today": 0,
        "open_positions": [],
        "hist": {"YES": [], "NO": []},  # [(ts, mid)]
    }

def save_state(state: Dict[str, Any]) -> None:
    # shrink hist
    hist = state.get("hist") or {}
    for k in ("YES", "NO"):
        if isinstance(hist.get(k), list) and len(hist[k]) > 400:
            hist[k] = hist[k][-400:]
    state["hist"] = hist

    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)

# =========================
# POLY CLIENT
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

def get_mid(client: ClobClient, token_id: str) -> Optional[float]:
    bid, ask = get_order_book_top(client, token_id)
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0

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
# RULES
# =========================
def can_trade(cfg: Config, state: Dict[str, Any]) -> Tuple[bool, str]:
    reset_day_if_needed(state)

    if int(state.get("trades_today", 0)) >= cfg.max_trades_per_day:
        return False, "max_trades_per_day"
    if len(state.get("open_positions", [])) >= cfg.max_open_positions:
        return False, "max_open_positions"

    pnl = float(state.get("daily_pnl_usd", 0.0))
    if pnl <= -cfg.daily_loss_limit_usd:
        return False, "daily_loss_limit_usd"
    if pnl >= cfg.daily_profit_target_usd:
        return False, "daily_profit_target_usd"

    return True, "ok"

def orderbook_ok(client: ClobClient, cfg: Config, token_id: str) -> Tuple[bool, Optional[float], Optional[float]]:
    bid, ask = get_order_book_top(client, token_id)
    if bid is None or ask is None:
        return False, None, None
    sp = spread_pct(bid, ask)
    return sp <= cfg.max_spread, bid, ask

# =========================
# SIGNAL (momentum on YES mid)
# =========================
def update_hist_and_signal(
    client: ClobClient,
    yes_id: str,
    no_id: str,
    hist: Dict[str, List[Tuple[float, float]]],
    window_sec: int,
    threshold: float,
) -> Optional[str]:
    ts = time.time()
    yes_mid = get_mid(client, yes_id)
    no_mid = get_mid(client, no_id)
    if yes_mid is None or no_mid is None:
        return None

    hist.setdefault("YES", []).append((ts, yes_mid))
    hist.setdefault("NO", []).append((ts, no_mid))

    keep = max(180, window_sec * 3)
    for k in ("YES", "NO"):
        hist[k] = [(t, p) for (t, p) in hist[k] if ts - t <= keep]

    past = [(t, p) for (t, p) in hist["YES"] if ts - t >= window_sec]
    if not past:
        return None

    _, p0 = past[0]
    mom = (yes_mid - p0) / p0

    if mom > threshold:
        return "LONG"   # buy YES
    if mom < -threshold:
        return "SHORT"  # buy NO
    return None

# =========================
# TRADING
# =========================
def open_position(client: ClobClient, cfg: Config, state: Dict[str, Any], token_id: str, label: str) -> bool:
    ok, reason = can_trade(cfg, state)
    if not ok:
        log(f"BLOCK OPEN: {reason}")
        return False

    ok_ob, bid, ask = orderbook_ok(client, cfg, token_id)
    if not ok_ob or ask is None:
        log("Orderbook not OK for entry.")
        return False

    price = safe_price(float(ask))
    stake = float(cfg.trade_usd)  # fixed $6
    shares = round_down(stake / price, 2)

    if shares < cfg.order_min_shares:
        log(f"Shares too small ({shares}) < ORDER_MIN_SHARES ({cfg.order_min_shares})")
        return False

    log(f"OPEN {label} price={price:.4f} shares={shares:.2f} stake=${stake:.2f} DRY_RUN={cfg.dry_run}")

    if cfg.dry_run:
        state["open_positions"].append({
            "token_id": token_id,
            "label": label,
            "entry_price": price,
            "shares": shares,
            "stake": stake,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "buy_order_id": "DRY_RUN",
        })
        state["trades_today"] = int(state.get("trades_today", 0)) + 1
        save_state(state)
        return True

    try:
        args = OrderArgs(token_id=str(token_id), price=price, size=shares, side=BUY)
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
            "shares": shares,
            "stake": stake,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "buy_order_id": oid,
        })
        state["trades_today"] = int(state.get("trades_today", 0)) + 1
        save_state(state)
        return True

    except Exception as e:
        log("BUY ERROR (FULL): " + repr(e))
        return False

def close_position(client: ClobClient, cfg: Config, state: Dict[str, Any], idx: int, reason: str) -> None:
    pos = state["open_positions"][idx]
    token_id = str(pos["token_id"])
    entry = float(pos["entry_price"])
    shares = float(pos["shares"])

    ok_ob, bid, ask = orderbook_ok(client, cfg, token_id)
    if not ok_ob or bid is None:
        return

    sell_price = safe_price(float(bid))
    pnl = (sell_price - entry) * shares

    log(f"CLOSE {reason} {pos['label']} entry={entry:.4f} sell={sell_price:.4f} shares={shares:.2f} pnl≈{pnl:.2f} DRY_RUN={cfg.dry_run}")

    if cfg.dry_run:
        state["daily_pnl_usd"] = float(state.get("daily_pnl_usd", 0.0)) + pnl
        state["open_positions"].pop(idx)
        save_state(state)
        return

    try:
        args = OrderArgs(token_id=token_id, price=sell_price, size=round_down(shares, 2), side=SELL)
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

        state["daily_pnl_usd"] = float(state.get("daily_pnl_usd", 0.0)) + pnl
        state["open_positions"].pop(idx)
        save_state(state)

    except Exception as e:
        log("SELL ERROR (FULL): " + repr(e))

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
# MAIN
# =========================
def main():
    cfg = Config.from_env()
    state = load_state()
    client = make_client(cfg)

    log(f"ONLINE DRY_RUN={cfg.dry_run} trade=${cfg.trade_usd:.2f} maxOpen={cfg.max_open_positions} maxTrades={cfg.max_trades_per_day}")
    log(f"Tokens: YES={cfg.yes_token_id[:10]}... NO={cfg.no_token_id[:10]}...")
    log(f"Signal: window={cfg.mom_window_sec}s threshold={cfg.mom_threshold} | maxSpread={cfg.max_spread}")

    while True:
        try:
            reset_day_if_needed(state)

            # manage exits
            if state.get("open_positions"):
                manage_tp_sl(client, cfg, state)

            ok, reason = can_trade(cfg, state)
            if not ok:
                log(f"PAUSED: {reason} pnl≈{state.get('daily_pnl_usd', 0.0):.2f} trades={state.get('trades_today', 0)} open={len(state.get('open_positions', []))}")
                time.sleep(cfg.scan_interval_sec)
                continue

            # validate orderbooks exist
            ok_yes, _, _ = orderbook_ok(client, cfg, cfg.yes_token_id)
            ok_no,  _, _ = orderbook_ok(client, cfg, cfg.no_token_id)
            if not ok_yes or not ok_no:
                log("Orderbook/spread not OK. retry...")
                time.sleep(cfg.scan_interval_sec)
                continue

            sig = update_hist_and_signal(
                client,
                yes_id=cfg.yes_token_id,
                no_id=cfg.no_token_id,
                hist=state.setdefault("hist", {"YES": [], "NO": []}),
                window_sec=cfg.mom_window_sec,
                threshold=cfg.mom_threshold,
            )
            save_state(state)

            if sig is None:
                time.sleep(cfg.scan_interval_sec)
                continue

            if sig == "LONG":
                opened = open_position(client, cfg, state, cfg.yes_token_id, "YES(UP)")
            else:
                opened = open_position(client, cfg, state, cfg.no_token_id, "NO(DOWN)")

            time.sleep(cfg.cooldown_after_trade_sec if opened else cfg.scan_interval_sec)

        except Exception as e:
            log("ERROR (FULL): " + repr(e))
            time.sleep(5)

if __name__ == "__main__":
    main()
