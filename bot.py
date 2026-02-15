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

SERIES_SLUG = "btc-up-or-down-5m"
STATE_FILE = "state.json"


# =========================
# CONFIG
# =========================
@dataclass
class Config:
    clob_host: str
    gamma_host: str
    chain_id: int

    private_key: str
    funder: Optional[str]
    signature_type: int

    # fixed stake per trade (USD)
    trade_usd: float

    # limits
    max_trades_per_day: int
    max_open_positions: int

    # TP/SL on token price
    tp_pct: float
    sl_pct: float

    # daily hard limits (USD)
    daily_loss_limit_usd: float
    daily_profit_target_usd: float

    # execution filters
    max_spread: float
    scan_interval_sec: int
    order_timeout_sec: int
    cooldown_after_trade_sec: int
    end_buffer_sec: int

    # signal
    mom_window_sec: int
    mom_threshold: float

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

            trade_usd=float(os.getenv("TRADE_USD", "6")),

            max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "10")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "2")),

            tp_pct=float(os.getenv("TP_PCT", "0.06")),
            sl_pct=float(os.getenv("SL_PCT", "0.04")),

            daily_loss_limit_usd=float(os.getenv("DAILY_LOSS_LIMIT_USD", "15")),
            daily_profit_target_usd=float(os.getenv("DAILY_PROFIT_TARGET_USD", "20")),

            max_spread=float(os.getenv("MAX_SPREAD", "0.05")),
            scan_interval_sec=int(os.getenv("SCAN_INTERVAL_SEC", "10")),
            order_timeout_sec=int(os.getenv("ORDER_TIMEOUT_SEC", "45")),
            cooldown_after_trade_sec=int(os.getenv("COOLDOWN_AFTER_TRADE_SEC", "20")),
            end_buffer_sec=int(os.getenv("END_BUFFER_SEC", "45")),

            mom_window_sec=int(os.getenv("MOM_WINDOW_SEC", "60")),
            mom_threshold=float(os.getenv("MOM_THRESHOLD", "0.008")),

            dry_run=os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes", "y"),
        )


# =========================
# UTIL / STATE
# =========================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def round_down(x: float, decimals: int) -> float:
    p = 10 ** decimals
    return math.floor(x * p) / p

def safe_price(p: float) -> float:
    return max(0.01, min(0.99, p))

def _parse_iso_z(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

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
                s.setdefault("current_market", None)
                return s
        except Exception:
            pass

    return {
        "day": str(date.today()),
        "daily_pnl_usd": 0.0,
        "trades_today": 0,
        "open_positions": [],
        "hist": {"YES": [], "NO": []},  # [(ts, mid)]
        "current_market": None,
    }

def save_state(state: Dict[str, Any]) -> None:
    # shrink hist
    try:
        hist = state.get("hist") or {}
        for k in ("YES", "NO"):
            if k in hist and isinstance(hist[k], list) and len(hist[k]) > 300:
                hist[k] = hist[k][-300:]
        state["hist"] = hist
    except Exception:
        pass

    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)


# =========================
# POLY CLIENT / ORDERS
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
# GAMMA MARKET PICK (FIXED)
# =========================
def pick_series_slug(m: Dict[str, Any]) -> Optional[str]:
    # sometimes seriesSlug is not on root
    if m.get("seriesSlug"):
        return str(m.get("seriesSlug"))
    ev = m.get("events")
    if isinstance(ev, list) and ev:
        s = ev[0].get("seriesSlug")
        if s:
            return str(s)
    se = m.get("series")
    if isinstance(se, list) and se:
        s = se[0].get("slug")
        if s:
            return str(s)
    return None

def parse_token_ids(m: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    ct = m.get("clobTokenIds")
    if isinstance(ct, str):
        try:
            ct = json.loads(ct)
        except Exception:
            ct = None
    if isinstance(ct, list) and len(ct) >= 2:
        return str(ct[0]), str(ct[1])

    ct2 = m.get("outcomeTokenIds")
    if isinstance(ct2, str):
        try:
            ct2 = json.loads(ct2)
        except Exception:
            ct2 = None
    if isinstance(ct2, list) and len(ct2) >= 2:
        return str(ct2[0]), str(ct2[1])

    return None

def gamma_find_best_market(cfg: Config, series_slug: str, end_buffer_sec: int) -> Optional[Dict[str, Any]]:
    url = f"{cfg.gamma_host}/markets"
    params = {"limit": "200", "offset": "0", "order": "updatedAt", "ascending": "false"}

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    markets = r.json() or []

    now_utc = datetime.now(timezone.utc)

    for m in markets:
        sslug = pick_series_slug(m)
        if sslug != series_slug:
            continue

        if not m.get("enableOrderBook", False):
            continue
        if m.get("closed", False) is True:
            continue

        end_str = m.get("endDate")
        if not end_str:
            ev = m.get("events")
            if isinstance(ev, list) and ev and ev[0].get("endDate"):
                end_str = ev[0].get("endDate")
        if not end_str:
            continue

        end_dt = _parse_iso_z(end_str)
        if not end_dt:
            continue

        if (end_dt - now_utc).total_seconds() <= end_buffer_sec:
            continue

        toks = parse_token_ids(m)
        if not toks:
            continue
        yes_id, no_id = toks

        slug = m.get("slug") or ""
        if not slug:
            ev = m.get("events")
            if isinstance(ev, list) and ev and ev[0].get("slug"):
                slug = ev[0].get("slug")

        title = m.get("question") or m.get("title") or ""
        return {
            "slug": slug,
            "title": title,
            "endDate": end_str,
            "yes_token_id": yes_id,
            "no_token_id": no_id,
        }

    return None


# =========================
# SIGNAL: Polymarket mid-price momentum
# =========================
def get_mid(client: ClobClient, token_id: str) -> Optional[float]:
    bid, ask = get_order_book_top(client, token_id)
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0

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
    no_mid  = get_mid(client, no_id)
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
# LIMITS / TRADING
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

def orderbook_ok(client: ClobClient, cfg: Config, token_id: str) -> bool:
    bid, ask = get_order_book_top(client, token_id)
    if bid is None or ask is None:
        return False
    sp = spread_pct(bid, ask)
    return sp <= cfg.max_spread

def open_position(client: ClobClient, cfg: Config, state: Dict[str, Any], token_id: str, label: str) -> bool:
    ok, reason = can_trade(cfg, state)
    if not ok:
        log(f"BLOCK OPEN: {reason}")
        return False

    bid, ask = get_order_book_top(client, token_id)
    if bid is None or ask is None:
        return False

    sp = spread_pct(bid, ask)
    if sp > cfg.max_spread:
        log(f"Spread too high: {sp*100:.2f}%")
        return False

    price = safe_price(float(ask))  # marketable buy
    stake = round_down(cfg.trade_usd, 2)  # <-- FIXED $6
    size = round_down(stake / price, 2)
    if size <= 0:
        return False

    log(f"OPEN {label} price={price:.4f} size={size:.2f} stake={stake:.2f} DRY_RUN={cfg.dry_run}")

    if cfg.dry_run:
        state["open_positions"].append({
            "token_id": token_id,
            "label": label,
            "entry_price": price,
            "size": size,
            "stake": stake,
            "opened_at": datetime.now(timezone.utc).isoformat(),
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
            "opened_at": datetime.now(timezone.utc).isoformat(),
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
    log(f"CLOSE {reason} {pos['label']} entry={entry:.4f} sell={sell_price:.4f} size={size:.2f} DRY_RUN={cfg.dry_run}")

    pnl = (sell_price - entry) * size

    if cfg.dry_run:
        state["daily_pnl_usd"] = float(state.get("daily_pnl_usd", 0.0)) + pnl
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

        state["daily_pnl_usd"] = float(state.get("daily_pnl_usd", 0.0)) + pnl
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
# MAIN
# =========================
def main():
    cfg = Config.from_env()
    state = load_state()

    client = make_client(cfg)

    log(f"ONLINE DRY_RUN={cfg.dry_run} tradeUSD={cfg.trade_usd:.2f} maxOpen={cfg.max_open_positions} maxTrades={cfg.max_trades_per_day}")
    log(f"Signal: window={cfg.mom_window_sec}s threshold={cfg.mom_threshold:.4f} endBuffer={cfg.end_buffer_sec}s maxSpread={cfg.max_spread:.2f}")

    last_market_refresh = 0.0
    market = None

    while True:
        try:
            reset_day_if_needed(state)

            # exits first
            if state.get("open_positions"):
                manage_tp_sl(client, cfg, state)

            ok, reason = can_trade(cfg, state)
            if not ok:
                log(f"PAUSED: {reason} pnl={state.get('daily_pnl_usd', 0.0):.2f} trades={state.get('trades_today', 0)} open={len(state.get('open_positions', []))}")
                time.sleep(cfg.scan_interval_sec)
                continue

            # refresh market (these rotate fast)
            now_ts = time.time()
            if (now_ts - last_market_refresh) > 5 or market is None:
                market = gamma_find_best_market(cfg, SERIES_SLUG, cfg.end_buffer_sec)
                last_market_refresh = now_ts

                if not market:
                    log("No active series market found (gamma). retry...")
                    time.sleep(cfg.scan_interval_sec)
                    continue

                yes_id = market["yes_token_id"]
                no_id = market["no_token_id"]

                # ensure orderbook is tradable
                if not orderbook_ok(client, cfg, yes_id) or not orderbook_ok(client, cfg, no_id):
                    log("Market found but orderbook/spread not OK. retry...")
                    time.sleep(cfg.scan_interval_sec)
                    continue

                state["current_market"] = market
                save_state(state)
                log(f"Market: {market['slug']} | ends {market['endDate']}")

            cm = state.get("current_market") or market
            if not cm:
                time.sleep(cfg.scan_interval_sec)
                continue

            yes_id = cm["yes_token_id"]
            no_id = cm["no_token_id"]

            sig = update_hist_and_signal(
                client,
                yes_id=yes_id,
                no_id=no_id,
                hist=state.setdefault("hist", {"YES": [], "NO": []}),
                window_sec=cfg.mom_window_sec,
                threshold=cfg.mom_threshold,
            )
            save_state(state)

            if sig is None:
                time.sleep(cfg.scan_interval_sec)
                continue

            # execute
            if sig == "LONG":
                opened = open_position(client, cfg, state, yes_id, "YES(UP)")
            else:
                opened = open_position(client, cfg, state, no_id, "NO(DOWN)")

            if opened:
                time.sleep(cfg.cooldown_after_trade_sec)
            else:
                time.sleep(cfg.scan_interval_sec)

        except Exception as e:
            log(f"ERROR: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
