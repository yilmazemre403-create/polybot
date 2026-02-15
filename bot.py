import os, json, time, math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

import requests
from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------
@dataclass
class Config:
    private_key: str
    chain_id: int
    gamma_host: str
    clob_host: str
    series_id: int

    dry_run: bool

    # Risk controls
    trade_usd: float
    max_open_positions: int
    max_trades_per_day: int
    daily_max_loss_usd: float  # NEW: daily hard stop loss

    tp_pct: float
    sl_pct: float

    max_spread: float          # spread ratio threshold (e.g., 0.03 = 3%)
    min_top_size: int          # NEW: require some liquidity at top of book

    scan_interval_sec: int
    end_buffer_sec: int
    cooldown_after_trade_sec: int

    mom_window_sec: int
    mom_threshold: float

    @staticmethod
    def from_env() -> "Config":
        def req(name: str) -> str:
            v = os.getenv(name, "").strip()
            if not v:
                raise RuntimeError(f"Missing env var: {name}")
            return v

        pk = req("POLYMARKET_PK")
        if not pk.startswith("0x"):
            raise RuntimeError("POLYMARKET_PK must start with 0x")
        hexpart = pk[2:]
        if any(c not in "0123456789abcdefABCDEF" for c in hexpart):
            raise RuntimeError("POLYMARKET_PK is not valid hex")

        # Daha düşük risk için güvenli defaultlar:
        # - trade_usd 6 -> 3
        # - max_open_positions 2 -> 1
        # - max_trades_per_day 10 -> 6
        # - max_spread 0.15 -> 0.05 (5%)  (istersen 0.08-0.12 arası deneyebilirsin)
        # - günlük max loss: 6$ (istersen 3-10 arası)
        return Config(
            private_key=pk,
            chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
            gamma_host=os.getenv("GAMMA_HOST", "https://gamma-api.polymarket.com").strip(),
            clob_host=os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com").strip(),
            series_id=int(os.getenv("SERIES_ID", "10684")),

            dry_run=os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes", "y"),

            trade_usd=float(os.getenv("TRADE_USD", "3")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "1")),
            max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "6")),
            daily_max_loss_usd=float(os.getenv("DAILY_MAX_LOSS_USD", "6")),

            tp_pct=float(os.getenv("TP_PCT", "0.03")),
            sl_pct=float(os.getenv("SL_PCT", "0.02")),

            max_spread=float(os.getenv("MAX_SPREAD", "0.05")),
            min_top_size=int(os.getenv("MIN_TOP_SIZE", "20")),

            scan_interval_sec=int(os.getenv("SCAN_INTERVAL_SEC", "3")),
            end_buffer_sec=int(os.getenv("END_BUFFER_SEC", "10")),
            cooldown_after_trade_sec=int(os.getenv("COOLDOWN_AFTER_TRADE_SEC", "25")),

            mom_window_sec=int(os.getenv("MOM_WINDOW_SEC", "60")),
            mom_threshold=float(os.getenv("MOM_THRESHOLD", "0.008")),
        )

# ----------------------------
# LOG
# ----------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ----------------------------
# STATE
# ----------------------------
class RollingMid:
    def __init__(self):
        self.buf: List[Tuple[float, float]] = []  # (ts, mid)

    def add(self, ts: float, mid: float):
        self.buf.append((ts, mid))

    def prune(self, now_ts: float, window: int):
        cut = now_ts - window
        while self.buf and self.buf[0][0] < cut:
            self.buf.pop(0)

    def mom(self) -> Optional[float]:
        if len(self.buf) < 2:
            return None
        first = self.buf[0][1]
        last = self.buf[-1][1]
        if first <= 0:
            return None
        return (last - first) / first

# ----------------------------
# POLY HELPERS
# ----------------------------
def make_client(cfg: Config) -> ClobClient:
    client = ClobClient(cfg.clob_host, key=cfg.private_key, chain_id=cfg.chain_id)
    client.set_api_creds(client.create_or_derive_api_creds())
    return client

def gamma_get_live_event(cfg: Config) -> Optional[Dict[str, Any]]:
    url = f"{cfg.gamma_host}/events"
    params = {
        "series_id": cfg.series_id,
        "active": "true",
        "closed": "false",
        "order": "startTime",
        "ascending": "true",
        "limit": "25",
        "offset": "0",
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    events = data if isinstance(data, list) else []
    if not events:
        return None

    now = datetime.now(timezone.utc).timestamp()
    for ev in events:
        end = ev.get("endDate") or ev.get("endTime")
        start = ev.get("startTime") or ev.get("startDate")
        if not end or not start:
            continue
        try:
            end_ts = datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp()
            start_ts = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
        except Exception:
            continue

        if end_ts <= now + cfg.end_buffer_sec:
            continue

        markets = ev.get("markets") or []
        if not markets:
            continue
        m0 = markets[0]
        if m0.get("acceptingOrders") is False:
            continue
        return ev

    return None

def parse_token_ids(market_obj: Dict[str, Any]) -> Tuple[str, str]:
    ids = market_obj.get("clobTokenIds")
    if isinstance(ids, str):
        ids = json.loads(ids)
    if not isinstance(ids, list) or len(ids) < 2:
        raise RuntimeError("Market missing clobTokenIds")
    return str(ids[0]), str(ids[1])

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def clamp_price(p: float) -> float:
    # Polymarket price bounds
    p = max(0.01, min(0.99, p))
    # tick gibi davran: 0.001 yuvarla (daha stabil)
    return round(p, 3)

def shares_for_usd(usd: float, price: float) -> int:
    if price <= 0:
        return 0
    return int(math.floor(usd / price))

def get_best_bid_ask_size(client: ClobClient, token_id: str) -> Tuple[Optional[float], Optional[float], int, int]:
    """
    Return (bid, ask, bid_size, ask_size).
    If empty/invalid, return None.
    """
    ob = client.get_order_book(token_id)
    bids = ob.bids or []
    asks = ob.asks or []

    if not bids or not asks:
        return None, None, 0, 0

    bid = safe_float(bids[0].price)
    ask = safe_float(asks[0].price)
    bsz = int(safe_float(getattr(bids[0], "size", 0)) or 0)
    asz = int(safe_float(getattr(asks[0], "size", 0)) or 0)

    # basic sanity
    if bid is None or ask is None:
        return None, None, 0, 0
    if bid <= 0 or ask <= 0 or ask < bid:
        return None, None, 0, 0

    # Polymarket bounds sanity
    if bid > 1 or ask > 1:
        return None, None, 0, 0

    return bid, ask, bsz, asz

def spread_ratio(bid: float, ask: float) -> Optional[float]:
    """
    Spread ratio vs mid. Example: 0.03 = 3%
    """
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    return (ask - bid) / mid

def post_gtc(client: ClobClient, args: OrderArgs) -> Dict[str, Any]:
    signed = client.create_order(args)
    return client.post_order(signed, OrderType.GTC)

def try_fill_or_cancel(client: ClobClient, order_id: str, wait_sec: int = 6) -> str:
    deadline = time.time() + wait_sec
    last_status = "unknown"
    while time.time() < deadline:
        o = client.get_order(order_id)
        last_status = str(o.get("status", "")).lower()
        if last_status in ("filled", "canceled", "cancelled", "expired", "rejected"):
            return last_status
        time.sleep(1.0)
    try:
        client.cancel(order_id)
    except Exception:
        pass
    return last_status

# ----------------------------
# TRADING LOOP
# ----------------------------
def main():
    cfg = Config.from_env()
    client = make_client(cfg)

    log(f"ONLINE DRY_RUN={cfg.dry_run} trade=${cfg.trade_usd:.2f} maxOpen={cfg.max_open_positions} maxTrades={cfg.max_trades_per_day}")
    log(f"Filters: maxSpread={cfg.max_spread} minTopSize={cfg.min_top_size} endBuffer={cfg.end_buffer_sec}s momWindow={cfg.mom_window_sec}s thr={cfg.mom_threshold}")
    log(f"Risk: TP={cfg.tp_pct} SL={cfg.sl_pct} dailyMaxLoss=${cfg.daily_max_loss_usd:.2f}")

    open_positions: List[Dict[str, Any]] = []
    trades_today = 0
    day_key = datetime.now().strftime("%Y-%m-%d")

    mids: Dict[str, RollingMid] = {}

    # simple PnL tracking (approx)
    realized_pnl_usd = 0.0

    while True:
        today = datetime.now().strftime("%Y-%m-%d")
        if today != day_key:
            day_key = today
            trades_today = 0
            open_positions = []
            mids = {}
            realized_pnl_usd = 0.0
            log("New day: counters reset")

        if trades_today >= cfg.max_trades_per_day:
            time.sleep(10)
            continue

        # daily loss hard stop
        if realized_pnl_usd <= -abs(cfg.daily_max_loss_usd):
            log(f"DAILY STOP: realizedPnL=${realized_pnl_usd:.2f} <= -${abs(cfg.daily_max_loss_usd):.2f}. Sleeping...")
            time.sleep(60)
            continue

        try:
            ev = gamma_get_live_event(cfg)
            if not ev:
                log("Gamma: no live event for series, retry...")
                time.sleep(cfg.scan_interval_sec)
                continue

            markets = ev.get("markets") or []
            m0 = markets[0]
            yes_id, no_id = parse_token_ids(m0)

            # Fetch books with size
            yb, ya, ybsz, yasz = get_best_bid_ask_size(client, yes_id)
            nb, na, nbsz, nasz = get_best_bid_ask_size(client, no_id)

            # If any side invalid -> skip (no more 9e9 nonsense)
            if yb is None or ya is None or nb is None or na is None:
                log("Skip: orderbook missing/invalid (bid/ask None)")
                time.sleep(cfg.scan_interval_sec)
                continue

            # Liquidity filter (top size)
            if min(ybsz, yasz) < cfg.min_top_size or min(nbsz, nasz) < cfg.min_top_size:
                log(f"Skip: low top liquidity YES(bidSz={ybsz},askSz={yasz}) NO(bidSz={nbsz},askSz={nasz})")
                time.sleep(cfg.scan_interval_sec)
                continue

            ysp = spread_ratio(yb, ya)
            nsp = spread_ratio(nb, na)
            if ysp is None or nsp is None:
                log("Skip: spread calc invalid")
                time.sleep(cfg.scan_interval_sec)
                continue

            # IMPORTANT FIX:
            # Önceki kod min(ysp,nsp) ile bakıyordu -> biri iyi görünürse geçebiliyordu.
            # Daha güvenlisi: her ikisi de limitin altında olmalı.
            if ysp > cfg.max_spread or nsp > cfg.max_spread:
                log(f"Skip: spreads too wide YES={ysp:.3f} (bid={yb:.3f} ask={ya:.3f}) NO={nsp:.3f} (bid={nb:.3f} ask={na:.3f})")
                time.sleep(cfg.scan_interval_sec)
                continue

            now_ts = time.time()

            # track mids
            ym = (yb + ya) / 2
            nm = (nb + na) / 2
            mids.setdefault(yes_id, RollingMid()).add(now_ts, ym)
            mids[yes_id].prune(now_ts, cfg.mom_window_sec)
            mids.setdefault(no_id, RollingMid()).add(now_ts, nm)
            mids[no_id].prune(now_ts, cfg.mom_window_sec)

            # event timestamps
            end_iso = ev.get("endDate") or ev.get("endTime")
            start_iso = ev.get("startTime") or ev.get("startDate")
            end_ts = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).timestamp()
            start_ts = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp()

            # manage exits
            still_open: List[Dict[str, Any]] = []
            for pos in open_positions:
                token_id = pos["token_id"]
                entry = pos["entry_price"]
                qty = pos["qty"]
                pos_outcome = pos["outcome"]

                bid, ask, bsz, asz = get_best_bid_ask_size(client, token_id)
                if bid is None or ask is None:
                    still_open.append(pos)
                    continue

                mid = (bid + ask) / 2
                tp_hit = (mid - entry) / entry >= cfg.tp_pct
                sl_hit = (entry - mid) / entry >= cfg.sl_pct
                time_left = end_ts - time.time()

                if tp_hit or sl_hit or time_left <= cfg.end_buffer_sec:
                    reason = "TP" if tp_hit else ("SL" if sl_hit else "TIME")
                    exit_price = clamp_price(bid)  # exit at bid
                    log(f"EXIT {reason} {pos_outcome}: qty={qty} entry={entry:.3f} mid={mid:.3f} sell@{exit_price:.3f}")

                    # Approx realized PnL in USD (shares * price delta)
                    approx_pnl = qty * (exit_price - entry)
                    realized_pnl_usd += approx_pnl

                    if not cfg.dry_run:
                        args = OrderArgs(price=exit_price, size=qty, side=SELL, token_id=token_id)
                        resp = post_gtc(client, args)
                        oid = resp.get("orderID") or resp.get("orderId") or resp.get("id")
                        if oid:
                            st = try_fill_or_cancel(client, oid, wait_sec=6)
                            log(f"EXIT order status: {st}")
                    else:
                        log("DRY_RUN: exit order skipped")

                    time.sleep(cfg.cooldown_after_trade_sec)
                else:
                    still_open.append(pos)

            open_positions = still_open

            # entry logic
            if len(open_positions) >= cfg.max_open_positions:
                time.sleep(cfg.scan_interval_sec)
                continue

            # don't open too close to end
            if end_ts - time.time() <= cfg.end_buffer_sec + 8:
                time.sleep(cfg.scan_interval_sec)
                continue

            # momentum signal
            y_mom = mids.get(yes_id).mom()
            n_mom = mids.get(no_id).mom()

            pick = None
            if y_mom is not None and y_mom >= cfg.mom_threshold:
                pick = ("YES", yes_id)
            elif n
