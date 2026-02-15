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

    trade_usd: float
    max_open_positions: int
    max_trades_per_day: int

    tp_pct: float
    sl_pct: float
    max_spread: float

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
        # basic pk check (0x + hex)
        if not pk.startswith("0x"):
            raise RuntimeError("POLYMARKET_PK must start with 0x")
        hexpart = pk[2:]
        if any(c not in "0123456789abcdefABCDEF" for c in hexpart):
            raise RuntimeError("POLYMARKET_PK is not valid hex")

        return Config(
            private_key=pk,
            chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
            gamma_host=os.getenv("GAMMA_HOST", "https://gamma-api.polymarket.com").strip(),
            clob_host=os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com").strip(),
            series_id=int(os.getenv("SERIES_ID", "10684")),

            dry_run=os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes", "y"),

            trade_usd=float(os.getenv("TRADE_USD", "6")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "2")),
            max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "10")),

            tp_pct=float(os.getenv("TP_PCT", "0.06")),
            sl_pct=float(os.getenv("SL_PCT", "0.04")),
            max_spread=float(os.getenv("MAX_SPREAD", "0.15")),

            scan_interval_sec=int(os.getenv("SCAN_INTERVAL_SEC", "3")),
            end_buffer_sec=int(os.getenv("END_BUFFER_SEC", "10")),
            cooldown_after_trade_sec=int(os.getenv("COOLDOWN_AFTER_TRADE_SEC", "20")),

            mom_window_sec=int(os.getenv("MOM_WINDOW_SEC", "60")),
            mom_threshold=float(os.getenv("MOM_THRESHOLD", "0.008")),
        )

# ----------------------------
# LOG
# ----------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ----------------------------
# STATE (in-memory)
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
    client = ClobClient(
        cfg.clob_host,
        key=cfg.private_key,
        chain_id=cfg.chain_id,
    )
    # creates L2 creds needed for trading endpoints
    client.set_api_creds(client.create_or_derive_api_creds())
    return client

def gamma_get_live_event(cfg: Config) -> Optional[Dict[str, Any]]:
    # Use series_id filter (this is the key for 5m recurring series)
    # order by startTime asc so first is nearest upcoming/current
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
    events = r.json() if isinstance(r.json(), list) else []
    if not events:
        return None

    now = datetime.now(timezone.utc).timestamp()
    for ev in events:
        # choose an event that hasn't ended yet (with buffer)
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
        # must have markets list
        markets = ev.get("markets") or []
        if not markets:
            continue
        m0 = markets[0]
        # only tradable
        if m0.get("acceptingOrders") is False:
            continue
        return ev

    return None

def parse_token_ids(market_obj: Dict[str, Any]) -> Tuple[str, str]:
    # clobTokenIds can be list or a JSON string
    ids = market_obj.get("clobTokenIds")
    if isinstance(ids, str):
        ids = json.loads(ids)
    if not isinstance(ids, list) or len(ids) < 2:
        raise RuntimeError("Market missing clobTokenIds")
    yes_id = str(ids[0])
    no_id = str(ids[1])
    return yes_id, no_id

def get_best_bid_ask(client: ClobClient, token_id: str) -> Tuple[Optional[float], Optional[float]]:
    ob = client.get_order_book(token_id)
    bids = ob.bids or []
    asks = ob.asks or []
    bid = float(bids[0].price) if bids else None
    ask = float(asks[0].price) if asks else None
    return bid, ask

def spread_pct(bid: Optional[float], ask: Optional[float]) -> float:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return 9e9
    mid = (bid + ask) / 2
    return (ask - bid) / mid if mid > 0 else 9e9

def clamp_price(p: float) -> float:
    return max(0.01, min(0.99, p))

def shares_for_usd(usd: float, price: float) -> int:
    # Polymarket shares are integer
    if price <= 0:
        return 0
    return int(math.floor(usd / price))

def post_gtc(client: ClobClient, args: OrderArgs) -> Dict[str, Any]:
    signed = client.create_order(args)
    return client.post_order(signed, OrderType.GTC)

def try_fill_or_cancel(client: ClobClient, order_id: str, wait_sec: int = 8) -> str:
    deadline = time.time() + wait_sec
    last_status = "unknown"
    while time.time() < deadline:
        o = client.get_order(order_id)
        last_status = str(o.get("status", "")).lower()
        if last_status in ("filled", "canceled", "cancelled", "expired", "rejected"):
            return last_status
        time.sleep(1.0)
    # cancel if still open
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
    log(f"Filters: maxSpread={cfg.max_spread} endBuffer={cfg.end_buffer_sec}s momWindow={cfg.mom_window_sec}s thr={cfg.mom_threshold}")

    open_positions: List[Dict[str, Any]] = []
    trades_today = 0
    day_key = datetime.now().strftime("%Y-%m-%d")

    mids: Dict[str, RollingMid] = {}

    while True:
        # reset daily counter
        today = datetime.now().strftime("%Y-%m-%d")
        if today != day_key:
            day_key = today
            trades_today = 0
            open_positions = []
            log("New day: counters reset")

        if trades_today >= cfg.max_trades_per_day:
            time.sleep(10)
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

            # ensure orderbooks exist + spread ok
            yb, ya = get_best_bid_ask(client, yes_id)
            nb, na = get_best_bid_ask(client, no_id)

            ysp = spread_pct(yb, ya)
            nsp = spread_pct(nb, na)
            if min(ysp, nsp) > cfg.max_spread:
                log(f"Skip: spreads too wide YES={ysp:.3f} NO={nsp:.3f}")
                time.sleep(cfg.scan_interval_sec)
                continue

            now_ts = time.time()
            # track mids
            if yb is not None and ya is not None:
                ym = (yb + ya) / 2
                mids.setdefault(yes_id, RollingMid()).add(now_ts, ym)
                mids[yes_id].prune(now_ts, cfg.mom_window_sec)

            if nb is not None and na is not None:
                nm = (nb + na) / 2
                mids.setdefault(no_id, RollingMid()).add(now_ts, nm)
                mids[no_id].prune(now_ts, cfg.mom_window_sec)

            # manage exits
            still_open: List[Dict[str, Any]] = []
            for pos in open_positions:
                token_id = pos["token_id"]
                side = pos["side"]  # BUY on entry
                entry = pos["entry_price"]
                qty = pos["qty"]
                opened_at = pos["opened_at"]
                end_ts = pos["end_ts"]

                bid, ask = get_best_bid_ask(client, token_id)
                if bid is None or ask is None:
                    still_open.append(pos)
                    continue
                mid = (bid + ask) / 2

                # TP/SL on mid vs entry
                tp_hit = (mid - entry) / entry >= cfg.tp_pct
                sl_hit = (entry - mid) / entry >= cfg.sl_pct
                time_left = end_ts - time.time()

                if tp_hit or sl_hit or time_left <= cfg.end_buffer_sec:
                    reason = "TP" if tp_hit else ("SL" if sl_hit else "TIME")
                    exit_price = clamp_price(bid)  # sell at bid to exit fast
                    log(f"EXIT {reason}: token={token_id[:10]}.. qty={qty} entry={entry:.3f} mid={mid:.3f} sell@{exit_price:.3f}")

                    if not cfg.dry_run:
                        args = OrderArgs(
                            price=exit_price,
                            size=qty,
                            side=SELL,
                            token_id=token_id,
                        )
                        resp = post_gtc(client, args)
                        oid = resp.get("orderID") or resp.get("orderId") or resp.get("id")
                        if oid:
                            st = try_fill_or_cancel(client, oid, wait_sec=8)
                            log(f"EXIT order status: {st}")
                    else:
                        log("DRY_RUN: exit order skipped")

                    time.sleep(cfg.cooldown_after_trade_sec)
                else:
                    still_open.append(pos)

            open_positions = still_open

            # entry logic (only if we have room)
            if len(open_positions) >= cfg.max_open_positions:
                time.sleep(cfg.scan_interval_sec)
                continue

            # momentum signal using mid history
            y_mom = mids.get(yes_id).mom() if yes_id in mids else None
            n_mom = mids.get(no_id).mom() if no_id in mids else None

            # choose side
            pick = None
            if y_mom is not None and y_mom >= cfg.mom_threshold:
                pick = ("YES", yes_id, ya)  # buy at ask
            elif n_mom is not None and n_mom >= cfg.mom_threshold:
                pick = ("NO", no_id, na)

            if not pick:
                # no signal
                time.sleep(cfg.scan_interval_sec)
                continue

            outcome, token_id, ask_price = pick
            if ask_price is None:
                time.sleep(cfg.scan_interval_sec)
                continue

            price = clamp_price(float(ask_price))
            qty = shares_for_usd(cfg.trade_usd, price)
            if qty < 1:
                time.sleep(cfg.scan_interval_sec)
                continue

            # event times for time-based exit
            end_iso = ev.get("endDate") or ev.get("endTime")
            start_iso = ev.get("startTime") or ev.get("startDate")
            end_ts = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).timestamp()
            start_ts = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp()

            # don't open too close to end
            if end_ts - time.time() <= cfg.end_buffer_sec + 5:
                time.sleep(cfg.scan_interval_sec)
                continue

            log(f"ENTRY {outcome}: qty={qty} buy@{price:.3f} momYES={y_mom} momNO={n_mom}")

            if not cfg.dry_run:
                args = OrderArgs(
                    price=price,
                    size=qty,
                    side=BUY,
                    token_id=token_id,
                )
                resp = post_gtc(client, args)
                oid = resp.get("orderID") or resp.get("orderId") or resp.get("id")
                if oid:
                    st = try_fill_or_cancel(client, oid, wait_sec=8)
                    log(f"ENTRY order status: {st}")
                    if st != "filled":
                        # if not filled, don't count as trade / position
                        time.sleep(cfg.scan_interval_sec)
                        continue
            else:
                log("DRY_RUN: entry order skipped")

            # record open position
            open_positions.append({
                "side": BUY,
                "outcome": outcome,
                "token_id": token_id,
                "qty": qty,
                "entry_price": price,
                "opened_at": time.time(),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "event_slug": ev.get("slug"),
            })
            trades_today += 1

            time.sleep(cfg.scan_interval_sec)

        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}")
            time.sleep(max(3, cfg.scan_interval_sec))

if __name__ == "__main__":
    main()
