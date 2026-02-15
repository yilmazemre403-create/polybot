import os
import time
import json
import requests
from dataclasses import dataclass
from typing import Optional, Tuple, List

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

load_dotenv()

# ================= CONFIG =================

@dataclass
class Config:
    private_key: str
    series_slug: str
    trade_usd: float
    scan_interval_sec: int
    dry_run: bool
    chain_id: int

    @staticmethod
    def from_env():
        pk = os.getenv("POLYMARKET_PK", "").strip()
        if not pk:
            raise RuntimeError("Missing env var: POLYMARKET_PK")
        # allow 0x, strip if present
        if pk.startswith("0x"):
            pk = pk[2:]
        return Config(
            private_key=pk,
            series_slug=os.getenv("SERIES_SLUG", "btc-up-or-down-5m").strip(),
            trade_usd=float(os.getenv("TRADE_USD", "6")),
            scan_interval_sec=int(os.getenv("SCAN_INTERVAL_SEC", "10")),
            dry_run=os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes", "y"),
            chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
        )

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ================= GAMMA: LIVE TOKENS =================

def get_live_tokens(series_slug: str) -> Optional[Tuple[str, str]]:
    url = "https://gamma-api.polymarket.com/events"
    r = requests.get(url, params={"slug": series_slug}, timeout=20)
    r.raise_for_status()
    events = r.json() or []
    if not events:
        return None

    ev = events[0]
    markets = ev.get("markets") or []
    if not markets:
        return None

    # pick first market with orderbook enabled & not closed
    for m in markets:
        if not m.get("enableOrderBook"):
            continue
        if m.get("closed"):
            continue

        toks = m.get("clobTokenIds")
        if isinstance(toks, str):
            try:
                toks = json.loads(toks)
            except Exception:
                toks = None

        if isinstance(toks, list) and len(toks) >= 2:
            return str(toks[0]), str(toks[1])

    return None

# ================= ORDERBOOK / MID =================

def get_best_bid_ask(client: ClobClient, token_id: str) -> Tuple[Optional[float], Optional[float]]:
    ob = client.get_order_book(token_id=token_id)
    # py_clob_client bazen dict döner, bazen obj - ikisini de handle
    bids = ob.get("bids", []) if isinstance(ob, dict) else (ob.bids or [])
    asks = ob.get("asks", []) if isinstance(ob, dict) else (ob.asks or [])

    if not bids or not asks:
        return None, None

    # dict format
    if isinstance(bids[0], dict):
        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
    else:
        best_bid = float(bids[0].price)
        best_ask = float(asks[0].price)

    return best_bid, best_ask

def get_mid(client: ClobClient, token_id: str) -> Optional[float]:
    bid, ask = get_best_bid_ask(client, token_id)
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0

# ================= MOMENTUM SIGNAL =================

class Momentum:
    def __init__(self, window_sec: int = 60, threshold: float = 0.008):
        self.window_sec = window_sec
        self.threshold = threshold
        self.hist: List[Tuple[float, float]] = []

    def update(self, price: float):
        now = time.time()
        self.hist.append((now, price))
        # keep ~3 windows
        keep = self.window_sec * 3
        self.hist = [(t, p) for (t, p) in self.hist if now - t <= keep]

    def signal(self) -> Optional[str]:
        now = time.time()
        past = [(t, p) for (t, p) in self.hist if now - t >= self.window_sec]
        if not past:
            return None
        p0 = past[0][1]
        p1 = self.hist[-1][1]
        mom = (p1 - p0) / p0
        if mom > self.threshold:
            return "UP"
        if mom < -self.threshold:
            return "DOWN"
        return None

# ================= ORDER =================

def place_buy(client: ClobClient, token_id: str, usd: float, price: float, dry_run: bool) -> bool:
    # shares = usd / price
    size = usd / price
    # min tick: round 2 decimals for size is safe
    size = float(f"{size:.2f}")
    price = float(f"{price:.2f}")  # price tick 0.01

    log(f"BUY token={token_id[:10]}... price={price} size={size} usd={usd} DRY_RUN={dry_run}")

    if dry_run:
        return True

    args = OrderArgs(
        token_id=str(token_id),
        side=BUY,
        price=price,
        size=size
    )

    signed = client.create_order(args)
    resp = client.post_order(signed, OrderType.GTC)
    oid = resp.get("orderID") or resp.get("id") or resp.get("order_id")
    log(f"ORDER SENT id={oid} resp={resp}")
    return True

# ================= MAIN =================

def main():
    cfg = Config.from_env()

    # IMPORTANT: chain_id=137
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=cfg.private_key,
        chain_id=cfg.chain_id,
    )

    # API creds derive (needed)
    client.set_api_creds(client.create_or_derive_api_creds())

    log(f"ONLINE DRY_RUN={cfg.dry_run} trade=${cfg.trade_usd} slug={cfg.series_slug} chain_id={cfg.chain_id}")

    mom_yes = Momentum(window_sec=60, threshold=0.008)
    mom_no = Momentum(window_sec=60, threshold=0.008)

    last_tokens = (None, None)

    while True:
        try:
            tokens = get_live_tokens(cfg.series_slug)
            if not tokens:
                log("Gamma: no live tokens, retry...")
                time.sleep(cfg.scan_interval_sec)
                continue

            yes_id, no_id = tokens

            if tokens != last_tokens:
                log(f"NEW MARKET TOKENS YES={yes_id[:10]}... NO={no_id[:10]}...")
                last_tokens = tokens
                mom_yes = Momentum(window_sec=60, threshold=0.008)
                mom_no = Momentum(window_sec=60, threshold=0.008)

            yes_mid = get_mid(client, yes_id)
            no_mid = get_mid(client, no_id)

            if yes_mid is None or no_mid is None:
                log("Orderbook not ready yet...")
                time.sleep(cfg.scan_interval_sec)
                continue

            mom_yes.update(yes_mid)
            mom_no.update(no_mid)

            # signal: if YES momentum up -> buy YES, if YES momentum down -> buy NO
            sig = mom_yes.signal()
            if sig == "UP":
                place_buy(client, yes_id, cfg.trade_usd, yes_mid, cfg.dry_run)
            elif sig == "DOWN":
                place_buy(client, no_id, cfg.trade_usd, no_mid, cfg.dry_run)

        except Exception as e:
            log("ERROR(FULL): " + repr(e))

        time.sleep(cfg.scan_interval_sec)

if __name__ == "__main__":
    main()
