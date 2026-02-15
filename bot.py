import os
import time
import json
import math
import requests
from dataclasses import dataclass
from typing import Optional, Tuple

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.order_builder.constants import BUY, SELL

load_dotenv()

# ================= CONFIG =================

@dataclass
class Config:
    private_key: str
    series_slug: str
    trade_amount: float
    max_open_positions: int
    scan_interval_sec: int
    dry_run: bool

    @staticmethod
    def from_env():
        return Config(
            private_key=os.environ["POLYMARKET_PK"],
            series_slug=os.getenv("SERIES_SLUG", "btc-up-or-down-5m"),
            trade_amount=float(os.getenv("TRADE_AMOUNT", "6")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "2")),
            scan_interval_sec=int(os.getenv("SCAN_INTERVAL_SEC", "15")),
            dry_run=os.getenv("DRY_RUN", "true").lower() == "true",
        )

# ================= LOG =================

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ================= GAMMA TOKEN FETCH =================

def get_live_tokens(series_slug: str) -> Optional[Tuple[str, str]]:
    url = "https://gamma-api.polymarket.com/events"
    r = requests.get(url, params={"slug": series_slug}, timeout=20)
    r.raise_for_status()
    events = r.json()

    if not events:
        return None

    event = events[0]
    markets = event.get("markets", [])

    for m in markets:
        if not m.get("enableOrderBook"):
            continue
        if m.get("closed"):
            continue

        tokens = m.get("clobTokenIds")
        if isinstance(tokens, str):
            tokens = json.loads(tokens)

        if isinstance(tokens, list) and len(tokens) >= 2:
            return tokens[0], tokens[1]

    return None

# ================= ORDERBOOK CHECK =================

def get_mid_price(client: ClobClient, token_id: str) -> Optional[float]:
    ob = client.get_order_book(token_id=token_id)
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids or not asks:
        return None

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    return (best_bid + best_ask) / 2

# ================= SIMPLE MOMENTUM =================

class Momentum:
    def __init__(self):
        self.history = []

    def update(self, price):
        now = time.time()
        self.history.append((now, price))
        self.history = [(t, p) for t, p in self.history if now - t <= 60]

    def signal(self):
        if len(self.history) < 2:
            return None
        first = self.history[0][1]
        last = self.history[-1][1]
        change = (last - first) / first
        if change > 0.005:
            return "UP"
        if change < -0.005:
            return "DOWN"
        return None

# ================= MAIN =================

def main():
    cfg = Config.from_env()

    client = ClobClient(
        host="https://clob.polymarket.com",
        key=cfg.private_key
    )

    log(f"ONLINE DRY_RUN={cfg.dry_run} trade=${cfg.trade_amount}")

    mom_yes = Momentum()
    mom_no = Momentum()

    while True:
        try:
            pair = get_live_tokens(cfg.series_slug)
            if not pair:
                log("No active market found, retry...")
                time.sleep(cfg.scan_interval_sec)
                continue

            yes_id, no_id = pair
            log(f"Live market YES={yes_id[:8]}... NO={no_id[:8]}...")

            price_yes = get_mid_price(client, yes_id)
            price_no = get_mid_price(client, no_id)

            if not price_yes or not price_no:
                log("Orderbook not ready...")
                time.sleep(cfg.scan_interval_sec)
                continue

            mom_yes.update(price_yes)
            mom_no.update(price_no)

            sig_yes = mom_yes.signal()
            sig_no = mom_no.signal()

            if sig_yes == "UP":
                log("Signal: BUY YES")
                if not cfg.dry_run:
                    client.create_order(
                        token_id=yes_id,
                        side=BUY,
                        price=price_yes,
                        size=cfg.trade_amount / price_yes,
                        order_type="market"
                    )

            if sig_no == "UP":
                log("Signal: BUY NO")
                if not cfg.dry_run:
                    client.create_order(
                        token_id=no_id,
                        side=BUY,
                        price=price_no,
                        size=cfg.trade_amount / price_no,
                        order_type="market"
                    )

        except Exception as e:
            log(f"ERROR: {e}")

        time.sleep(cfg.scan_interval_sec)

if __name__ == "__main__":
    main()
