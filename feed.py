"""
feed.py - market data for the SYN bots (NQ, 1-minute candles, US/Eastern).

Default  : Yahoo Finance via yfinance. Free, but DELAYED (often 10-15+ min).
Real-time: set the DATABENTO_API_KEY environment variable and the bots use
           Databento's CME Globex feed instead (needs a Databento account with
           real-time CME access; check their site for plans and CME fees).

Both paths return a DataFrame of CLOSED candles with columns
Open, High, Low, Close, Volume and a New York time index.
"""

import os
import threading

import pandas as pd

API_KEY = os.environ.get("DATABENTO_API_KEY")
USING_DATABENTO = bool(API_KEY)
DATA_LABEL = "Live data (Databento)" if USING_DATABENTO else "Delayed data (can be 10+ min)"
STALE_MINUTES = 5 if USING_DATABENTO else 30   # older than this = market closed / feed down

YF_SYMBOL = "NQ=F"
DB_DATASET = "GLBX.MDP3"
DB_SYMBOL = "NQ.c.0"          # front-month NQ by calendar
COLS = ["Open", "High", "Low", "Close", "Volume"]
ET = "America/New_York"


# ---------- free path: yfinance ----------

def _yf_bars() -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(YF_SYMBOL, interval="1m", period="5d",
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df.iloc[:-1]  # drop the still-forming candle


# ---------- real-time path: Databento ----------

class DatabentoFeed:
    """Seeds ~5 days of history, then keeps 1m bars updating from the live stream."""

    def __init__(self, key: str):
        import databento as db
        self.db = db
        self.key = key
        self.rows = {}
        self.lock = threading.Lock()
        self.live = None
        self._seed()
        self._start_live()
        print("Data source: Databento live (NQ front month)", flush=True)

    def _add_df(self, df):
        if df is None or len(df) == 0:
            return
        df = df[["open", "high", "low", "close", "volume"]]
        with self.lock:
            for ts, r in zip(df.index, df.itertuples(index=False)):
                self.rows[pd.Timestamp(ts)] = (float(r[0]), float(r[1]), float(r[2]),
                                               float(r[3]), float(r[4]))

    def _seed(self):
        try:
            end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)).floor("min")
            start = end - pd.Timedelta(days=5)
            hist = self.db.Historical(self.key)
            data = hist.timeseries.get_range(
                dataset=DB_DATASET, symbols=DB_SYMBOL, stype_in="continuous",
                schema="ohlcv-1m", start=start, end=end)
            self._add_df(data.to_df())
        except Exception as ex:
            print(f"Databento history seed failed: {ex}", flush=True)

    def _start_live(self):
        db = self.db
        replay_from = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=23, minutes=50)
        for start in (replay_from, None):     # fall back to live-only if replay is refused
            try:
                live = db.Live(key=self.key)
                kwargs = dict(dataset=DB_DATASET, schema="ohlcv-1m",
                              stype_in="continuous", symbols=DB_SYMBOL)
                if start is not None:
                    kwargs["start"] = start
                live.subscribe(**kwargs)
                live.add_callback(self._on_record)
                live.start()
                self.live = live
                return
            except Exception as ex:
                print(f"Databento live start failed (start={start}): {ex}", flush=True)

    def _on_record(self, r):
        db = self.db
        if isinstance(r, db.OHLCVMsg):
            ts = getattr(r, "pretty_ts_event", None)
            if ts is None:
                ts = pd.Timestamp(r.ts_event, unit="ns", tz="UTC")

            def px(name):
                v = getattr(r, "pretty_" + name, None)
                return float(v) if v is not None else getattr(r, name) / 1e9

            with self.lock:
                self.rows[pd.Timestamp(ts)] = (px("open"), px("high"), px("low"),
                                               px("close"), float(r.volume))
        elif isinstance(r, db.ErrorMsg):
            print(f"Databento error: {r}", flush=True)

    def ensure_live(self):
        try:
            ok = self.live is not None and self.live.is_connected()
        except Exception:
            ok = self.live is not None
        if not ok:
            print("Databento stream not connected - restarting", flush=True)
            self._start_live()

    def get_df(self) -> pd.DataFrame:
        with self.lock:
            items = sorted(self.rows.items(), key=lambda kv: kv[0])
        if not items:
            return pd.DataFrame()
        idx = pd.DatetimeIndex([k for k, _ in items])
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        df = pd.DataFrame([v for _, v in items], index=idx.tz_convert(ET), columns=COLS)
        cutoff = df.index[-1] - pd.Timedelta(days=6)
        with self.lock:
            for k in [k for k in self.rows if pd.Timestamp(k).tz_convert(ET) < cutoff]:
                del self.rows[k]
        return df[df.index >= cutoff]


_feed = None


def fetch_bars() -> pd.DataFrame:
    """Closed 1m candles. Empty DataFrame if no data is available yet."""
    global _feed
    if USING_DATABENTO:
        if _feed is None:
            _feed = DatabentoFeed(API_KEY)
        _feed.ensure_live()
        return _feed.get_df()
    return _yf_bars()
