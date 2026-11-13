"""
SYN BASIC - all-session, basic-strategy signal bot for Nasdaq E-mini futures (NQ).
No order flow, no volume. The setup name is NOT shown in the signal.

Strategies (1m closed candles, US/Eastern):
  1. Opening Range Breakout   (New York hours only)
  2. Prior-day high/low breakout
  3. 30-bar range (Donchian) breakout with trend filter
  4. EMA trend pullback (EMA9 > EMA21 > EMA50, reclaim of EMA9)
  5. EMA 9/21 cross with EMA50 filter
  6. MACD signal-line cross with EMA50 filter
  7. Bollinger + RSI fade (sideways markets)
  8. RSI extreme reversal (crosses back through 30 / 70)

Runs every session (Asia, London, New York). Overnight stops are wider.
One signal per candle at most; cooldown per side. No daily cap by default.

Data: see feed.py (free yfinance is DELAYED; set DATABENTO_API_KEY for real-time).
NOT FINANCIAL ADVICE. Paper trade before risking money.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import timedelta

import discord
import pandas as pd
from discord.ext import tasks
from dotenv import load_dotenv

from feed import DATA_LABEL, STALE_MINUTES, fetch_bars

load_dotenv()

TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["SIGNALS_CHANNEL_ID"])

SYMBOL = "NQ=F"
LABEL = "NQ"
INTERVAL = "1m"
POLL_SECONDS = 60
TARGET_RS = (1, 2, 3)
TICK = 0.25
POINT_VALUE = 20.0            # $ per point, 1 NQ (MNQ is $2)
ATR_STOP_MULT = 1.5           # New York hours
ATR_STOP_MULT_OVERNIGHT = 2.0 # Asia + London (wider, choppier)
MIN_RISK = 8.0                # points
MAX_RISK = 80.0
COOLDOWN_BARS = 45            # min 1m bars between signals on the same side
MAX_PER_DAY = None            # None = no cap; set a number (e.g. 15) to limit


@dataclass
class Signal:
    side: str
    price: float
    stop: float
    targets: list
    risk: float
    bar_time: pd.Timestamp


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    pc = c.shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_up"], df["bb_lo"] = mid + 2 * sd, mid - 2 * sd

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd"] = macd
    df["macd_sig"] = macd.ewm(span=9, adjust=False).mean()

    df["hi30"] = df["High"].rolling(30).max().shift(1)
    df["lo30"] = df["Low"].rolling(30).min().shift(1)
    return df


def snap(x: float) -> float:
    return round(round(x / TICK) * TICK, 2)


def ref_date(ts):
    """Trade date: after the 16:00 ET close a bar counts toward the next day."""
    d = ts.date()
    return d + timedelta(days=1) if ts.strftime("%H:%M") >= "16:00" else d


def prior_day_levels(df: pd.DataFrame, ref):
    dates = df.index.date
    for d in sorted(set(dates), reverse=True):
        if d >= ref:
            continue
        s = df[dates == d].between_time("09:30", "15:59")
        if len(s) >= 300:
            return float(s["High"].max()), float(s["Low"].min())
    return None


def fresh_state(ref):
    return {"date": ref, "count": 0, "pdlv": None,
            "orb": {"long": False, "short": False},
            "pdb": {"long": False, "short": False}}


def check_signal(df: pd.DataFrame, state: dict):
    if len(df) < 60:
        return None
    ts = df.index[-1]
    t = ts.strftime("%H:%M")
    ref = ref_date(ts)
    if state.get("date") != ref:
        keep = state.get("last", {"long": None, "short": None})
        state.clear()
        state.update(fresh_state(ref))
        state["last"] = keep
    state.setdefault("last", {"long": None, "short": None})

    if MAX_PER_DAY is not None and state["count"] >= MAX_PER_DAY:
        return None

    now, prev = df.iloc[-1], df.iloc[-2]
    price, atr = float(now["Close"]), float(now["atr"])
    if pd.isna(atr) or pd.isna(now["bb_up"]) or pd.isna(now["hi30"]):
        return None

    ny = "09:30" <= t < "16:00"
    mult = ATR_STOP_MULT if ny else ATR_STOP_MULT_OVERNIGHT
    atr_risk = max(mult * atr, MIN_RISK)
    green, red = price > float(now["Open"]), price < float(now["Open"])

    found = []   # (side, risk, tag)

    # 1. Opening range breakout (New York only)
    if "09:45" <= t <= "15:45":
        day = df[df.index.date == ts.date()].between_time("09:30", "15:59")
        orb = day.between_time("09:30", "09:44")
        if len(orb) >= 10:
            orh, orl = float(orb["High"].max()), float(orb["Low"].min())
            mid = (orh + orl) / 2
            if not state["orb"]["long"] and prev["Close"] <= orh < price and MIN_RISK <= price - mid <= MAX_RISK:
                found.append(("long", price - mid, "orb"))
            if not state["orb"]["short"] and prev["Close"] >= orl > price and MIN_RISK <= mid - price <= MAX_RISK:
                found.append(("short", mid - price, "orb"))

    # 2. Prior-day high / low breakout
    if state["pdlv"] is None:
        state["pdlv"] = prior_day_levels(df, ref) or False
    if state["pdlv"]:
        pdh, pdl = state["pdlv"]
        if not state["pdb"]["long"] and prev["Close"] <= pdh < price:
            found.append(("long", atr_risk, "pdb"))
        if not state["pdb"]["short"] and prev["Close"] >= pdl > price:
            found.append(("short", atr_risk, "pdb"))

    # 3. 30-bar range breakout with trend filter
    if price > now["hi30"] >= prev["Close"] and now["ema21"] > now["ema50"]:
        found.append(("long", atr_risk, None))
    if price < now["lo30"] <= prev["Close"] and now["ema21"] < now["ema50"]:
        found.append(("short", atr_risk, None))

    # 4. EMA trend pullback
    if now["ema9"] > now["ema21"] > now["ema50"] and prev["Close"] <= prev["ema9"] and price > now["ema9"] and green:
        found.append(("long", atr_risk, None))
    if now["ema9"] < now["ema21"] < now["ema50"] and prev["Close"] >= prev["ema9"] and price < now["ema9"] and red:
        found.append(("short", atr_risk, None))

    # 5. EMA 9/21 cross
    if prev["ema9"] <= prev["ema21"] and now["ema9"] > now["ema21"] and price > now["ema50"]:
        found.append(("long", atr_risk, None))
    if prev["ema9"] >= prev["ema21"] and now["ema9"] < now["ema21"] and price < now["ema50"]:
        found.append(("short", atr_risk, None))

    # 6. MACD cross
    if prev["macd"] <= prev["macd_sig"] and now["macd"] > now["macd_sig"] and price > now["ema50"]:
        found.append(("long", atr_risk, None))
    if prev["macd"] >= prev["macd_sig"] and now["macd"] < now["macd_sig"] and price < now["ema50"]:
        found.append(("short", atr_risk, None))

    # 7. Bollinger + RSI fade (sideways only)
    if abs(float(now["ema21"]) - float(now["ema50"])) < 1.5 * atr:
        if prev["Close"] < prev["bb_lo"] and price > now["bb_lo"] and now["rsi"] < 40 and green:
            found.append(("long", atr_risk, None))
        if prev["Close"] > prev["bb_up"] and price < now["bb_up"] and now["rsi"] > 60 and red:
            found.append(("short", atr_risk, None))

    # 8. RSI extreme reversal
    if prev["rsi"] < 30 <= now["rsi"] and green:
        found.append(("long", atr_risk, None))
    if prev["rsi"] > 70 >= now["rsi"] and red:
        found.append(("short", atr_risk, None))

    def cooled(side):
        last = state["last"][side]
        return last is None or int((df.index > last).sum()) >= COOLDOWN_BARS

    for side, risk, tag in found:
        if tag:
            state[tag][side] = True      # these only fire once per side per day
        if not cooled(side) or risk > MAX_RISK:
            continue
        state["count"] += 1
        state["last"][side] = ts
        sign = 1 if side == "long" else -1
        return Signal(side, snap(price), snap(price - sign * risk),
                      [snap(price + sign * risk * r) for r in TARGET_RS], risk, ts)
    return None


def build_embed(sig: Signal) -> discord.Embed:
    bull = sig.side == "long"
    arrow, word = ("▲", "Bullish") if bull else ("▼", "Bearish")
    ts = int(sig.bar_time.timestamp())

    e = discord.Embed(
        title=f"{arrow} {word} SYN · {LABEL}",
        color=discord.Color.green() if bull else discord.Color.red(),
    )
    e.add_field(name="Price", value=f"`{sig.price:.2f}`", inline=True)
    e.add_field(name="Signal time", value=f"<t:{ts}:t> (<t:{ts}:R>)", inline=True)
    e.add_field(name="Chart", value=f"`{INTERVAL}`", inline=True)

    e.add_field(name="Entry", value=f"`{sig.price:.2f}`", inline=True)
    e.add_field(name="Stop", value=f"`{sig.stop:.2f}`", inline=True)
    e.add_field(
        name="Targets",
        value="\n".join(f"TP{i+1} `{t:.2f}` ({r}R)"
                        for i, (t, r) in enumerate(zip(sig.targets, TARGET_RS))),
        inline=True,
    )
    e.add_field(
        name="Risk",
        value=f"{sig.risk:.2f} pts · ${sig.risk * POINT_VALUE:,.0f} per NQ · "
              f"${sig.risk * POINT_VALUE / 10:,.0f} per MNQ",
        inline=False,
    )
    age = max(0, int((pd.Timestamp.now(tz="UTC") - sig.bar_time).total_seconds() // 60))
    e.set_footer(text=f"Data was {age} min old when posted · not financial advice")
    e.timestamp = discord.utils.utcnow()
    return e


class SynBasicBot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.last_bar = None
        self.state = {}

    async def setup_hook(self):
        self.scan.start()

    async def on_ready(self):
        print(f"SYN BASIC online as {self.user}")

    @tasks.loop(seconds=POLL_SECONDS)
    async def scan(self):
        try:
            df = await asyncio.to_thread(fetch_bars)
            if df.empty:
                return
            latest = df.index[-1]
            if self.last_bar is None:       # don't fire on old data at startup
                self.last_bar = latest
                return
            if latest <= self.last_bar:
                return
            if (pd.Timestamp.now(tz="America/New_York") - latest) > pd.Timedelta(minutes=STALE_MINUTES):
                self.last_bar = latest      # market closed / stale data
                return
            self.last_bar = latest

            sig = check_signal(add_indicators(df), self.state)
            if not sig:
                return
            channel = self.get_channel(CHANNEL_ID) or await self.fetch_channel(CHANNEL_ID)
            await channel.send(embed=build_embed(sig))
        except Exception as ex:
            print(f"[{time.strftime('%X')}] scan error: {ex}")

    @scan.before_loop
    async def before_scan(self):
        await self.wait_until_ready()


if __name__ == "__main__":
    SynBasicBot().run(TOKEN)
