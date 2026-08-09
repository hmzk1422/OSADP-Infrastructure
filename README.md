# Intraday-Momentum-Composite
Systematic framework for capturing short-term alpha in high-volatility crypto-asset markets. By constructing a synthetic, equal-weighted intraday index, we neutralize idiosyncratic risk while isolating systemic momentum. We hypothesize that the persistence of price action is functionally linked to the second derivative of price (MA Acceleration). By identifying regimes where MA acceleration remains $\text{acc} > 0$ (long) or $\text{acc} < 0$ (short), we optimize entry points for RSI-based mean reversion (snapback) strategies, effectively simulating the structural drift of traditional equity indices within a high-frequency digital asset context.

# Focus
1. Investigating how RSI snapback speed has a very strong correlation with MA acceleration.
2. Approaches to solving the 'momnentum decay' factor relating to MA acceleration
3. Creathing synthetic asset - Intraday based S&P500 model where instead we are taking advantage of high momentum crypto
4. Utilizing ML and PDE to solve for momentum decay issue

# Guide

## Running on phone 24/7
1. Enable Autostart
Xiaomi blocks apps from starting or staying active unless explicitly told otherwise.

Open your phone's Security app (or go to Settings > Apps > Permissions).

Tap on Auto-start.

Find Termux and toggle it On.

2. Remove Battery Restrictions
Go to Settings > Apps > Manage apps.

Search for Termux and tap on it.

Tap Battery saver (or App battery saver) and change it to "No restrictions".

3. Lock Termux in the Recents Menu
This prevents MIUI from clearing the app out of memory when RAM runs low.

Open Termux so it is active in your background history.

Swipe up from the bottom and hold to open your Recent Apps screen.

Long-press the Termux window (or swipe it downward depending on your MIUI version) until a menu appears.

Tap the Padlock icon. A small lock should appear on the app card.

## On Termux
termux-wake-lock
pkg update && pkg upgrade -y
pkg install python git clang make -y
termux-setup-storage

git clone <your-repository-url>
cd <your-repository-folder>

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nohup python bot.py > bot.log 2>&1 &

nohup python -m main.execution > bot.log 2>&1 &

python -m main.executor

## Downloading Research File
```
python binance_downloader.py \
  --symbol BTCUSDT \
  --start-date 2026-08-01 \
  --end-date 2026-08-07 \
  --outdir ./research/csv
```