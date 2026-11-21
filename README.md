# gta6-trophy-rarity-forecast-01-entry-v6

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saifulsidik801-ship-it/gta6-trophy-rarity-forecast-01-entry-v6/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saifulsidik801-ship-it/gta6-trophy-rarity-forecast-01-entry-v6/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Final-regime trend forecaster for GTA VI trophy earn rates on Exophase. The practice's top entries hardcoded the 04:30 refresh rates, which are gone and cannot be recovered, so this entry is built for the final (Nov 26 target), a genuine 4.5-day extrapolation: per-trophy logit-linear fits on the last three distinct Exophase refresh states, the launch-night flat state dropped (it is a different, reviewer population that poisons every fit), slopes shrunk halfway toward the shared median slope, and a calibrated spread (random walk + residual, narrowed 0.85x). Forecasts each trophy's earn rate at the case's target as five quantiles.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=mxhczr02&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
