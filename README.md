# DFS Optimizer

Production MLB DraftKings GPP lineup optimizer. Generates up to 150 contest-ready lineups per slate using XGBoost quantile regression for points projection, an ownership model for leverage scoring, and correlated Monte Carlo simulation for portfolio construction.

## Architecture

- **Data layer:** DK CSV ingestion + pybaseball + MLB Stats API + Odds API + OpenWeatherMap
- **ML layer:** XGBoost quantile regression (q15/q50/q85) + ownership model with Tweedie objective
- **Optimizer:** Block covariance Monte Carlo + Cholesky decomposition + PuLP/HiGHS solver
- **Backend:** FastAPI + Pydantic strict mode + Supabase
- **Frontend:** Next.js + Tailwind + Recharts

## Status

Pre-MVP. Foundation phase.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your credentials
```

## Project Structure

See `/docs/architecture.md` for full system architecture.
