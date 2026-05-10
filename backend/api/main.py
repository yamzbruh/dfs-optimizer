"""FastAPI backend for DFS War Room dashboard.

Endpoints
---------
* ``POST /api/upload`` — upload DK salary CSV, parse, return player pool
* ``POST /api/projections`` — build projections for parsed players (DK avg
  proxy until full slate feature inference is wired in V2)
* ``POST /api/optimize`` — generate lineups from current projections
* ``GET /api/model-info`` — loaded model metadata and feature column lists
* ``POST /api/export`` — export current lineups as DK upload CSV
* ``GET /api/health`` — process health and model load flags
"""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

# Project root: backend/api/main.py → backend/api → backend → repo root
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.ingestion.dk_csv_exporter import DKLineupExporter  # noqa: E402
from data_pipeline.ingestion.dk_csv_parser import DKCSVParser, DKPlayer  # noqa: E402
from ml.training.points_model import PitcherPointsModel, PointsModel  # noqa: E402
from optimizer.constraints.lineup_optimizer import (  # noqa: E402
    LineupOptimizer,
    LineupResult,
    PlayerProjection,
)

app = FastAPI(
    title="DFS War Room API",
    description="Connects the Next.js dashboard to the DK parser, models, and optimizer.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_parser = DKCSVParser()
_optimizer = LineupOptimizer()
_exporter = DKLineupExporter()

_hitter_model: PointsModel | None = None
_pitcher_model: PitcherPointsModel | None = None

_current_players: list[DKPlayer] = []
_current_projections: list[PlayerProjection] = []
_current_lineups: list[LineupResult] = []
_upload_meta: dict = {}


# ---------------------------------------------------------------------------
# Pydantic response / request models
# ---------------------------------------------------------------------------


class PlayerResponse(BaseModel):
    """One player row returned after a successful salary CSV parse."""

    dk_id: str
    name: str
    team: str
    opponent: str
    position: str
    position_eligibility: list[str]
    salary: int
    avg_points_per_game: float
    is_pitcher: bool
    game_info: str


class ProjectionResponse(BaseModel):
    """Quantile projection bundle for the slate UI."""

    dk_id: str
    name: str
    team: str
    position: str
    salary: int
    pts_q15: float
    pts_q50: float
    pts_q85: float
    ownership_proj: float
    leverage_score: float
    is_pitcher: bool


class LineupPlayerResponse(BaseModel):
    """Single slot assignment inside a generated lineup."""

    dk_id: str
    name: str
    team: str
    salary: int
    slot: str


class LineupResponse(BaseModel):
    """One optimizer lineup with salary and validity flags."""

    lineup_number: int
    players: list[LineupPlayerResponse]
    total_salary: int
    projected_pts: float
    leverage_score: float
    is_valid: bool


class OptimizeRequest(BaseModel):
    """Optimizer controls from the dashboard."""

    n_lineups: int = Field(default=20, ge=1, le=150)
    locked_ids: list[str] = Field(default_factory=list)
    banned_ids: list[str] = Field(default_factory=list)
    max_exposure: float = Field(default=0.70, ge=0.0, le=1.0)


class UploadResponse(BaseModel):
    """Parse summary plus the full player pool for the slate."""

    player_count: int
    pitcher_count: int
    hitter_count: int
    game_count: int
    team_count: int
    games: list[str]
    sha256: str
    players: list[PlayerResponse]


class ModelInfoResponse(BaseModel):
    """Which models are loaded and which feature columns they expect."""

    hitter_metrics: dict | None = None
    pitcher_metrics: dict | None = None
    hitter_features: list[str] = Field(default_factory=list)
    pitcher_features: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Startup — load latest trained quantile models (if present)
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def load_models() -> None:
    """Load the most recent hitter and pitcher ``q50`` run ids on boot.

    Discovers ``points_q50_<run_id>.joblib`` and ``pitcher_q50_<run_id>.joblib``
    under ``data/models/``, picks the lexicographically last stem (timestamp
    runs sort correctly).  Failures are logged but do not crash the app.
    """
    global _hitter_model, _pitcher_model

    models_dir = Path("data/models")
    if not models_dir.is_dir():
        logger.warning(f"Models directory missing: {models_dir}")
        return

    try:
        hitter_files = sorted(models_dir.glob("points_q50_*.joblib"))
        if hitter_files:
            stem = hitter_files[-1].stem
            run_id = stem.replace("points_q50_", "", 1)
            _hitter_model = PointsModel()
            _hitter_model.load_models(run_id)
            logger.info(f"Loaded hitter models  run_id={run_id}")
        else:
            logger.warning("No hitter model files found (points_q50_*.joblib)")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to load hitter models: {exc}")
        _hitter_model = None

    try:
        pitcher_files = sorted(models_dir.glob("pitcher_q50_*.joblib"))
        if pitcher_files:
            stem = pitcher_files[-1].stem
            run_id = stem.replace("pitcher_q50_", "", 1)
            _pitcher_model = PitcherPointsModel()
            _pitcher_model.load_models(run_id)
            logger.info(f"Loaded pitcher models  run_id={run_id}")
        else:
            logger.warning("No pitcher model files found (pitcher_q50_*.joblib)")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to load pitcher models: {exc}")
        _pitcher_model = None


def _opponent_for_player(p: DKPlayer) -> str:
    """Return the opposing team abbreviation for display."""
    if p.team and p.home_team and p.away_team:
        return p.away_team if p.team == p.home_team else p.home_team
    return ""


def _primary_slot_label(p: DKPlayer) -> str:
    """Single position string for API (DK roster slot / primary)."""
    return p.dk_position or (p.position_eligibility[0] if p.position_eligibility else "")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/api/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a DraftKings salary CSV, parse it, and store the player pool."""
    global _current_players, _upload_meta, _current_projections, _current_lineups

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Expected a .csv file (DraftKings salary export).",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    sha256 = hashlib.sha256(content).hexdigest()[:12]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb")
    try:
        tmp.write(content)
        tmp.close()

        _parser.parse(tmp.name)
        result = _parser.last_result

        if result.validation_errors:
            logger.warning(
                f"Upload had {len(result.validation_errors)} validation message(s)"
            )

        if not result.players:
            raise HTTPException(
                status_code=422,
                detail="CSV parsed but no valid players were produced.",
            )

        _current_players = result.players
        _current_projections = []
        _current_lineups = []

        games = sorted(
            {
                f"{p.away_team}@{p.home_team}"
                for p in result.players
                if p.away_team and p.home_team
            }
        )
        teams = sorted({p.team for p in result.players if p.team})

        players_resp = [
            PlayerResponse(
                dk_id=p.dk_id,
                name=p.name,
                team=p.team,
                opponent=_opponent_for_player(p),
                position=_primary_slot_label(p),
                position_eligibility=list(p.position_eligibility),
                salary=p.salary,
                avg_points_per_game=float(p.avg_points_per_game),
                is_pitcher=p.is_pitcher,
                game_info=p.game_info_raw or "",
            )
            for p in result.players
        ]

        _upload_meta = {
            "player_count": len(result.players),
            "sha256": sha256,
            "games": games,
        }

        return UploadResponse(
            player_count=len(result.players),
            pitcher_count=sum(1 for p in result.players if p.is_pitcher),
            hitter_count=sum(1 for p in result.players if not p.is_pitcher),
            game_count=len(games),
            team_count=len(teams),
            games=games,
            sha256=sha256,
            players=players_resp,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Upload / parse failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/projections", response_model=list[ProjectionResponse])
async def generate_projections() -> list[ProjectionResponse]:
    """Build ``PlayerProjection`` rows for the current slate (DK avg proxy).

    V2 will join Statcast / feature rows and call ``PointsModel`` /
    ``PitcherPointsModel.predict``; for now quantiles are derived from
    ``AvgPointsPerGame`` so the dashboard and optimizer stay usable
    without a full inference pipeline on every slate upload.
    """
    global _current_projections

    if not _current_players:
        raise HTTPException(
            status_code=400,
            detail="No players loaded. Upload a salary CSV via POST /api/upload first.",
        )

    projections: list[PlayerProjection] = []
    responses: list[ProjectionResponse] = []

    for player in _current_players:
        if player.is_pitcher:
            pts_q50 = float(player.avg_points_per_game)
            pts_q15 = max(0.0, pts_q50 * 0.4)
            pts_q85 = pts_q50 * 2.0
            ownership_proj = 15.0
        else:
            pts_q50 = float(player.avg_points_per_game)
            pts_q15 = max(0.0, pts_q50 * 0.5)
            pts_q85 = pts_q50 * 1.8
            ownership_proj = 10.0

        leverage = pts_q50 / max(ownership_proj, 0.1)

        proj = PlayerProjection(
            player=player,
            pts_q15=round(pts_q15, 2),
            pts_q50=round(pts_q50, 2),
            pts_q85=round(pts_q85, 2),
            ownership_proj=round(ownership_proj, 2),
            leverage_score=round(leverage, 2),
        )
        projections.append(proj)

        responses.append(
            ProjectionResponse(
                dk_id=player.dk_id,
                name=player.name,
                team=player.team,
                position=_primary_slot_label(player),
                salary=player.salary,
                pts_q15=proj.pts_q15,
                pts_q50=proj.pts_q50,
                pts_q85=proj.pts_q85,
                ownership_proj=proj.ownership_proj,
                leverage_score=proj.leverage_score,
                is_pitcher=player.is_pitcher,
            )
        )

    _current_projections = projections

    logger.info(
        f"Projections ready for {len(projections)} players "
        f"({sum(1 for p in projections if p.player.is_pitcher)} P, "
        f"{sum(1 for p in projections if not p.player.is_pitcher)} H)"
    )

    return responses


@app.post("/api/optimize", response_model=list[LineupResponse])
async def optimize_lineups(request: OptimizeRequest) -> list[LineupResponse]:
    """Run the PuLP optimizer on the projection list built by ``/api/projections``."""
    global _current_lineups

    if not _current_projections:
        raise HTTPException(
            status_code=400,
            detail="No projections available. Call POST /api/projections first.",
        )

    locked = set(request.locked_ids) if request.locked_ids else None
    banned = set(request.banned_ids) if request.banned_ids else None

    tuned = [
        replace(p, max_exposure=request.max_exposure) for p in _current_projections
    ]

    lineups = _optimizer.generate_lineups(
        projections=tuned,
        n_lineups=request.n_lineups,
        locked_ids=locked,
        banned_ids=banned,
    )

    _current_lineups = lineups

    responses: list[LineupResponse] = []
    for i, lineup in enumerate(lineups, start=1):
        players = [
            LineupPlayerResponse(
                dk_id=player.dk_id,
                name=player.name,
                team=player.team,
                salary=player.salary,
                slot=slot,
            )
            for player, slot in lineup.players
        ]
        responses.append(
            LineupResponse(
                lineup_number=i,
                players=players,
                total_salary=lineup.total_salary,
                projected_pts=round(lineup.projected_pts, 2),
                leverage_score=round(lineup.leverage_score, 2),
                is_valid=lineup.is_valid,
            )
        )

    valid_count = sum(1 for lu in lineups if lu.is_valid)
    logger.info(f"Optimize complete: {len(lineups)} lineups, {valid_count} valid")

    return responses


@app.post("/api/export")
async def export_lineups() -> StreamingResponse:
    """Stream a DK-upload CSV built from the most recent ``/api/optimize`` run."""
    if not _current_lineups:
        raise HTTPException(
            status_code=400,
            detail="No lineups to export. Call POST /api/optimize first.",
        )

    lineup_data = [lu.players for lu in _current_lineups]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp_path = Path(tmp.name)
    try:
        tmp.close()
        _exporter.export(lineup_data, tmp_path)
        content = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)

    buf = io.BytesIO(content)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="DK_Upload.csv"'},
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Summarise which quantile models are loaded and their feature column lists."""
    hitter_features = list(_hitter_model.feature_columns) if _hitter_model else []
    pitcher_features = list(_pitcher_model.feature_columns) if _pitcher_model else []

    return ModelInfoResponse(
        hitter_metrics={
            "loaded": _hitter_model is not None,
            "feature_count": len(hitter_features),
        },
        pitcher_metrics={
            "loaded": _pitcher_model is not None,
            "feature_count": len(pitcher_features),
        },
        hitter_features=hitter_features,
        pitcher_features=pitcher_features,
    )


@app.get("/api/health")
async def health() -> dict:
    """Lightweight readiness probe for orchestrators and the frontend."""
    return {
        "status": "ok",
        "hitter_model": _hitter_model is not None,
        "pitcher_model": _pitcher_model is not None,
        "players_loaded": len(_current_players),
        "projections_ready": len(_current_projections) > 0,
        "lineups_ready": len(_current_lineups) > 0,
    }
