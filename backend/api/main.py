"""FastAPI backend for DFS War Room dashboard.

Endpoints
---------
* ``GET /api/slates`` — today's DK draft groups + salary CSV status
* ``POST /api/select-slate`` — load slate by ``dg`` (salary CSV from disk or DK)
* ``POST /api/upload`` — upload DK salary CSV, parse, return player pool
* ``POST /api/projections`` — build projections via ``SlateInference`` (XGBoost)
  when initialized; otherwise DK avg fallback
* ``POST /api/ownership`` — Monte Carlo ownership proxy (optional; replaces flat 10%/15%)
* ``POST /api/optimize`` — generate lineups from current projections
* ``GET /api/model-info`` — loaded model metadata and feature column lists
* ``POST /api/export`` — export current lineups as DK upload CSV
* ``GET /api/health`` — process health and model load flags
* ``GET /api/lineup-status`` — IL/OUT/SUSP auto-ban report and DTD flags
"""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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
from data_pipeline.ingestion.lineup_status import (  # noqa: E402
    LineupStatusChecker,
    team_ids_from_dk_players,
)
from ml.features.ownership_projector import OwnershipProjector  # noqa: E402
from ml.inference.slate_inference import SlateInference  # noqa: E402
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

_ownership_projector: OwnershipProjector | None = None

_slate_inference: SlateInference | None = None

_status_checker: LineupStatusChecker | None = None
_auto_banned_ids: set[str] = set()
_lineup_status_report: list[dict[str, str]] = []


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


class LineupStatusRow(BaseModel):
    """One player row for the lineup / injury status panel."""

    name: str
    team: str
    dk_id: str
    status: str
    reason: str


class LineupStatusPayload(BaseModel):
    """Auto-banned (IL/OUT/SUSP) and DTD flags for the dashboard."""

    report: list[LineupStatusRow]
    auto_banned_ids: list[str]


class SlateInfoRow(BaseModel):
    """One DraftKings MLB Classic slate for today (draft group)."""

    dg: int
    name: str
    lock_time: str
    lock_time_et: str
    contest_count: int
    max_entries: int
    total_current_entries: int
    max_prize_pool: float
    csv_path: str | None
    csv_exists: bool


class SelectSlateResponse(BaseModel):
    """Slate selected from automation / DK draft group (mirrors upload + dg)."""

    dg: int
    csv_path: str
    lock_time: str
    lock_time_et: str
    player_count: int
    pitcher_count: int
    hitter_count: int
    game_count: int
    team_count: int
    games: list[str]
    sha256: str
    players: list[PlayerResponse]


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
    global _hitter_model, _pitcher_model, _ownership_projector, _slate_inference, _status_checker

    models_dir = Path("data/models")
    if models_dir.is_dir():
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
                logger.warning(
                    "No pitcher model files found (pitcher_q50_*.joblib)"
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load pitcher models: {exc}")
            _pitcher_model = None
    else:
        logger.warning(f"Models directory missing: {models_dir}")

    try:
        _ownership_projector = OwnershipProjector()
        logger.info("OwnershipProjector initialized")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to init OwnershipProjector: {exc}")
        _ownership_projector = None

    try:
        _slate_inference = SlateInference()
        _slate_inference.load_models()
        _slate_inference.load_feature_matrices()
        _slate_inference.load_vegas()
        logger.info("SlateInference initialized")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to init SlateInference: {exc}")
        _slate_inference = None

    try:
        _status_checker = LineupStatusChecker()
        logger.info("LineupStatusChecker initialized")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to init LineupStatusChecker: {exc}")
        _status_checker = None


def _build_fallback_projections(players: list[DKPlayer]) -> list[PlayerProjection]:
    """Fallback projections using DK avg when models unavailable."""
    projections = []
    for player in players:
        avg = float(player.avg_points_per_game)
        if player.is_pitcher:
            is_starter = (player.dk_position or "").upper() == "SP"
            if not is_starter:
                avg = min(avg, 15.0)
            q50, q15, q85 = avg, max(0.0, avg * 0.4), avg * 2.0
            ownership = 15.0
        else:
            q50, q15, q85 = avg, max(0.0, avg * 0.5), avg * 1.8
            ownership = 10.0
        leverage = q50 / max(ownership, 0.1)
        projections.append(
            PlayerProjection(
                player=player,
                pts_q15=round(q15, 2),
                pts_q50=round(q50, 2),
                pts_q85=round(q85, 2),
                ownership_proj=round(ownership, 2),
                leverage_score=round(leverage, 3),
            )
        )
    return projections


def _opponent_for_player(p: DKPlayer) -> str:
    """Return the opposing team abbreviation for display."""
    if p.team and p.home_team and p.away_team:
        return p.away_team if p.team == p.home_team else p.home_team
    return ""


def _primary_slot_label(p: DKPlayer) -> str:
    """Single position string for API (DK roster slot / primary)."""
    return p.dk_position or (p.position_eligibility[0] if p.position_eligibility else "")


def _players_to_responses(players: list[DKPlayer]) -> list[PlayerResponse]:
    return [
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
        for p in players
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/api/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a DraftKings salary CSV, parse it, and store the player pool."""
    global _current_players, _upload_meta, _current_projections, _current_lineups
    global _auto_banned_ids, _lineup_status_report

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
        _auto_banned_ids = set()
        _lineup_status_report = []
        if _status_checker is not None:
            _status_checker.reset_cache()

        games = sorted(
            {
                f"{p.away_team}@{p.home_team}"
                for p in result.players
                if p.away_team and p.home_team
            }
        )
        teams = sorted({p.team for p in result.players if p.team})

        players_resp = _players_to_responses(result.players)

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


@app.get("/api/slates", response_model=list[SlateInfoRow])
async def get_todays_slates() -> list[SlateInfoRow]:
    """Return all available slates for today with their salary CSV status."""
    from automation.scheduler import _get_todays_slates

    slates = _get_todays_slates()
    result: list[SlateInfoRow] = []
    for s in slates:
        lt = s["lock_time"]
        lock_iso = lt.isoformat() if hasattr(lt, "isoformat") else str(lt)
        lock_et = s["lock_time_et"]
        csv_path = s.get("csv_path")
        csv_str = str(csv_path) if csv_path else None
        exists = bool(csv_path and Path(csv_path).exists())
        result.append(
            SlateInfoRow(
                dg=int(s["dg"]),
                name=str(s.get("name", "")),
                lock_time=lock_iso,
                lock_time_et=lock_et,
                contest_count=int(s.get("contest_count", 0)),
                max_entries=int(s.get("max_entries", 0)),
                total_current_entries=int(s.get("total_current_entries", 0)),
                max_prize_pool=float(s.get("max_prize_pool", 0.0)),
                csv_path=csv_str,
                csv_exists=exists,
            )
        )
    return result


@app.post("/api/select-slate", response_model=SelectSlateResponse)
async def select_slate(dg: int = Query(..., description="DraftKings draft group id")) -> SelectSlateResponse:
    """Select a slate by draft group ID and load its salary CSV."""
    global _current_players, _current_projections, _current_lineups
    global _auto_banned_ids, _lineup_status_report

    from automation.scheduler import _download_salary_csv, _get_todays_slates

    slates = _get_todays_slates()
    slate = next((s for s in slates if int(s["dg"]) == int(dg)), None)
    if not slate:
        raise HTTPException(status_code=404, detail=f"Slate dg={dg} not found")

    csv_path = slate.get("csv_path")
    if not csv_path or not Path(csv_path).exists():
        csv_path = _download_salary_csv(int(dg), slate["lock_time"])
    if not csv_path:
        raise HTTPException(status_code=500, detail="Failed to download salary CSV")

    _parser.parse(str(csv_path))
    result = _parser.last_result
    if not result.players:
        raise HTTPException(
            status_code=422, detail="No players parsed from salary CSV"
        )

    _current_players = result.players
    _current_projections = []
    _current_lineups = []
    _auto_banned_ids = set()
    _lineup_status_report = []
    if _status_checker is not None:
        _status_checker.reset_cache()

    games = sorted(
        {
            f"{p.away_team}@{p.home_team}"
            for p in result.players
            if p.away_team and p.home_team
        }
    )
    teams = sorted({p.team for p in result.players if p.team})
    players_resp = _players_to_responses(result.players)
    lock_iso = slate["lock_time"].isoformat()

    return SelectSlateResponse(
        dg=int(dg),
        csv_path=str(csv_path),
        lock_time=lock_iso,
        lock_time_et=str(slate["lock_time_et"]),
        player_count=len(result.players),
        pitcher_count=sum(1 for p in result.players if p.is_pitcher),
        hitter_count=sum(1 for p in result.players if not p.is_pitcher),
        game_count=len(games),
        team_count=len(teams),
        games=games,
        sha256=(result.file_hash or "")[:12],
        players=players_resp,
    )


@app.post("/api/projections", response_model=list[ProjectionResponse])
async def generate_projections() -> list[ProjectionResponse]:
    """Build ``PlayerProjection`` rows for the current slate.

    Uses ``SlateInference`` (cached Statcast features + XGBoost) when
    available. Call ``POST /api/ownership`` after this to get real
    ownership percentages from Monte Carlo simulation.

    Ownership defaults to flat 10% (hitters) / 15% (pitchers) until
    ``POST /api/ownership`` runs the Monte Carlo ownership proxy.
    """
    global _current_projections, _current_lineups, _auto_banned_ids, _lineup_status_report

    if not _current_players:
        raise HTTPException(
            status_code=400,
            detail="No players loaded. Upload a CSV first.",
        )

    try:
        if _slate_inference is not None:
            logger.info("Using SlateInference for real model projections")
            _slate_inference.load_vegas()
            from automation.scheduler import get_mlb_probable_pitchers

            try:
                probable_pitchers = get_mlb_probable_pitchers()
                logger.info(
                    f"Probable pitchers loaded: {len(probable_pitchers)} teams"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to load probable pitchers: {exc}")
                probable_pitchers = {}

            projections = _slate_inference.build_projections(
                _current_players,
                use_models=True,
                probable_pitchers=probable_pitchers,
            )
        else:
            logger.warning("SlateInference not available — using DK avg fallback")
            projections = _build_fallback_projections(_current_players)

        _current_projections = projections
        _current_lineups = []

        try:
            if _status_checker is not None and _slate_inference is not None:
                tids = team_ids_from_dk_players(_current_players)
                _status_checker.load_statuses(tids)
                _auto_banned_ids = _status_checker.get_unavailable_dk_ids(
                    _current_players,
                    _slate_inference.match_dk_player_to_mlbam,
                )
                scratched_ids = _status_checker.get_scratched_dk_ids(
                    _current_players,
                    _slate_inference.match_dk_player_to_mlbam,
                )
                _auto_banned_ids = _auto_banned_ids | scratched_ids
                _lineup_status_report = _status_checker.get_status_report(
                    _current_players,
                    _slate_inference.match_dk_player_to_mlbam,
                )
            else:
                _auto_banned_ids = set()
                _lineup_status_report = []
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Lineup availability refresh failed: {exc}")
            _auto_banned_ids = set()
            _lineup_status_report = []

        logger.info(
            f"Projections ready for {len(projections)} players "
            f"({sum(1 for p in projections if p.player.is_pitcher)} P, "
            f"{sum(1 for p in projections if not p.player.is_pitcher)} H)"
        )

        return [
            ProjectionResponse(
                dk_id=p.player.dk_id,
                name=p.player.name,
                team=p.player.team,
                position=_primary_slot_label(p.player),
                salary=p.player.salary,
                pts_q15=p.pts_q15,
                pts_q50=p.pts_q50,
                pts_q85=p.pts_q85,
                ownership_proj=p.ownership_proj,
                leverage_score=p.leverage_score,
                is_pitcher=p.player.is_pitcher,
            )
            for p in projections
        ]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Projection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/ownership", response_model=list[ProjectionResponse])
async def project_ownership(
    n_sims: int = Query(default=10000, ge=1000, le=10000),
) -> list[ProjectionResponse]:
    """Run ownership projection for the current slate.

    Runs Monte Carlo sims + Vegas totals + value scores (several minutes at
    high ``n_sims``). Updates in-memory projections with ``ownership_proj``
    and ``leverage_score``. Safe to skip — lineups still work with flat
    ownership from ``/api/projections`` until this is called.

    Args:
        n_sims: Monte Carlo lineup draws for the frequency backbone (1k–10k).
    """
    global _current_projections, _current_lineups

    if not _current_projections:
        raise HTTPException(
            status_code=400,
            detail="No projections available. Call POST /api/projections first.",
        )

    if _ownership_projector is None:
        raise HTTPException(
            status_code=503,
            detail="Ownership projector not initialized.",
        )

    try:
        logger.info(f"Running ownership projection ({n_sims} sims)...")

        players = [p.player for p in _current_projections]

        ownership = _ownership_projector.project(
            players=players,
            base_projections=_current_projections,
            n_sims=n_sims,
            n_jobs=4,
        )

        updated = _ownership_projector.update_projections_with_ownership(
            _current_projections,
            ownership,
        )

        _current_projections = updated
        _current_lineups = []

        logger.info(
            f"Ownership projection complete for {len(updated)} players"
        )

        return [
            ProjectionResponse(
                dk_id=p.player.dk_id,
                name=p.player.name,
                team=p.player.team,
                position=_primary_slot_label(p.player),
                salary=p.player.salary,
                pts_q15=p.pts_q15,
                pts_q50=p.pts_q50,
                pts_q85=p.pts_q85,
                ownership_proj=p.ownership_proj,
                leverage_score=p.leverage_score,
                is_pitcher=p.player.is_pitcher,
            )
            for p in updated
        ]

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ownership projection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
    banned = (set(request.banned_ids) if request.banned_ids else set()) | _auto_banned_ids
    banned_arg = banned if banned else None

    tuned = [
        replace(p, max_exposure=request.max_exposure) for p in _current_projections
    ]

    lineups = _optimizer.generate_lineups(
        projections=tuned,
        n_lineups=request.n_lineups,
        locked_ids=locked,
        banned_ids=banned_arg,
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


@app.get("/api/lineup-status", response_model=LineupStatusPayload)
async def lineup_status() -> LineupStatusPayload:
    """IL/OUT/SUSP auto-bans and DTD flags from the last projections refresh."""
    rows = [LineupStatusRow(**r) for r in _lineup_status_report]
    return LineupStatusPayload(
        report=rows,
        auto_banned_ids=sorted(_auto_banned_ids),
    )


@app.get("/api/health")
async def health() -> dict:
    """Lightweight readiness probe for orchestrators and the frontend."""
    return {
        "status": "ok",
        "hitter_model": _hitter_model is not None,
        "pitcher_model": _pitcher_model is not None,
        "inference_ready": _slate_inference is not None,
        "players_loaded": len(_current_players),
        "projections_ready": len(_current_projections) > 0,
        "lineups_ready": len(_current_lineups) > 0,
    }
