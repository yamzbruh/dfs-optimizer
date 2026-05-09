"""Temporary Streamlit debug inspector for the DK salary CSV parser.

Internal tooling only. Lets us visually sanity-check parser output
on real DraftKings CSVs before wiring the parser into the production
ingestion pipeline.

Run:
    streamlit run scripts/debug_inspector.py

This file is intentionally not imported by anything else and is
scheduled for removal once the proper Next.js admin UI lands in V2.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from loguru import logger

# Ensure imports from the project root resolve when Streamlit runs this
# script directly (which sets sys.path to the script's directory).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_pipeline.ingestion.dk_csv_parser import (  # noqa: E402
    DKCSVParser,
    DKPlayer,
    ParseResult,
)


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------

st.set_page_config(page_title="DFS Debug Inspector", layout="wide")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _save_upload_to_temp(uploaded_file: Any) -> Path:
    """Persist the Streamlit upload to a temp file and return its path.

    The parser expects a filesystem path so it can SHA256-hash the
    raw bytes; ``UploadedFile`` is an in-memory ``BytesIO``.
    """
    suffix = Path(uploaded_file.name).suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, delete=False
    )
    try:
        tmp.write(uploaded_file.getvalue())
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def _players_to_dataframe(players: list[DKPlayer]) -> pd.DataFrame:
    """Flatten parsed players into a DataFrame for display."""
    if not players:
        return pd.DataFrame()
    rows = []
    for p in players:
        d = asdict(p)
        # Lists render better in Streamlit when shown as joined strings.
        d["position_eligibility"] = "/".join(p.position_eligibility)
        rows.append(d)
    df = pd.DataFrame(rows)
    column_order = [
        "name",
        "team",
        "dk_position",
        "position_eligibility",
        "salary",
        "game_info_raw",
        "avg_points_per_game",
        "is_pitcher",
        "dk_id",
        "away_team",
        "home_team",
        "game_date",
        "game_time_et",
    ]
    return df[[c for c in column_order if c in df.columns]]


SEVERITY_COLORS: dict[str, str] = {
    "warning": "#FFC107",   # yellow
    "error": "#FF7043",     # orange
    "critical": "#D32F2F",  # red
}


def _style_validation_errors(df: pd.DataFrame) -> Any:
    """Color the severity column by level."""

    def _row_style(row: pd.Series) -> list[str]:
        color = SEVERITY_COLORS.get(str(row.get("severity", "")).lower(), "")
        if not color:
            return [""] * len(row)
        # Light tinted background, bold severity text.
        return [
            f"background-color: {color}33; color: black"
            for _ in range(len(row))
        ]

    return df.style.apply(_row_style, axis=1)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------


def _render_sidebar(file_hash: str | None) -> None:
    st.sidebar.title("DFS Debug Inspector")
    st.sidebar.markdown(
        "**Purpose:** Temporary debug tool — remove in V2."
    )
    st.sidebar.markdown(
        "Visual sanity check for the DK salary CSV parser. "
        "Not the production dashboard."
    )
    st.sidebar.divider()
    st.sidebar.subheader("Current file")
    if file_hash:
        st.sidebar.code(file_hash, language=None)
    else:
        st.sidebar.markdown("_no file loaded_")


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------


def _section_upload() -> tuple[ParseResult | None, str | None]:
    """Section 1 — file upload + parse + summary banners."""
    st.header("1. File Upload")
    uploaded = st.file_uploader(
        "Upload a DraftKings salary CSV",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info(
            "Upload a DraftKings salary CSV (the file you download from "
            "the DK contest page) to inspect parser output."
        )
        return None, None

    try:
        tmp_path = _save_upload_to_temp(uploaded)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save uploaded file to temp")
        st.error(f"Could not stage uploaded file: {exc}")
        return None, None

    parser = DKCSVParser()
    try:
        parser.parse(tmp_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Parser raised unexpectedly")
        st.error(f"Parser crashed (this should not happen): {exc}")
        return None, None

    result = parser.last_result

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Raw rows", result.raw_row_count)
    with col2:
        st.metric("Parsed rows", result.parsed_row_count)
    with col3:
        st.metric(
            "Validation errors",
            len(result.validation_errors),
        )

    st.markdown("**SHA256 file hash**")
    st.code(result.file_hash, language=None)

    has_errors = any(
        e["severity"] in ("error", "critical")
        for e in result.validation_errors
    )
    if (
        result.parsed_row_count == result.raw_row_count
        and not has_errors
    ):
        st.success(
            f"Parsed {result.parsed_row_count} of {result.raw_row_count} "
            "rows cleanly."
        )
    elif result.parsed_row_count > 0:
        st.warning(
            f"Parsed {result.parsed_row_count} of {result.raw_row_count} "
            f"rows with {len(result.validation_errors)} validation issue(s)."
        )
    else:
        st.error(
            "Parser produced zero usable rows — check the validation "
            "errors section below."
        )

    return result, result.file_hash


def _section_slate_info(result: ParseResult) -> None:
    st.header("2. Slate Info")
    info = result.slate_info

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Slate date", str(info.get("slate_date") or "—"))
        st.metric("Players", info.get("player_count", 0))
    with col2:
        st.metric("Games", len(info.get("games", [])))
        st.metric("Pitchers", info.get("pitcher_count", 0))
    with col3:
        st.metric("Teams", len(info.get("teams", [])))
        st.metric("Hitters", info.get("hitter_count", 0))

    st.subheader("Games on slate")
    games = info.get("games", [])
    if games:
        st.dataframe(
            pd.DataFrame({"game": games}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown("_no games detected_")


def _section_player_table(players: list[DKPlayer]) -> None:
    st.header("3. Player Table")
    if not players:
        st.markdown("_no players to display_")
        return

    df = _players_to_dataframe(players)

    salary_min_raw = int(df["salary"].min())
    salary_max_raw = int(df["salary"].max())
    if salary_min_raw == salary_max_raw:
        salary_max_raw += 1  # st.slider requires min < max.

    col1, col2 = st.columns([2, 3])
    with col1:
        salary_range = st.slider(
            "Salary range",
            min_value=salary_min_raw,
            max_value=salary_max_raw,
            value=(salary_min_raw, salary_max_raw),
            step=100,
        )
    with col2:
        all_positions = sorted(
            {pos for p in players for pos in p.position_eligibility}
        )
        all_teams = sorted({p.team for p in players if p.team})
        positions_selected = st.multiselect(
            "Positions (any-of)",
            options=all_positions,
            default=all_positions,
        )
        teams_selected = st.multiselect(
            "Teams",
            options=all_teams,
            default=all_teams,
        )

    mask = (
        (df["salary"] >= salary_range[0])
        & (df["salary"] <= salary_range[1])
        & (df["team"].isin(teams_selected))
    )
    if positions_selected:
        position_set = set(positions_selected)
        # Match if ANY of the player's eligible positions is selected.
        mask &= df["position_eligibility"].apply(
            lambda s: bool(set(str(s).split("/")) & position_set)
        )

    filtered = df.loc[mask]

    st.markdown(
        f"**{len(filtered)}** of {len(df)} players match the filters."
    )
    display_cols = [
        "name",
        "team",
        "dk_position",
        "position_eligibility",
        "salary",
        "game_info_raw",
        "avg_points_per_game",
        "is_pitcher",
    ]
    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        hide_index=True,
    )


def _section_dual_eligible(players: list[DKPlayer]) -> None:
    st.header("4. Dual-Eligible Players")
    st.caption(
        "Players with more than one eligible position. These are the "
        "rows most likely to surface bugs in the optimizer's slot "
        "assignment logic."
    )
    duals = [p for p in players if len(p.position_eligibility) > 1]
    if not duals:
        st.markdown("_no dual-eligible players on this slate_")
        return

    df = pd.DataFrame(
        [
            {
                "name": p.name,
                "team": p.team,
                "position_eligibility": "/".join(p.position_eligibility),
                "salary": p.salary,
            }
            for p in duals
        ]
    )
    st.markdown(f"**{len(duals)}** dual-eligible players found.")
    st.dataframe(df, use_container_width=True, hide_index=True)


def _section_validation_errors(result: ParseResult) -> None:
    st.header("5. Validation Errors")
    errors = result.validation_errors
    if not errors:
        st.success("No validation errors. Clean parse.")
        return

    df = pd.DataFrame(errors)
    column_order = [c for c in ("row", "field", "message", "severity") if c in df.columns]
    df = df[column_order]

    counts = df["severity"].value_counts().to_dict() if "severity" in df.columns else {}
    if counts:
        chip_cols = st.columns(len(counts))
        for (severity, count), col in zip(counts.items(), chip_cols):
            with col:
                st.metric(severity.capitalize(), int(count))

    st.markdown(
        '<div style="border:2px solid #D32F2F; padding:8px; '
        'border-radius:6px;">',
        unsafe_allow_html=True,
    )
    st.dataframe(
        _style_validation_errors(df),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _section_raw_data(players: list[DKPlayer]) -> None:
    with st.expander("Show raw parsed data"):
        if not players:
            st.markdown("_no players parsed_")
            return
        st.dataframe(
            _players_to_dataframe(players),
            use_container_width=True,
            hide_index=True,
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    st.title("DFS Optimizer — Debug Inspector")
    st.caption(
        "Internal tooling for verifying DK salary CSV parser output. "
        "Not the production dashboard."
    )

    result, file_hash = _section_upload()
    _render_sidebar(file_hash)

    if result is None:
        return

    st.divider()
    _section_slate_info(result)
    st.divider()
    _section_player_table(result.players)
    st.divider()
    _section_dual_eligible(result.players)
    st.divider()
    _section_validation_errors(result)
    st.divider()
    _section_raw_data(result.players)


if __name__ == "__main__":
    main()
