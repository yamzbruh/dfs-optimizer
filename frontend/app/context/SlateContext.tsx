"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

// ─── API base ───────────────────────────────────────────────────────────────

const API_BASE = "http://localhost:8000";

// ─── API response types (match FastAPI) ─────────────────────────────────────

export interface PlayerResponse {
  dk_id: string;
  name: string;
  team: string;
  opponent: string;
  position: string;
  position_eligibility: string[];
  salary: number;
  avg_points_per_game: number;
  is_pitcher: boolean;
  game_info: string;
}

export interface ProjectionResponse {
  dk_id: string;
  name: string;
  team: string;
  position: string;
  salary: number;
  pts_q15: number;
  pts_q50: number;
  pts_q85: number;
  ownership_proj: number;
  leverage_score: number;
  is_pitcher: boolean;
}

export interface LineupPlayerResponse {
  dk_id: string;
  name: string;
  team: string;
  salary: number;
  slot: string;
}

export interface LineupResponse {
  lineup_number: number;
  players: LineupPlayerResponse[];
  total_salary: number;
  projected_pts: number;
  leverage_score: number;
  is_valid: boolean;
}

export interface SlateInfo {
  player_count: number;
  pitcher_count: number;
  hitter_count: number;
  game_count: number;
  team_count: number;
  games: string[];
  sha256: string;
  file_name?: string;
  display_date?: string;
  lock_time?: string;
  /** Set when slate was chosen via ``POST /api/select-slate`` */
  draft_group_id?: number;
  lock_time_et?: string;
  csv_path?: string;
}

/** One row from ``GET /api/slates`` */
export interface DraftGroupSlateRow {
  dg: number;
  name: string;
  lock_time: string;
  lock_time_et: string;
  contest_count: number;
  max_entries: number;
  total_current_entries: number;
  max_prize_pool: number;
  csv_path: string | null;
  csv_exists: boolean;
}

export interface SelectSlateApiResponse {
  dg: number;
  csv_path: string;
  lock_time: string;
  lock_time_et: string;
  player_count: number;
  pitcher_count: number;
  hitter_count: number;
  game_count: number;
  team_count: number;
  games: string[];
  sha256: string;
  players: PlayerResponse[];
}

export interface ModelInfoResponse {
  hitter_metrics: { loaded?: boolean; feature_count?: number } | null;
  pitcher_metrics: { loaded?: boolean; feature_count?: number } | null;
  hitter_features: string[];
  pitcher_features: string[];
}

export interface LineupStatusRow {
  name: string;
  team: string;
  dk_id: string;
  status: string;
  reason: string;
}

export interface LineupStatusPayload {
  report: LineupStatusRow[];
  auto_banned_ids: string[];
}

export interface HealthResponse {
  status: string;
  hitter_model: boolean;
  pitcher_model: boolean;
  players_loaded: number;
  projections_ready: boolean;
  lineups_ready: boolean;
}

// ─── UI row types (tables / charts) ───────────────────────────────────────────

export type LineupStatus =
  | "confirmed_starting"
  | "projected_starting"
  | "unknown"
  | "scratched";

export interface Player {
  dk_id: string;
  name: string;
  team: string;
  opp: string;
  dk_position: string;
  position_eligibility: string[];
  salary: number;
  avg_pts: number;
  proj_pts: number;
  proj_pts_q15: number;
  proj_pts_q85: number;
  ownership: number;
  leverage: number;
  status: LineupStatus;
  is_pitcher: boolean;
  game_info: string;
}

export interface LineupPlayer {
  player: Player;
  slot: string;
}

export interface Lineup {
  id: number;
  players: LineupPlayer[];
  total_salary: number;
  projected_pts: number;
  leverage_score: number;
  portfolio_score: number;
  is_valid: boolean;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

async function readApiError(res: Response): Promise<string> {
  try {
    const j: unknown = await res.json();
    if (j && typeof j === "object" && "detail" in j) {
      const d = (j as { detail: unknown }).detail;
      if (typeof d === "string") return d;
      if (Array.isArray(d))
        return d
          .map((x) =>
            x && typeof x === "object" && "msg" in x
              ? String((x as { msg: string }).msg)
              : String(x)
          )
          .join("; ");
    }
  } catch {
    /* ignore */
  }
  return res.statusText || `HTTP ${res.status}`;
}

function parseLockIsoFromGameInfo(gameInfo: string): string | null {
  const re =
    /(\d{1,2}\/\d{1,2}\/\d{4})\s+(\d{1,2}:\d{2}\s*(?:AM|PM))/i;
  const m = gameInfo.match(re);
  if (!m) return null;
  const dt = new Date(`${m[1]} ${m[2]}`);
  if (Number.isNaN(dt.getTime())) return null;
  return dt.toISOString();
}

function displayDateFromGameInfo(gameInfo: string): string | null {
  const re = /(\d{1,2}\/\d{1,2}\/\d{4})/;
  const m = gameInfo.match(re);
  return m ? m[1] : null;
}

export function mergeToPlayer(
  proj: ProjectionResponse,
  roster?: PlayerResponse
): Player {
  const posEl =
    roster?.position_eligibility?.length
      ? roster.position_eligibility
      : proj.is_pitcher
        ? ["P"]
        : [proj.position];
  return {
    dk_id: proj.dk_id,
    name: proj.name,
    team: proj.team,
    opp: roster?.opponent ?? "",
    dk_position: proj.position,
    position_eligibility: posEl,
    salary: proj.salary,
    avg_pts: roster?.avg_points_per_game ?? 0,
    proj_pts: proj.pts_q50,
    proj_pts_q15: proj.pts_q15,
    proj_pts_q85: proj.pts_q85,
    ownership: proj.ownership_proj,
    leverage: proj.leverage_score,
    status: "unknown",
    is_pitcher: proj.is_pitcher,
    game_info: roster?.game_info ?? "",
  };
}

function playerFromLineupSlot(
  lp: LineupPlayerResponse,
  projMap: Map<string, ProjectionResponse>,
  rosterMap: Map<string, PlayerResponse>
): Player {
  const proj = projMap.get(lp.dk_id);
  const roster = rosterMap.get(lp.dk_id);
  if (proj) return mergeToPlayer(proj, roster);
  const isPitcher = roster?.is_pitcher ?? false;
  return {
    dk_id: lp.dk_id,
    name: lp.name,
    team: lp.team,
    opp: roster?.opponent ?? "",
    dk_position: roster?.position ?? (isPitcher ? "P" : ""),
    position_eligibility: roster?.position_eligibility ?? [],
    salary: lp.salary,
    avg_pts: roster?.avg_points_per_game ?? 0,
    proj_pts: 0,
    proj_pts_q15: 0,
    proj_pts_q85: 0,
    ownership: 0,
    leverage: 0,
    status: "unknown",
    is_pitcher: isPitcher,
    game_info: roster?.game_info ?? "",
  };
}

function mapLineupResponse(
  lr: LineupResponse,
  projMap: Map<string, ProjectionResponse>,
  rosterMap: Map<string, PlayerResponse>
): Lineup {
  const players: LineupPlayer[] = lr.players.map((p) => ({
    player: playerFromLineupSlot(p, projMap, rosterMap),
    slot: p.slot,
  }));
  return {
    id: lr.lineup_number,
    players,
    total_salary: lr.total_salary,
    projected_pts: lr.projected_pts,
    leverage_score: lr.leverage_score,
    portfolio_score: parseFloat(
      (lr.projected_pts + lr.leverage_score * 1.2).toFixed(1)
    ),
    is_valid: lr.is_valid,
  };
}

// ─── Context ────────────────────────────────────────────────────────────────

interface SlateContextValue {
  players: PlayerResponse[];
  projections: ProjectionResponse[];
  lineups: Lineup[];
  modelInfo: ModelInfoResponse | null;
  loading: boolean;
  error: string | null;
  slateInfo: SlateInfo | null;
  lockedIds: Set<string>;
  bannedIds: Set<string>;
  /** Set after successful POST /api/ownership; ``null`` if still flat defaults */
  ownershipSimsApplied: number | null;
  /** Merged projection + roster rows for tables and charts */
  playerPool: Player[];
  /** IL/OUT/SUSP auto-ban + DTD from last projections (server); null before first fetch */
  lineupStatus: LineupStatusPayload | null;
  uploadCSV: (file: File) => Promise<boolean>;
  generateProjections: () => Promise<void>;
  projectOwnership: (nSims?: number) => Promise<number | undefined>;
  generateLineups: (n: number, maxExposurePct?: number) => Promise<boolean>;
  exportLineups: () => Promise<void>;
  lockPlayer: (dk_id: string) => void;
  unlockPlayer: (dk_id: string) => void;
  banPlayer: (dk_id: string) => void;
  unbanPlayer: (dk_id: string) => void;
  fetchModelInfo: () => Promise<void>;
  fetchHealth: () => Promise<void>;
  /** Today's DK slates (no global loading spinner). */
  fetchTodaysSlates: () => Promise<DraftGroupSlateRow[]>;
  /** Select slate by draft group; loads players and runs projections. */
  selectSlateByDg: (dg: number) => Promise<boolean>;
}

const SlateContext = createContext<SlateContextValue | null>(null);

export function SlateProvider({ children }: { children: ReactNode }) {
  const [players, setPlayers] = useState<PlayerResponse[]>([]);
  const [projections, setProjections] = useState<ProjectionResponse[]>([]);
  const [lineups, setLineups] = useState<Lineup[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [slateInfo, setSlateInfo] = useState<SlateInfo | null>(null);
  const [lockedIds, setLockedIds] = useState<Set<string>>(() => new Set());
  const [bannedIds, setBannedIds] = useState<Set<string>>(() => new Set());
  /** Last ``n_sims`` used by POST /api/ownership; ``null`` = flat defaults only */
  const [ownershipSimsApplied, setOwnershipSimsApplied] = useState<number | null>(
    null
  );

  const [lineupStatus, setLineupStatus] = useState<LineupStatusPayload | null>(
    null
  );

  const rosterMap = useMemo(() => {
    const m = new Map<string, PlayerResponse>();
    for (const p of players) m.set(p.dk_id, p);
    return m;
  }, [players]);

  const projMap = useMemo(() => {
    const m = new Map<string, ProjectionResponse>();
    for (const p of projections) m.set(p.dk_id, p);
    return m;
  }, [projections]);

  const playerPool = useMemo(() => {
    return projections.map((pr) => mergeToPlayer(pr, rosterMap.get(pr.dk_id)));
  }, [projections, rosterMap]);

  const fetchLineupStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/lineup-status`);
      if (!res.ok) {
        setLineupStatus(null);
        return;
      }
      const j = (await res.json()) as LineupStatusPayload;
      setLineupStatus(j);
    } catch {
      setLineupStatus(null);
    }
  }, []);

  const withRequest = useCallback(
    async <T,>(fn: () => Promise<T>): Promise<T | undefined> => {
      setLoading(true);
      try {
        const out = await fn();
        setError(null);
        return out;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        return undefined;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const uploadCSV = useCallback(
    async (file: File): Promise<boolean> => {
      if (!file.name.toLowerCase().endsWith(".csv")) {
        setError("Invalid file type. Please upload a DraftKings salary CSV.");
        return false;
      }
      const fd = new FormData();
      fd.append("file", file);
      const ok = await withRequest(async () => {
        const res = await fetch(`${API_BASE}/api/upload`, {
          method: "POST",
          body: fd,
        });
        if (!res.ok) throw new Error(await readApiError(res));
        const data = (await res.json()) as {
          player_count: number;
          pitcher_count: number;
          hitter_count: number;
          game_count: number;
          team_count: number;
          games: string[];
          sha256: string;
          players: PlayerResponse[];
        };
        setPlayers(data.players);
        setProjections([]);
        setLineups([]);
        setLockedIds(new Set());
        setBannedIds(new Set());
        setOwnershipSimsApplied(null);
        setLineupStatus(null);
        const firstGi = data.players.find((p) => p.game_info)?.game_info ?? "";
        const lockIso = parseLockIsoFromGameInfo(firstGi);
        const dispDate = displayDateFromGameInfo(firstGi);
        setSlateInfo({
          player_count: data.player_count,
          pitcher_count: data.pitcher_count,
          hitter_count: data.hitter_count,
          game_count: data.game_count,
          team_count: data.team_count,
          games: data.games,
          sha256: data.sha256,
          file_name: file.name,
          display_date: dispDate ?? undefined,
          lock_time: lockIso ?? undefined,
          draft_group_id: undefined,
          lock_time_et: undefined,
          csv_path: undefined,
        });
        const projRes = await fetch(`${API_BASE}/api/projections`, {
          method: "POST",
        });
        if (!projRes.ok) throw new Error(await readApiError(projRes));
        const projList = (await projRes.json()) as ProjectionResponse[];
        setProjections(projList);
        await fetchLineupStatus();
        return true;
      });
      return ok === true;
    },
    [withRequest, fetchLineupStatus]
  );

  const fetchTodaysSlates = useCallback(async (): Promise<DraftGroupSlateRow[]> => {
    try {
      const res = await fetch(`${API_BASE}/api/slates`);
      if (!res.ok) {
        setError(await readApiError(res));
        return [];
      }
      return (await res.json()) as DraftGroupSlateRow[];
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      return [];
    }
  }, []);

  const selectSlateByDg = useCallback(
    async (dg: number): Promise<boolean> => {
      const ok = await withRequest(async () => {
        const res = await fetch(
          `${API_BASE}/api/select-slate?dg=${encodeURIComponent(String(dg))}`,
          { method: "POST" }
        );
        if (!res.ok) throw new Error(await readApiError(res));
        const data = (await res.json()) as SelectSlateApiResponse;
        setPlayers(data.players);
        setProjections([]);
        setLineups([]);
        setLockedIds(new Set());
        setBannedIds(new Set());
        setOwnershipSimsApplied(null);
        setLineupStatus(null);
        const firstGi = data.players.find((p) => p.game_info)?.game_info ?? "";
        const lockIsoFromGi = parseLockIsoFromGameInfo(firstGi);
        const dispDate = displayDateFromGameInfo(firstGi);
        setSlateInfo({
          player_count: data.player_count,
          pitcher_count: data.pitcher_count,
          hitter_count: data.hitter_count,
          game_count: data.game_count,
          team_count: data.team_count,
          games: data.games,
          sha256: data.sha256,
          file_name: `DKSalaries_dg${data.dg}.csv`,
          display_date: dispDate ?? undefined,
          lock_time: data.lock_time || lockIsoFromGi || undefined,
          draft_group_id: data.dg,
          lock_time_et: data.lock_time_et,
          csv_path: data.csv_path,
        });
        const projRes = await fetch(`${API_BASE}/api/projections`, {
          method: "POST",
        });
        if (!projRes.ok) throw new Error(await readApiError(projRes));
        const projList = (await projRes.json()) as ProjectionResponse[];
        setProjections(projList);
        await fetchLineupStatus();
        return true;
      });
      return ok === true;
    },
    [withRequest, fetchLineupStatus]
  );

  const generateProjections = useCallback(async () => {
    await withRequest(async () => {
      const res = await fetch(`${API_BASE}/api/projections`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(await readApiError(res));
      const list = (await res.json()) as ProjectionResponse[];
      setProjections(list);
      setOwnershipSimsApplied(null);
      await fetchLineupStatus();
    });
  }, [withRequest, fetchLineupStatus]);

  const projectOwnership = useCallback(
    async (nSims: number = 10000): Promise<number | undefined> => {
      const clamped = Math.max(1000, Math.min(nSims, 10000));
      return await withRequest(async () => {
        const resp = await fetch(
          `${API_BASE}/api/ownership?n_sims=${clamped}`,
          { method: "POST" }
        );
        if (!resp.ok) throw new Error(await readApiError(resp));
        const data = (await resp.json()) as ProjectionResponse[];
        setProjections(data);
        setOwnershipSimsApplied(clamped);
        setLineups([]);
        return data.length;
      });
    },
    [withRequest]
  );

  const generateLineups = useCallback(
    async (n: number, maxExposurePct = 70): Promise<boolean> => {
      const ok = await withRequest(async () => {
        const res = await fetch(`${API_BASE}/api/optimize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            n_lineups: n,
            locked_ids: Array.from(lockedIds),
            banned_ids: Array.from(bannedIds),
            max_exposure: maxExposurePct / 100,
          }),
        });
        if (!res.ok) throw new Error(await readApiError(res));
        const raw = (await res.json()) as LineupResponse[];
        const mapped = raw.map((lr) =>
          mapLineupResponse(lr, projMap, rosterMap)
        );
        setLineups(mapped);
        return true;
      });
      return ok === true;
    },
    [withRequest, lockedIds, bannedIds, projMap, rosterMap]
  );

  const exportLineups = useCallback(async () => {
    await withRequest(async () => {
      const res = await fetch(`${API_BASE}/api/export`, { method: "POST" });
      if (!res.ok) throw new Error(await readApiError(res));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "DK_Upload.csv";
      a.click();
      URL.revokeObjectURL(url);
    });
  }, [withRequest]);

  const lockPlayer = useCallback((dk_id: string) => {
    setLockedIds((prev) => {
      const n = new Set(prev);
      n.add(dk_id);
      return n;
    });
    setBannedIds((prev) => {
      const n = new Set(prev);
      n.delete(dk_id);
      return n;
    });
  }, []);

  const unlockPlayer = useCallback((dk_id: string) => {
    setLockedIds((prev) => {
      const n = new Set(prev);
      n.delete(dk_id);
      return n;
    });
  }, []);

  const banPlayer = useCallback((dk_id: string) => {
    setBannedIds((prev) => {
      const n = new Set(prev);
      n.add(dk_id);
      return n;
    });
    setLockedIds((prev) => {
      const n = new Set(prev);
      n.delete(dk_id);
      return n;
    });
  }, []);

  const unbanPlayer = useCallback((dk_id: string) => {
    setBannedIds((prev) => {
      const n = new Set(prev);
      n.delete(dk_id);
      return n;
    });
  }, []);

  const fetchModelInfo = useCallback(async () => {
    await withRequest(async () => {
      const res = await fetch(`${API_BASE}/api/model-info`);
      if (!res.ok) throw new Error(await readApiError(res));
      const j = (await res.json()) as ModelInfoResponse;
      setModelInfo(j);
    });
  }, [withRequest]);

  const fetchHealth = useCallback(async () => {
    await withRequest(async () => {
      const res = await fetch(`${API_BASE}/api/health`);
      if (!res.ok) throw new Error(await readApiError(res));
      await res.json() as HealthResponse;
    });
  }, [withRequest]);

  const value = useMemo<SlateContextValue>(
    () => ({
      players,
      projections,
      lineups,
      modelInfo,
      loading,
      error,
      slateInfo,
      lockedIds,
      bannedIds,
      ownershipSimsApplied,
      playerPool,
      lineupStatus,
      uploadCSV,
      generateProjections,
      projectOwnership,
      generateLineups,
      exportLineups,
      lockPlayer,
      unlockPlayer,
      banPlayer,
      unbanPlayer,
      fetchModelInfo,
      fetchHealth,
      fetchTodaysSlates,
      selectSlateByDg,
    }),
    [
      players,
      projections,
      lineups,
      modelInfo,
      loading,
      error,
      slateInfo,
      lockedIds,
      bannedIds,
      ownershipSimsApplied,
      playerPool,
      lineupStatus,
      uploadCSV,
      generateProjections,
      projectOwnership,
      generateLineups,
      exportLineups,
      lockPlayer,
      unlockPlayer,
      banPlayer,
      unbanPlayer,
      fetchModelInfo,
      fetchHealth,
      fetchTodaysSlates,
      selectSlateByDg,
    ]
  );

  return (
    <SlateContext.Provider value={value}>{children}</SlateContext.Provider>
  );
}

export function useSlate(): SlateContextValue {
  const ctx = useContext(SlateContext);
  if (!ctx) throw new Error("useSlate must be used inside SlateProvider");
  return ctx;
}
