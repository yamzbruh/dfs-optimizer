"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  ReactNode,
} from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

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

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelMetrics {
  q50_rmse: number;
  q50_mae: number;
  q15_rmse: number;
  q85_rmse: number;
  q15_coverage: number;
  q85_coverage: number;
  interval_width: number;
  train_rows: number;
  test_rows: number;
  feature_count: number;
  feature_importance: FeatureImportance[];
}

export interface SlateInfo {
  date: string;
  lock_time: string;
  game_count: number;
  team_count: number;
  player_count: number;
  pitcher_count: number;
  hitter_count: number;
  file_name: string;
  file_hash: string;
}

// ─── Mock Players ─────────────────────────────────────────────────────────────

export const MOCK_PLAYERS: Player[] = [
  // Pitchers
  { dk_id: "p_sale", name: "Chris Sale", team: "ATL", opp: "LAD", dk_position: "SP", position_eligibility: ["P"], salary: 9700, avg_pts: 22.1, proj_pts: 23.5, proj_pts_q15: 12.0, proj_pts_q85: 36.0, ownership: 18.5, leverage: 1.27, status: "confirmed_starting", is_pitcher: true, game_info: "ATL@LAD" },
  { dk_id: "p_glasnow", name: "Tyler Glasnow", team: "LAD", opp: "ATL", dk_position: "SP", position_eligibility: ["P"], salary: 10200, avg_pts: 22.2, proj_pts: 22.8, proj_pts_q15: 10.0, proj_pts_q85: 37.0, ownership: 22.1, leverage: 1.03, status: "confirmed_starting", is_pitcher: true, game_info: "ATL@LAD" },
  { dk_id: "p_roupp", name: "Landen Roupp", team: "SF", opp: "PIT", dk_position: "RP", position_eligibility: ["P"], salary: 8800, avg_pts: 20.1, proj_pts: 21.0, proj_pts_q15: 10.0, proj_pts_q85: 33.5, ownership: 14.8, leverage: 1.42, status: "confirmed_starting", is_pitcher: true, game_info: "PIT@SF" },
  { dk_id: "p_mcdonald", name: "Trevor McDonald", team: "SF", opp: "PIT", dk_position: "SP", position_eligibility: ["P"], salary: 7800, avg_pts: 17.8, proj_pts: 19.2, proj_pts_q15: 8.0, proj_pts_q85: 30.0, ownership: 11.2, leverage: 1.71, status: "confirmed_starting", is_pitcher: true, game_info: "PIT@SF" },
  // Catchers
  { dk_id: "c_rushing", name: "Dalton Rushing", team: "LAD", opp: "ATL", dk_position: "C", position_eligibility: ["C"], salary: 4100, avg_pts: 6.8, proj_pts: 7.2, proj_pts_q15: 1.5, proj_pts_q85: 13.5, ownership: 7.2, leverage: 1.00, status: "confirmed_starting", is_pitcher: false, game_info: "ATL@LAD" },
  { dk_id: "c_susac", name: "Daniel Susac", team: "SF", opp: "PIT", dk_position: "C", position_eligibility: ["C"], salary: 2800, avg_pts: 5.1, proj_pts: 5.8, proj_pts_q15: 0.0, proj_pts_q85: 12.5, ownership: 4.1, leverage: 1.41, status: "confirmed_starting", is_pitcher: false, game_info: "PIT@SF" },
  // 1B
  { dk_id: "1b_olson", name: "Matt Olson", team: "ATL", opp: "LAD", dk_position: "1B", position_eligibility: ["1B"], salary: 5300, avg_pts: 7.4, proj_pts: 8.1, proj_pts_q15: 2.0, proj_pts_q85: 14.5, ownership: 12.8, leverage: 0.63, status: "confirmed_starting", is_pitcher: false, game_info: "ATL@LAD" },
  { dk_id: "1b_sheets", name: "Gavin Sheets", team: "SD", opp: "STL", dk_position: "1B", position_eligibility: ["1B"], salary: 3000, avg_pts: 5.9, proj_pts: 6.3, proj_pts_q15: 1.0, proj_pts_q85: 12.0, ownership: 5.8, leverage: 1.09, status: "confirmed_starting", is_pitcher: false, game_info: "STL@SD" },
  { dk_id: "1b_ohtani", name: "Shohei Ohtani", team: "LAD", opp: "ATL", dk_position: "1B/OF", position_eligibility: ["1B", "OF"], salary: 6400, avg_pts: 8.6, proj_pts: 9.8, proj_pts_q15: 3.0, proj_pts_q85: 18.0, ownership: 28.4, leverage: 0.35, status: "confirmed_starting", is_pitcher: false, game_info: "ATL@LAD" },
  // 2B
  { dk_id: "2b_lowe", name: "Brandon Lowe", team: "PIT", opp: "SF", dk_position: "2B", position_eligibility: ["2B"], salary: 5200, avg_pts: 8.4, proj_pts: 9.1, proj_pts_q15: 2.5, proj_pts_q85: 16.0, ownership: 10.4, leverage: 0.87, status: "confirmed_starting", is_pitcher: false, game_info: "PIT@SF" },
  { dk_id: "2b_cron", name: "Jake Cronenworth", team: "SD", opp: "STL", dk_position: "2B", position_eligibility: ["2B"], salary: 2200, avg_pts: 3.9, proj_pts: 4.5, proj_pts_q15: 0.0, proj_pts_q85: 10.0, ownership: 3.1, leverage: 1.45, status: "projected_starting", is_pitcher: false, game_info: "STL@SD" },
  // 3B
  { dk_id: "3b_riley", name: "Austin Riley", team: "ATL", opp: "LAD", dk_position: "3B", position_eligibility: ["3B"], salary: 4900, avg_pts: 6.8, proj_pts: 7.5, proj_pts_q15: 1.5, proj_pts_q85: 13.5, ownership: 14.6, leverage: 0.51, status: "confirmed_starting", is_pitcher: false, game_info: "ATL@LAD" },
  { dk_id: "3b_urias", name: "Ramon Urias", team: "STL", opp: "SD", dk_position: "3B", position_eligibility: ["3B"], salary: 3100, avg_pts: 3.8, proj_pts: 4.2, proj_pts_q15: 0.0, proj_pts_q85: 9.5, ownership: 3.8, leverage: 1.11, status: "unknown", is_pitcher: false, game_info: "STL@SD" },
  // SS / multi-eligible
  { dk_id: "ss_lawlar", name: "Jordan Lawlar", team: "ARI", opp: "NYM", dk_position: "SS", position_eligibility: ["3B", "SS"], salary: 2400, avg_pts: 5.2, proj_pts: 5.8, proj_pts_q15: 0.5, proj_pts_q85: 11.5, ownership: 4.6, leverage: 1.26, status: "confirmed_starting", is_pitcher: false, game_info: "NYM@ARI" },
  { dk_id: "ss_song", name: "Sung-Mun Song", team: "SD", opp: "STL", dk_position: "SS", position_eligibility: ["SS"], salary: 2400, avg_pts: 4.2, proj_pts: 4.8, proj_pts_q15: 0.0, proj_pts_q85: 10.5, ownership: 3.4, leverage: 1.41, status: "confirmed_starting", is_pitcher: false, game_info: "STL@SD" },
  { dk_id: "ss_dubon", name: "Mauricio Dubon", team: "ATL", opp: "LAD", dk_position: "OF/SS", position_eligibility: ["SS", "OF"], salary: 3000, avg_pts: 6.8, proj_pts: 7.2, proj_pts_q15: 1.5, proj_pts_q85: 13.0, ownership: 5.2, leverage: 1.38, status: "confirmed_starting", is_pitcher: false, game_info: "ATL@LAD" },
  // OF
  { dk_id: "of_soto", name: "Juan Soto", team: "NYM", opp: "ARI", dk_position: "OF", position_eligibility: ["OF"], salary: 6300, avg_pts: 8.4, proj_pts: 9.2, proj_pts_q15: 3.0, proj_pts_q85: 16.5, ownership: 24.1, leverage: 0.38, status: "confirmed_starting", is_pitcher: false, game_info: "NYM@ARI" },
  { dk_id: "of_cruz", name: "Oneil Cruz", team: "PIT", opp: "SF", dk_position: "OF", position_eligibility: ["OF"], salary: 6200, avg_pts: 10.7, proj_pts: 11.4, proj_pts_q15: 4.0, proj_pts_q85: 21.0, ownership: 15.3, leverage: 0.75, status: "confirmed_starting", is_pitcher: false, game_info: "PIT@SF" },
  { dk_id: "of_pages", name: "Andy Pages", team: "LAD", opp: "ATL", dk_position: "OF", position_eligibility: ["OF"], salary: 4700, avg_pts: 7.9, proj_pts: 8.5, proj_pts_q15: 2.0, proj_pts_q85: 15.0, ownership: 9.6, leverage: 0.89, status: "confirmed_starting", is_pitcher: false, game_info: "ATL@LAD" },
  { dk_id: "of_walker", name: "Jordan Walker", team: "STL", opp: "SD", dk_position: "OF", position_eligibility: ["OF"], salary: 5400, avg_pts: 8.6, proj_pts: 9.0, proj_pts_q15: 2.5, proj_pts_q85: 15.5, ownership: 12.1, leverage: 0.74, status: "confirmed_starting", is_pitcher: false, game_info: "STL@SD" },
  { dk_id: "of_rodriguez", name: "Jesus Rodriguez", team: "SF", opp: "PIT", dk_position: "OF", position_eligibility: ["OF"], salary: 2200, avg_pts: 4.8, proj_pts: 5.3, proj_pts_q15: 0.0, proj_pts_q85: 11.0, ownership: 3.2, leverage: 1.66, status: "projected_starting", is_pitcher: false, game_info: "PIT@SF" },
];

// ─── Mock Lineups ─────────────────────────────────────────────────────────────

function findPlayer(dk_id: string): Player {
  return MOCK_PLAYERS.find((p) => p.dk_id === dk_id)!;
}

function buildLineup(
  id: number,
  pairs: [string, string][]
): Lineup {
  const players: LineupPlayer[] = pairs.map(([dk_id, slot]) => ({
    player: findPlayer(dk_id),
    slot,
  }));
  const total_salary = players.reduce((s, lp) => s + lp.player.salary, 0);
  const projected_pts = parseFloat(
    players.reduce((s, lp) => s + lp.player.proj_pts, 0).toFixed(1)
  );
  const leverage_score = parseFloat(
    (players.reduce((s, lp) => s + lp.player.leverage, 0) / players.length).toFixed(2)
  );
  return {
    id,
    players,
    total_salary,
    projected_pts,
    leverage_score,
    portfolio_score: parseFloat((projected_pts + leverage_score * 1.2).toFixed(1)),
    is_valid: true,
  };
}

// Verified salary sums:
// L1: 9700+7800+4100+5300+5200+2400+2400+4700+5400+2200 = 49,200
// L2: 10200+7800+4100+5300+2200+3100+2400+6300+6200+2200 = 49,800
// L3: 9700+8800+2800+3000+5200+2400+2400+6400+6200+2200 = 49,100
// L4: 10200+8800+2800+5300+2200+3100+3000+4700+5400+2200 = 47,700
// L5: 9700+7800+2800+3000+5200+3100+2400+6200+2200+5400 = 47,800

export const MOCK_LINEUPS: Lineup[] = [
  buildLineup(1, [["p_sale","P"],["p_mcdonald","P"],["c_rushing","C"],["1b_olson","1B"],["2b_lowe","2B"],["ss_lawlar","3B"],["ss_song","SS"],["of_pages","OF"],["of_walker","OF"],["of_rodriguez","OF"]]),
  buildLineup(2, [["p_glasnow","P"],["p_mcdonald","P"],["c_rushing","C"],["1b_olson","1B"],["2b_cron","2B"],["3b_urias","3B"],["ss_song","SS"],["of_soto","OF"],["of_cruz","OF"],["of_rodriguez","OF"]]),
  buildLineup(3, [["p_sale","P"],["p_roupp","P"],["c_susac","C"],["1b_sheets","1B"],["2b_lowe","2B"],["ss_lawlar","3B"],["ss_song","SS"],["1b_ohtani","OF"],["of_cruz","OF"],["of_rodriguez","OF"]]),
  buildLineup(4, [["p_glasnow","P"],["p_roupp","P"],["c_susac","C"],["1b_olson","1B"],["2b_cron","2B"],["3b_urias","3B"],["ss_dubon","SS"],["of_pages","OF"],["of_walker","OF"],["of_rodriguez","OF"]]),
  buildLineup(5, [["p_sale","P"],["p_mcdonald","P"],["c_susac","C"],["1b_sheets","1B"],["2b_lowe","2B"],["3b_urias","3B"],["ss_song","SS"],["of_cruz","OF"],["of_rodriguez","OF"],["of_walker","OF"]]),
];

// ─── Mock Slate Info ──────────────────────────────────────────────────────────

export const MOCK_SLATE_INFO: SlateInfo = {
  date: "2026-05-08",
  lock_time: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),
  game_count: 5,
  team_count: 10,
  player_count: 352,
  pitcher_count: 72,
  hitter_count: 280,
  file_name: "DKSalaries_MAY8_26.csv",
  file_hash: "99c742f30a00e4b1f8c92a3d1e56789abcdef",
};

// ─── Mock Model Metrics ───────────────────────────────────────────────────────

export const MOCK_MODEL_METRICS: ModelMetrics = {
  q50_rmse: 2.43,
  q50_mae: 1.87,
  q15_rmse: 2.81,
  q85_rmse: 3.12,
  q15_coverage: 0.271,
  q85_coverage: 0.779,
  interval_width: 12.4,
  train_rows: 186_420,
  test_rows: 14_203,
  feature_count: 18,
  feature_importance: [
    { feature: "exit_velo_30d", importance: 0.142 },
    { feature: "xwoba_30d", importance: 0.138 },
    { feature: "barrel_rate_30d", importance: 0.121 },
    { feature: "hard_hit_30d", importance: 0.108 },
    { feature: "exit_velo_14d", importance: 0.094 },
    { feature: "batting_order_multiplier", importance: 0.089 },
    { feature: "xwoba_14d", importance: 0.082 },
    { feature: "platoon_advantage", importance: 0.071 },
    { feature: "run_diff", importance: 0.063 },
    { feature: "is_high_leverage", importance: 0.057 },
  ],
};

// ─── Context ──────────────────────────────────────────────────────────────────

interface SlateContextValue {
  players: Player[];
  lineups: Lineup[];
  slateInfo: SlateInfo;
  modelMetrics: ModelMetrics;
  lockedIds: string[];
  bannedIds: string[];
  isGenerating: boolean;
  setPlayers: (p: Player[]) => void;
  setLineups: (l: Lineup[]) => void;
  setSlateInfo: (s: SlateInfo) => void;
  addLock: (id: string) => void;
  removeLock: (id: string) => void;
  addBan: (id: string) => void;
  removeBan: (id: string) => void;
  setIsGenerating: (v: boolean) => void;
}

const SlateContext = createContext<SlateContextValue | null>(null);

export function SlateProvider({ children }: { children: ReactNode }) {
  const [players, setPlayers] = useState<Player[]>(MOCK_PLAYERS);
  const [lineups, setLineups] = useState<Lineup[]>(MOCK_LINEUPS);
  const [slateInfo, setSlateInfo] = useState<SlateInfo>(MOCK_SLATE_INFO);
  const [modelMetrics] = useState<ModelMetrics>(MOCK_MODEL_METRICS);
  const [lockedIds, setLockedIds] = useState<string[]>([]);
  const [bannedIds, setBannedIds] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const addLock = useCallback(
    (id: string) => setLockedIds((prev) => (prev.includes(id) ? prev : [...prev, id])),
    []
  );
  const removeLock = useCallback(
    (id: string) => setLockedIds((prev) => prev.filter((x) => x !== id)),
    []
  );
  const addBan = useCallback(
    (id: string) => setBannedIds((prev) => (prev.includes(id) ? prev : [...prev, id])),
    []
  );
  const removeBan = useCallback(
    (id: string) => setBannedIds((prev) => prev.filter((x) => x !== id)),
    []
  );

  return (
    <SlateContext.Provider
      value={{
        players,
        lineups,
        slateInfo,
        modelMetrics,
        lockedIds,
        bannedIds,
        isGenerating,
        setPlayers,
        setLineups,
        setSlateInfo,
        addLock,
        removeLock,
        addBan,
        removeBan,
        setIsGenerating,
      }}
    >
      {children}
    </SlateContext.Provider>
  );
}

export function useSlate(): SlateContextValue {
  const ctx = useContext(SlateContext);
  if (!ctx) throw new Error("useSlate must be used inside SlateProvider");
  return ctx;
}
