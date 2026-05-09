-- ============================================================
-- DFS Optimizer — Supabase PostgreSQL Schema
-- ============================================================
-- Single migration file. Run in the Supabase SQL editor.
--
-- Conventions:
--   * All primary keys are uuid via uuid_generate_v4()
--   * Every table has created_at timestamptz DEFAULT now()
--   * RLS enabled on every table
--   * Backend (service_role) has full access on every table
--   * Frontend (anon) has SELECT only on a curated read-safe set
--   * No anon writes anywhere; sensitive tables have no anon access at all
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- 1. slates
-- ============================================================
CREATE TABLE slates (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    dk_slate_id text UNIQUE NOT NULL,
    date date NOT NULL,
    slate_type text NOT NULL CHECK (slate_type IN ('main', 'afternoon', 'showdown', 'turbo')),
    lock_time timestamptz,
    status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'locked', 'complete')),
    created_at timestamptz DEFAULT now()
);

-- ============================================================
-- 2. teams
-- ============================================================
CREATE TABLE teams (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    mlbam_team_id integer UNIQUE NOT NULL,
    name text NOT NULL,
    abbreviation text NOT NULL,
    league text CHECK (league IN ('AL', 'NL')),
    division text,
    created_at timestamptz DEFAULT now()
);

-- ============================================================
-- 3. stadiums
-- ============================================================
CREATE TABLE stadiums (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id uuid REFERENCES teams(id),
    name text NOT NULL,
    lat numeric NOT NULL,
    lng numeric NOT NULL,
    home_plate_orientation_deg integer,
    roof_status text NOT NULL DEFAULT 'open' CHECK (roof_status IN ('open', 'retractable', 'fixed')),
    wall_height_lf integer,
    wall_height_cf integer,
    wall_height_rf integer,
    park_factor_runs numeric DEFAULT 100,
    park_factor_hr numeric DEFAULT 100,
    created_at timestamptz DEFAULT now()
);

-- ============================================================
-- 4. players
-- ============================================================
CREATE TABLE players (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    mlbam_id integer UNIQUE NOT NULL,
    full_name text NOT NULL,
    bats text CHECK (bats IN ('L', 'R', 'S')),
    throws text CHECK (throws IN ('L', 'R')),
    primary_position text NOT NULL,
    team_id uuid REFERENCES teams(id),
    active boolean DEFAULT true,
    created_at timestamptz DEFAULT now()
);

-- ============================================================
-- 5. player_ids
-- ============================================================
CREATE TABLE player_ids (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id uuid UNIQUE REFERENCES players(id),
    dk_name text,
    mlbam_id integer,
    fangraphs_id text,
    baseball_savant_id integer,
    bref_id text,
    created_at timestamptz DEFAULT now()
);

-- ============================================================
-- 6. games
-- ============================================================
CREATE TABLE games (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid NOT NULL REFERENCES slates(id),
    mlbam_game_id integer UNIQUE NOT NULL,
    home_team_id uuid REFERENCES teams(id),
    away_team_id uuid REFERENCES teams(id),
    stadium_id uuid REFERENCES stadiums(id),
    game_time timestamptz,
    status text DEFAULT 'scheduled',
    weather_wind_speed numeric,
    weather_wind_dir integer,
    weather_temp numeric,
    wind_vector_result text CHECK (wind_vector_result IN ('blowing_out', 'blowing_in', 'crosswind', 'neutral', 'dome')),
    implied_total_home numeric,
    implied_total_away numeric,
    over_under numeric,
    line_delta_home numeric,
    line_delta_away numeric,
    umpire_id text,
    created_at timestamptz DEFAULT now()
);

-- ============================================================
-- 7. salaries
-- ============================================================
CREATE TABLE salaries (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid NOT NULL REFERENCES slates(id),
    player_id uuid NOT NULL REFERENCES players(id),
    game_id uuid REFERENCES games(id),
    dk_player_id text,
    salary integer NOT NULL CHECK (salary >= 2000 AND salary <= 15000),
    position_eligibility text[] NOT NULL,
    batting_order integer CHECK (batting_order >= 1 AND batting_order <= 9),
    batting_order_multiplier numeric DEFAULT 1.0,
    lineup_status text NOT NULL DEFAULT 'unknown' CHECK (lineup_status IN ('confirmed_starting', 'projected_starting', 'bench', 'scratched', 'unknown')),
    is_home boolean,
    last_updated timestamptz DEFAULT now(),
    created_at timestamptz DEFAULT now(),
    UNIQUE (slate_id, player_id)
);

-- ============================================================
-- 8. raw_uploads
-- ============================================================
CREATE TABLE raw_uploads (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid REFERENCES slates(id),
    upload_type text NOT NULL CHECK (upload_type IN ('dk_salary', 'dk_results', 'ownership')),
    filename text NOT NULL,
    file_hash text NOT NULL,
    storage_path text,
    uploaded_at timestamptz DEFAULT now(),
    UNIQUE (file_hash)
);

-- ============================================================
-- 9. validation_errors
-- ============================================================
CREATE TABLE validation_errors (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid REFERENCES slates(id),
    entity_type text NOT NULL CHECK (entity_type IN ('player', 'salary', 'feature', 'lineup', 'projection', 'upload')),
    entity_id text,
    severity text NOT NULL CHECK (severity IN ('warning', 'error', 'critical')),
    message text NOT NULL,
    raw_payload jsonb,
    resolved boolean DEFAULT false,
    created_at timestamptz DEFAULT now()
);

-- ============================================================
-- 10. features
-- ============================================================
CREATE TABLE features (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid NOT NULL REFERENCES slates(id),
    player_id uuid NOT NULL REFERENCES players(id),
    feature_vector jsonb NOT NULL,
    created_at timestamptz DEFAULT now(),
    UNIQUE (slate_id, player_id)
);

-- ============================================================
-- 11. model_runs
-- ============================================================
CREATE TABLE model_runs (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type text NOT NULL CHECK (model_type IN ('points_q15', 'points_q50', 'points_q85', 'ownership_proxy', 'ownership_real')),
    model_version text NOT NULL,
    trained_at timestamptz DEFAULT now(),
    training_slate_count integer,
    rmse numeric,
    mae numeric,
    q15_coverage numeric,
    q85_coverage numeric,
    interval_width numeric,
    hyperparameters jsonb,
    feature_importance jsonb,
    notes text,
    created_at timestamptz DEFAULT now()
);

-- ============================================================
-- 12. projections
-- ============================================================
CREATE TABLE projections (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid NOT NULL REFERENCES slates(id),
    player_id uuid NOT NULL REFERENCES players(id),
    model_run_id uuid NOT NULL REFERENCES model_runs(id),
    pts_q15 numeric,
    pts_q50 numeric,
    pts_q85 numeric,
    created_at timestamptz DEFAULT now(),
    UNIQUE (slate_id, player_id, model_run_id)
);

-- ============================================================
-- 13. ownership_projections
-- ============================================================
CREATE TABLE ownership_projections (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid NOT NULL REFERENCES slates(id),
    player_id uuid NOT NULL REFERENCES players(id),
    model_run_id uuid NOT NULL REFERENCES model_runs(id),
    ownership_raw numeric,
    ownership_normalized numeric,
    leverage_score numeric,
    phase text NOT NULL DEFAULT 'proxy' CHECK (phase IN ('proxy', 'real')),
    created_at timestamptz DEFAULT now(),
    UNIQUE (slate_id, player_id, model_run_id)
);

-- ============================================================
-- 14. player_constraints
-- ============================================================
CREATE TABLE player_constraints (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid NOT NULL REFERENCES slates(id),
    player_id uuid NOT NULL REFERENCES players(id),
    is_locked boolean DEFAULT false,
    is_banned boolean DEFAULT false,
    min_exposure numeric DEFAULT 0.0 CHECK (min_exposure >= 0 AND min_exposure <= 1),
    max_exposure numeric DEFAULT 0.70 CHECK (max_exposure >= 0 AND max_exposure <= 1),
    notes text,
    created_at timestamptz DEFAULT now(),
    UNIQUE (slate_id, player_id)
);

-- ============================================================
-- 15. lineups
-- ============================================================
CREATE TABLE lineups (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid NOT NULL REFERENCES slates(id),
    model_run_id uuid REFERENCES model_runs(id),
    lineup_number integer NOT NULL CHECK (lineup_number >= 1 AND lineup_number <= 20),
    total_salary integer CHECK (total_salary >= 0 AND total_salary <= 50000),
    projected_pts_q50 numeric,
    projected_ownership numeric,
    leverage_score numeric,
    simulated_win_rate numeric,
    portfolio_score numeric,
    is_valid boolean DEFAULT false,
    exported_at timestamptz,
    created_at timestamptz DEFAULT now(),
    UNIQUE (slate_id, lineup_number, model_run_id)
);

-- ============================================================
-- 16. lineup_players
-- ============================================================
CREATE TABLE lineup_players (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    lineup_id uuid NOT NULL REFERENCES lineups(id),
    player_id uuid NOT NULL REFERENCES players(id),
    roster_position text NOT NULL CHECK (roster_position IN ('SP', 'C', '1B', '2B', '3B', 'SS', 'OF', 'UTIL')),
    salary integer,
    pts_q50 numeric,
    ownership_normalized numeric,
    created_at timestamptz DEFAULT now()
);

-- ============================================================
-- 17. actuals
-- ============================================================
CREATE TABLE actuals (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid NOT NULL REFERENCES slates(id),
    player_id uuid NOT NULL REFERENCES players(id),
    dk_points_actual numeric,
    batting_order_actual integer,
    game_played boolean DEFAULT true,
    collected_at timestamptz DEFAULT now(),
    UNIQUE (slate_id, player_id)
);

-- ============================================================
-- 18. contest_ownership
-- ============================================================
CREATE TABLE contest_ownership (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_id uuid NOT NULL REFERENCES slates(id),
    player_id uuid NOT NULL REFERENCES players(id),
    contest_name text,
    field_size integer,
    entry_fee numeric,
    ownership_pct numeric CHECK (ownership_pct >= 0 AND ownership_pct <= 100),
    collected_at timestamptz DEFAULT now()
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX idx_slates_date ON slates(date);
CREATE INDEX idx_slates_status ON slates(status);

CREATE INDEX idx_games_slate_id ON games(slate_id);
CREATE INDEX idx_games_home_team_id ON games(home_team_id);
CREATE INDEX idx_games_away_team_id ON games(away_team_id);

CREATE INDEX idx_salaries_slate_id ON salaries(slate_id);
CREATE INDEX idx_salaries_player_id ON salaries(player_id);
CREATE INDEX idx_salaries_lineup_status ON salaries(lineup_status);
CREATE INDEX idx_salaries_salary ON salaries(salary);

CREATE INDEX idx_projections_slate_id ON projections(slate_id);
CREATE INDEX idx_projections_player_id ON projections(player_id);
CREATE INDEX idx_projections_model_run_id ON projections(model_run_id);

CREATE INDEX idx_ownership_projections_slate_id ON ownership_projections(slate_id);
CREATE INDEX idx_ownership_projections_leverage_score ON ownership_projections(leverage_score DESC);

CREATE INDEX idx_lineups_slate_id ON lineups(slate_id);
CREATE INDEX idx_lineups_portfolio_score ON lineups(portfolio_score DESC);

CREATE INDEX idx_lineup_players_lineup_id ON lineup_players(lineup_id);
CREATE INDEX idx_lineup_players_player_id ON lineup_players(player_id);

CREATE INDEX idx_actuals_slate_id ON actuals(slate_id);
CREATE INDEX idx_actuals_player_id ON actuals(player_id);

CREATE INDEX idx_validation_errors_slate_id ON validation_errors(slate_id);
CREATE INDEX idx_validation_errors_severity ON validation_errors(severity);
CREATE INDEX idx_validation_errors_resolved ON validation_errors(resolved);

-- ============================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================
-- Enable RLS on every table.
ALTER TABLE slates                ENABLE ROW LEVEL SECURITY;
ALTER TABLE teams                 ENABLE ROW LEVEL SECURITY;
ALTER TABLE stadiums              ENABLE ROW LEVEL SECURITY;
ALTER TABLE players               ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_ids            ENABLE ROW LEVEL SECURITY;
ALTER TABLE games                 ENABLE ROW LEVEL SECURITY;
ALTER TABLE salaries              ENABLE ROW LEVEL SECURITY;
ALTER TABLE raw_uploads           ENABLE ROW LEVEL SECURITY;
ALTER TABLE validation_errors     ENABLE ROW LEVEL SECURITY;
ALTER TABLE features              ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_runs            ENABLE ROW LEVEL SECURITY;
ALTER TABLE projections           ENABLE ROW LEVEL SECURITY;
ALTER TABLE ownership_projections ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_constraints    ENABLE ROW LEVEL SECURITY;
ALTER TABLE lineups               ENABLE ROW LEVEL SECURITY;
ALTER TABLE lineup_players        ENABLE ROW LEVEL SECURITY;
ALTER TABLE actuals               ENABLE ROW LEVEL SECURITY;
ALTER TABLE contest_ownership     ENABLE ROW LEVEL SECURITY;

-- ------------------------------------------------------------
-- service_role: full access on every table.
-- (service_role bypasses RLS by default in Supabase, but we
-- create explicit permissive policies so behavior is unambiguous
-- and survives any future changes to bypass settings.)
-- ------------------------------------------------------------
CREATE POLICY service_role_all_slates                ON slates                FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_teams                 ON teams                 FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_stadiums              ON stadiums              FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_players               ON players               FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_player_ids            ON player_ids            FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_games                 ON games                 FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_salaries              ON salaries              FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_raw_uploads           ON raw_uploads           FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_validation_errors     ON validation_errors     FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_features              ON features              FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_model_runs            ON model_runs            FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_projections           ON projections           FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_ownership_projections ON ownership_projections FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_player_constraints    ON player_constraints    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_lineups               ON lineups               FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_lineup_players        ON lineup_players        FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_actuals               ON actuals               FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all_contest_ownership     ON contest_ownership     FOR ALL TO service_role USING (true) WITH CHECK (true);

-- ------------------------------------------------------------
-- anon: SELECT only on the read-safe set.
-- No anon write access on any table. No anon access at all on:
--   player_ids, raw_uploads, validation_errors, features,
--   model_runs, player_constraints, actuals, contest_ownership
-- ------------------------------------------------------------
CREATE POLICY anon_select_slates                ON slates                FOR SELECT TO anon USING (true);
CREATE POLICY anon_select_teams                 ON teams                 FOR SELECT TO anon USING (true);
CREATE POLICY anon_select_stadiums              ON stadiums              FOR SELECT TO anon USING (true);
CREATE POLICY anon_select_players               ON players               FOR SELECT TO anon USING (true);
CREATE POLICY anon_select_games                 ON games                 FOR SELECT TO anon USING (true);
CREATE POLICY anon_select_salaries              ON salaries              FOR SELECT TO anon USING (true);
CREATE POLICY anon_select_projections           ON projections           FOR SELECT TO anon USING (true);
CREATE POLICY anon_select_ownership_projections ON ownership_projections FOR SELECT TO anon USING (true);
CREATE POLICY anon_select_lineups               ON lineups               FOR SELECT TO anon USING (true);
CREATE POLICY anon_select_lineup_players        ON lineup_players        FOR SELECT TO anon USING (true);

-- ============================================================
-- END OF SCHEMA
-- ============================================================
