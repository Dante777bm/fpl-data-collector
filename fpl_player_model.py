#!/usr/bin/env python3
"""
fpl_player_model.py (formerly fpl_full_team_assets.py)

Features:
 - Build team attack/defense indices + fixture-adjusted attack index
 - Build player model (season + recent form) with tuned ranking:
     * attackers get extra weight for xGI_per90
     * team fixture-adjusted attack index contributes to attacker scores
     * defenders/goalkeepers are boosted by team defense index
 - Save team_model.csv, player_model.csv
 - Save positional shortlists and combined shortlist
 - Build a sample 15-player squad under budget (default 100.0)
 - Save squad to squad_sample.csv

Usage:
    python fpl_player_model.py
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os

# ---------- Config ----------
BUDGET = 100.0
SQUAD_RULES = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}

# ---------- Helpers ----------
def read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read {path}: {e}")

def safe_div(a, b):
    try:
        return a / b if pd.notnull(a) and pd.notnull(b) and b != 0 else np.nan
    except Exception:
        return np.nan

# ---------- Team Model with Fixture Adjustment ----------
def build_team_model(merged_gws, top50):
    # Basic aggregation
    agg = merged_gws.groupby("Team").agg({
        "Goals": "sum",
        "Assist": "sum",
        "CS": "sum",
        "GC": "sum",
        "xG": "mean",
        "xGC": "mean",
        "GW Points": "sum"
    }).reset_index()

    # Matches
    if "GW" in merged_gws.columns:
        matches = merged_gws.groupby("Team")["GW"].nunique().reset_index().rename(columns={"GW": "Matches"})
    else:
        matches = merged_gws.groupby("Team").size().reset_index().rename(columns={0: "Matches"})

    team_stats = pd.merge(agg, matches, on="Team", how="left")
    team_stats["Goals_per_Match"] = team_stats["Goals"] / team_stats["Matches"]
    team_stats["Assists_per_Match"] = team_stats["Assist"] / team_stats["Matches"]
    team_stats["GC_per_Match"] = team_stats["GC"] / team_stats["Matches"]
    team_stats["Points_per_Match"] = team_stats["GW Points"] / team_stats["Matches"]

    # Merge season totals if present
    if {"Team", "Season_Goals"}.issubset(set(top50.columns)):
        team_summary = top50.groupby("Team").agg({
            "Season_Goals": "sum",
            "Season_Assists": "sum",
            "Total_CS": "sum",
            "Total_Points": "sum",
            "Season_xGI": "sum",
            "Season_xGC": "sum"
        }).reset_index()
        team_model = pd.merge(team_stats, team_summary, on="Team", how="outer")
    else:
        team_model = team_stats.copy()

    # Fill NAs & compute indices
    team_model[["xG", "xGC", "Goals_per_Match", "GC_per_Match"]] = team_model[["xG", "xGC", "Goals_per_Match", "GC_per_Match"]].fillna(0)
    team_model["Attack_Index"] = team_model["Goals_per_Match"] * 0.6 + team_model["xG"] * 0.4
    team_model["Defense_Index"] = (1 / (1 + team_model["GC_per_Match"])) * 0.6 + (1 / (1 + team_model["xGC"])) * 0.4

    # --- Fixture difficulty and adjustment (simple heuristic) ---
    # Build opponent defensive strength map using GC_per_Match
    opp_def_map = team_model.set_index("Team")["GC_per_Match"].to_dict()

    # Try to find upcoming opponents:
    max_gw = merged_gws["GW"].max() if "GW" in merged_gws.columns else None
    upcoming_map = {}

    # 1) future fixtures rows (GW > max_gw) - often empty
    if max_gw is not None:
        future = merged_gws[merged_gws["GW"] > max_gw]
        if not future.empty and "Team" in future.columns and "Opponent Team" in future.columns:
            for t, g in future.groupby("Team"):
                upcoming_map[t] = g["Opponent Team"].tolist()

    # 2) Next Opponent column
    if not upcoming_map and "Next Opponent" in merged_gws.columns and "Team" in merged_gws.columns:
        nxt = merged_gws.dropna(subset=["Next Opponent"]).drop_duplicates(subset=["Team"])
        for _, r in nxt.iterrows():
            upcoming_map[r["Team"]] = [r["Next Opponent"]]

    # 3) fallback: last 3 GW opponents as proxy
    if not upcoming_map and max_gw is not None and "Opponent Team" in merged_gws.columns:
        recent_cutoff = max_gw - 2
        recent_rows = merged_gws[merged_gws["GW"] >= recent_cutoff]
        for t, g in recent_rows.groupby("Team"):
            upcoming_map[t] = g.sort_values("GW", ascending=False)["Opponent Team"].head(3).tolist()

    # Compute avg opponent GC_per_Match upcoming -> higher = easier
    team_upcoming_gc = {}
    for team, opps in upcoming_map.items():
        vals = [opp_def_map.get(o) for o in opps if opp_def_map.get(o) is not None]
        if vals:
            team_upcoming_gc[team] = float(np.mean(vals))

    team_model["Opp_GC_per_Match_Upcoming"] = team_model["Team"].map(team_upcoming_gc)

    # Normalize to 1-5 scale (1 hard - 5 easy)
    if team_model["Opp_GC_per_Match_Upcoming"].notnull().any():
        s = team_model["Opp_GC_per_Match_Upcoming"].dropna()
        mn, mx = s.min(), s.max()
        if mx - mn > 0:
            team_model["Fixture_Difficulty_Score"] = 1 + 4 * (team_model["Opp_GC_per_Match_Upcoming"] - mn) / (mx - mn)
        else:
            team_model["Fixture_Difficulty_Score"] = 3.0
    else:
        team_model["Fixture_Difficulty_Score"] = pd.NA

    # Apply small boost to Attack_Index for easier runs
    # boost factor: 1 + (score - 3) * 0.05  -> score 5 => +10%, score 1 => -10%
    team_model["Attack_Index_Fixture_Adjusted"] = team_model["Attack_Index"] * (1 + (team_model["Fixture_Difficulty_Score"].fillna(3.0) - 3.0) * 0.05)

    return team_model

# ---------- Player Model + Tuned Ranking ----------
def build_player_model(top50, merged_gws, team_model, recent_n=5):
    # Recent form
    if "GW" in merged_gws.columns:
        cutoff = merged_gws["GW"].max() - (recent_n - 1)
        recent = merged_gws[merged_gws["GW"] >= cutoff]
    else:
        recent = merged_gws.copy()

    agg_map = {}
    if "GW Points" in recent.columns:
        agg_map["GW Points"] = "mean"
    if "xGI" in recent.columns:
        agg_map["xGI"] = "mean"
    if "xGC" in recent.columns:
        agg_map["xGC"] = "mean"
    if "Minutes" in recent.columns:
        agg_map["Minutes"] = "mean"
    if "Starts" in recent.columns:
        agg_map["Starts"] = "sum"

    if agg_map:
        recent_form = recent.groupby("Web name").agg(agg_map).reset_index().rename(columns={
            "GW Points": "Recent_Points", "xGI": "Recent_xGI", "xGC": "Recent_xGC", "Minutes": "Recent_Minutes", "Starts": "Recent_Starts"
        })
    else:
        recent_form = recent[["Web name"]].drop_duplicates().reset_index(drop=True)

    # Merge with top50 season summary
    pm = pd.merge(top50, recent_form, on="Web name", how="left")

    # Safe fields and computed metrics
    pm["Cost"] = pm.get("Cost", pd.Series(np.nan))
    pm["Total_Points"] = pm.get("Total_Points", pd.Series(np.nan))
    pm["Total_Minutes"] = pm.get("Total_Minutes", pd.Series(np.nan))
    pm["Season_xGI"] = pm.get("Season_xGI", pd.Series(np.nan))
    pm["Form"] = pm.get("Form", pd.Series(np.nan))
    pm["Team"] = pm.get("Team", pd.Series(np.nan))
    pm["Position"] = pm.get("Position", pd.Series(np.nan))

    pm["Points_per_Million"] = pm.apply(lambda r: safe_div(r.get("Total_Points", np.nan), r.get("Cost", np.nan)), axis=1)
    pm["xGI_per90"] = pm.apply(lambda r: safe_div(r.get("Season_xGI", np.nan), (r.get("Total_Minutes", np.nan) / 90.0) if r.get("Total_Minutes", np.nan) else np.nan), axis=1)

    # Tuned Form Value:
    #   - give more weight to Recent_Points and Form, but also to xGI_per90 for attackers
    # We'll compute a base Form_Value then attacker_boost and team boost
    def compute_base_form_value(row):
        base = 0.0
        if pd.notnull(row.get("Recent_Points")):
            base += 0.6 * row["Recent_Points"]
        if pd.notnull(row.get("Form")):
            base += 0.4 * row["Form"]
        return safe_div(base, row.get("Cost", np.nan))

    pm["Base_Form_Value"] = pm.apply(compute_base_form_value, axis=1)

    # attacker bias: scale xGI_per90 into a small boost (only for MID/FWD)
    # normalize xGI_per90 across players
    xgi = pm["xGI_per90"].fillna(0)
    if xgi.max() > xgi.min():
        xgi_norm = (xgi - xgi.min()) / (xgi.max() - xgi.min())
    else:
        xgi_norm = xgi * 0.0

    pm["xGI_norm"] = xgi_norm

    # map team indices into players
    team_attack_adj = team_model.set_index("Team")["Attack_Index_Fixture_Adjusted"].to_dict()
    team_defense_adj = team_model.set_index("Team")["Defense_Index"].to_dict()

    # Final tuned score:
    # - Attackers (MID/FWD): score = Base_Form_Value + 0.45 * xGI_norm + 0.35 * normalized_team_attack
    # - Defenders/GK: score = Base_Form_Value + 0.5 * normalized_team_defense
    # Normalize team indices
    atk_series = pd.Series(list(team_attack_adj.values())) if team_attack_adj else pd.Series([0.0])
    def_series = pd.Series(list(team_defense_adj.values())) if team_defense_adj else pd.Series([0.0])

    atk_min, atk_max = atk_series.min(), atk_series.max()
    def_min, def_max = def_series.min(), def_series.max()

    def norm_team_attack(t):
        v = team_attack_adj.get(t, np.nan)
        if pd.isna(v): return 0.0
        if atk_max - atk_min == 0: return 0.0
        return (v - atk_min) / (atk_max - atk_min)

    def norm_team_defense(t):
        v = team_defense_adj.get(t, np.nan)
        if pd.isna(v): return 0.0
        if def_max - def_min == 0: return 0.0
        return (v - def_min) / (def_max - def_min)

    pm["Team_Attack_Norm"] = pm["Team"].apply(norm_team_attack)
    pm["Team_Defence_Norm"] = pm["Team"].apply(norm_team_defense)

    def compute_tuned_score(r):
        pos = r.get("Position", "")
        base = r.get("Base_Form_Value", 0.0) if pd.notnull(r.get("Base_Form_Value")) else 0.0
        if pos in ["MID", "FWD"]:
            return base + 0.45 * r.get("xGI_norm", 0.0) + 0.35 * r.get("Team_Attack_Norm", 0.0)
        else:
            return base + 0.50 * r.get("Team_Defence_Norm", 0.0)

    pm["Tuned_Score"] = pm.apply(compute_tuned_score, axis=1)

    # Sort and return
    pm = pm.sort_values(by="Tuned_Score", ascending=False).reset_index(drop=True)
    return pm

# ---------- Extract top assets for target teams ----------
def extract_assets_for_teams(player_model, target_teams, top_n=20):
    df = player_model[player_model["Team"].isin(target_teams)].copy()
    if df.empty:
        return df
    cols = ["Web name", "Team", "Position", "Cost", "Total_Points", "Points_per_Million", "xGI_per90", "Base_Form_Value", "Tuned_Score"]
    cols = [c for c in cols if c in df.columns]
    return df.sort_values(by="Tuned_Score", ascending=False).head(top_n)[cols].reset_index(drop=True)

# ---------- Greedy squad builder ----------
def build_sample_squad(candidates, budget=BUDGET, formation=SQUAD_RULES):
    """
    Greedy approach:
      - For each position, select required number of players in descending Tuned_Score while respecting budget.
      - If over budget, attempt to replace the cheapest selected with next best until under budget or fail.
    This is heuristic and not guaranteed optimal but works fast.
    """
    squad = []
    remaining_budget = budget

    # order positions to fill: GKP, DEF, MID, FWD
    for pos, count in formation.items():
        pos_pool = candidates[candidates["Position"] == pos].sort_values(by="Tuned_Score", ascending=False).reset_index(drop=True)
        # pick top `count` if available
        selected = pos_pool.head(count).copy()
        if len(selected) < count:
            # not enough players for this position; fill with what's available
            selected = selected
        squad.append(selected)

    if not squad:
        return pd.DataFrame()

    squad_df = pd.concat(squad, ignore_index=True)
    # If any costs missing, set high to avoid selection
    squad_df["Cost"] = squad_df["Cost"].fillna(999.0)

    # If initial squad over budget, attempt to adjust
    total_cost = squad_df["Cost"].sum()
    # candidates for swaps: for each position, additional players beyond initial picks
    extras = {}
    for pos in formation.keys():
        extras[pos] = candidates[candidates["Position"] == pos].sort_values(by="Tuned_Score", ascending=False).reset_index(drop=True)

    # If over budget, simple loop: try replace most expensive selected with next best cheaper candidate in same position
    attempts = 0
    while total_cost > budget and attempts < 200:
        # find selected player with highest cost-to-score ratio (cheapness priority)
        squad_df = squad_df.sort_values(by=["Cost", "Tuned_Score"], ascending=[False, True]).reset_index(drop=True)
        replaced = False
        for idx, row in squad_df.iterrows():
            pos = row["Position"]
            # find next candidate not already in squad
            pool = extras.get(pos)
            if pool is None or pool.empty:
                continue
            # find first in pool not already selected
            for i, cand in pool.iterrows():
                if cand["Web name"] in squad_df["Web name"].values:
                    continue
                # if cand is cheaper than current, swap
                if cand["Cost"] < row["Cost"]:
                    squad_df.loc[idx] = cand
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            break
        total_cost = squad_df["Cost"].sum()
        attempts += 1

    # Final check: if still over budget, return empty (couldn't fit)
    if squad_df["Cost"].sum() > budget:
        return pd.DataFrame()

    return squad_df.reset_index(drop=True)

# ---------- Main flow ----------
def get_current_season_folder():
    """Gets the current season's data folder."""
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("FPL_Data_"):
            if "Unknown" not in item:
                return item
    return None

def main():
    current_season_folder = get_current_season_folder()
    if not current_season_folder:
        print("Current season folder not found.", file=sys.stderr)
        sys.exit(1)

    analysis_dir = os.path.join(current_season_folder, "analysis")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # Define input paths
    merged_csv_path = os.path.join(current_season_folder, "merged_gws.csv")
    top50_csv_path = os.path.join(current_season_folder, "top_50_players.csv")

    merged = read_csv(merged_csv_path)
    top50 = read_csv(top50_csv_path)

    # Team model
    team_model = build_team_model(merged, top50)
    team_model.to_csv(os.path.join(analysis_dir, "team_model.csv"), index=False)
    print("Saved team_model.csv")

    # Select top teams
    top_attack = team_model.sort_values(by="Attack_Index_Fixture_Adjusted", ascending=False).head(5)["Team"].tolist()
    top_defense = team_model.sort_values(by="Defense_Index", ascending=False).head(5)["Team"].tolist()
    best_teams = list(dict.fromkeys(top_attack + top_defense))
    print("Top attacking teams (fixture adjusted):", top_attack)
    print("Top defensive teams:", top_defense)
    print("Combined best teams:", best_teams)

    # Player model
    player_model = build_player_model(top50, merged, team_model, recent_n=5)
    player_model.to_csv(os.path.join(analysis_dir, "player_model.csv"), index=False)
    print("Saved player_model.csv")

    # Extract assets from best teams
    assets = extract_assets_for_teams(player_model, best_teams, top_n=200)
    assets.to_csv(os.path.join(analysis_dir, "assets_from_best_teams.csv"), index=False)
    print("Saved assets_from_best_teams.csv")

    # Save positional shortlists
    pos_names = {
        "DEF": "best_team_defenders.csv",
        "MID": "best_team_midfielders.csv",
        "FWD": "best_team_forwards.csv",
        "GKP": "best_team_goalkeepers.csv"
    }
    for pos, fname in pos_names.items():
        subset = assets[assets["Position"] == pos]
        if not subset.empty:
            subset.to_csv(os.path.join(analysis_dir, fname), index=False)
            print(f"Saved {fname}")

    # Combined shortlist
    shortlist_path = os.path.join(analysis_dir, "best_shortlist.csv")
    assets.head(100).to_csv(shortlist_path, index=False)
    print("Saved", os.path.basename(shortlist_path))

    # Build sample squad
    sample_candidates = player_model.copy()
    squad = build_sample_squad(sample_candidates, budget=BUDGET, formation=SQUAD_RULES)
    if squad.empty:
        print("Could not build a squad under the budget with greedy heuristic. Try increasing budget or expanding candidate pool.")
    else:
        squad.to_csv(os.path.join(analysis_dir, "squad_sample.csv"), index=False)
        print("Saved squad_sample.csv (sample 15-player squad under budget)")

    # Print short previews
    print("\n=== Top 10 assets (from combined shortlist) ===")
    print(assets.head(10).to_string(index=False))

    if not squad.empty:
        print("\n=== Sample Squad Preview ===")
        print(squad.to_string(index=False))
        print(f"Total cost: {squad['Cost'].sum():.1f} (budget {BUDGET})")

if __name__ == "__main__":
    main()
