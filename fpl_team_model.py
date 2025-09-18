import pandas as pd
import os

def get_current_season_folder():
    """Gets the current season's data folder."""
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("FPL_Data_"):
            if "Unknown" not in item:
                return item
    return None

def main():
    """Main function to run the team model."""
    current_season_folder = get_current_season_folder()

    if not current_season_folder:
        print("Current season folder not found.")
        return

    # Define input paths
    merged_gws_path = os.path.join(current_season_folder, "merged_gws.csv")
    top_50_players_path = os.path.join(current_season_folder, "top_50_players.csv")

    # Check if input files exist
    if not os.path.exists(merged_gws_path) or not os.path.exists(top_50_players_path):
        print(f"Input files not found in {current_season_folder}")
        return

    # Define output directory
    output_dir = os.path.join(current_season_folder, "analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === Load Data ===
    print("Loading data...")
    merged_gws = pd.read_csv(merged_gws_path)
    top_50_players = pd.read_csv(top_50_players_path)

    # === Step 1: Aggregate team attacking & defensive performance from merged_gws ===
    print("Aggregating team performance...")
    team_stats = (
        merged_gws.groupby("Team")
        .agg({
            "Goals": "sum",
            "Assist": "sum",
            "CS": "sum",
            "GC": "sum",
            "xG": "mean",
            "xGC": "mean",
            "Minutes": "sum",
            "GW Points": "sum"
        })
        .reset_index()
    )

    # Per match averages
    team_matches = merged_gws.groupby("Team")["GW"].nunique().reset_index().rename(columns={"GW": "Matches"})
    team_stats = pd.merge(team_stats, team_matches, on="Team", how="left")

    team_stats["Goals_per_Match"] = team_stats["Goals"] / team_stats["Matches"]
    team_stats["Assists_per_Match"] = team_stats["Assist"] / team_stats["Matches"]
    team_stats["GC_per_Match"] = team_stats["GC"] / team_stats["Matches"]
    team_stats["Points_per_Match"] = team_stats["GW Points"] / team_stats["Matches"]

    # === Step 2: Add season totals from top_50_players (team perspective) ===
    print("Adding season totals...")
    team_summary = (
        top_50_players.groupby("Team")
        .agg({
            "Season_Goals": "sum",
            "Season_Assists": "sum",
            "Total_CS": "sum",
            "Total_Points": "sum",
            "Season_xGI": "sum",
            "Season_xGC": "sum"
        })
        .reset_index()
    )

    # Merge with team stats
    total_team_model = pd.merge(team_stats, team_summary, on="Team", how="outer")

    # === Step 3: Compute Team Strength Indices ===
    print("Computing team strength indices...")
    # Attack Index combines observed goals per match and xG (weighted)
    total_team_model["Attack_Index"] = total_team_model["Goals_per_Match"] * 0.6 + total_team_model["xG"] * 0.4
    # Defense Index inverts goals conceded and xGC (higher is better)
    total_team_model["Defense_Index"] = (1 / (1 + total_team_model["GC_per_Match"])) * 0.6 + (1 / (1 + total_team_model["xGC"])) * 0.4

    # === Step 4: Fixture difficulty adjustment (team level) ===
    print("Adjusting for fixture difficulty...")

    # 4a: compute opponent defensive strength map
    opp_def_strength = total_team_model.set_index("Team")["GC_per_Match"].to_dict()

    # 4b: helper to get upcoming opponents
    max_gw = merged_gws["GW"].max() if "GW" in merged_gws.columns else None
    upcoming_map = {}

    if max_gw is not None:
        # Fallback: use the last 3 GWs opponents as a proxy for upcoming fixture difficulty
        recent_cutoff = max_gw - 2
        if recent_cutoff > 0:
            recent_rows = merged_gws[merged_gws["GW"] >= recent_cutoff]
            if "Team" in recent_rows.columns and "Opponent Team" in recent_rows.columns:
                for team, group in recent_rows.groupby("Team"):
                    opps = group.sort_values("GW", ascending=False)["Opponent Team"].head(3).tolist()
                    upcoming_map[team] = opps

    # 4c: compute average opponent defensive weakness per team
    team_fixture_difficulty = {}
    for team, opps in upcoming_map.items():
        vals = [opp_def_strength.get(opp) for opp in opps if opp_def_strength.get(opp) is not None and not pd.isna(opp_def_strength.get(opp))]
        if vals:
            team_fixture_difficulty[team] = sum(vals) / len(vals)

    total_team_model["Opp_GC_per_Match_Upcoming"] = total_team_model["Team"].map(team_fixture_difficulty)

    # 4d: Normalize difficulty and apply boost
    if total_team_model["Opp_GC_per_Match_Upcoming"].notnull().any():
        s = total_team_model["Opp_GC_per_Match_Upcoming"].dropna()
        minv, maxv = s.min(), s.max()
        if maxv - minv > 0:
            total_team_model["Fixture_Difficulty_Score"] = 1 + 4 * (total_team_model["Opp_GC_per_Match_Upcoming"] - minv) / (maxv - minv)
        else:
            total_team_model["Fixture_Difficulty_Score"] = 3.0

        total_team_model["Attack_Index_Fixture_Adjusted"] = total_team_model["Attack_Index"] * (1 + (total_team_model["Fixture_Difficulty_Score"].fillna(3.0) - 3.0) * 0.05)
    else:
        total_team_model["Fixture_Difficulty_Score"] = pd.NA
        total_team_model["Attack_Index_Fixture_Adjusted"] = total_team_model["Attack_Index"]


    # === Step 5: Save team dataset ===
    print("Saving team dataset...")
    output_path = os.path.join(output_dir, "team_model.csv")
    total_team_model.to_csv(output_path, index=False)

    # === Step 6: Example Outputs ===
    print("\nTop 5 Attacking Teams (by Attack_Index, fixture adjusted):")
    print(total_team_model.sort_values(by="Attack_Index_Fixture_Adjusted", ascending=False).head(5)[["Team", "Goals_per_Match", "xG", "Attack_Index_Fixture_Adjusted"]])

    print("\nTop 5 Defensive Teams (by Defense_Index):")
    print(total_team_model.sort_values(by="Defense_Index", ascending=False).head(5)[["Team", "GC_per_Match", "xGC", "Defense_Index"]])

    print(f"\nTeam dataset saved as {output_path}")

if __name__ == "__main__":
    main()
