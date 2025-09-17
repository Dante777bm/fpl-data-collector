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
    """Main function to run the player model."""
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

    # === Step 1: Aggregate recent form (last 5 GWs) ===
    print("Aggregating recent form...")
    recent_gws = merged_gws[merged_gws['GW'] >= merged_gws['GW'].max() - 4]
    recent_form = (
        recent_gws.groupby("Web name")
        .agg({
            "GW Points": "mean",   # avg points in last 5 GWs
            "xGI": "mean",         # expected attacking involvement
            "xGC": "mean",         # expected goals conceded
            "Minutes": "mean",     # avg minutes
            "Starts": "sum",       # total starts
        })
        .reset_index()
        .rename(columns={"GW Points": "Recent_Points", "xGI": "Recent_xGI", "xGC": "Recent_xGC"})
    )

    # === Step 2: Merge with Top 50 season summary ===
    print("Merging with top 50 players...")
    player_model = pd.merge(top_50_players, recent_form, on="Web name", how="left")

    # === Step 3: Create value metrics ===
    print("Creating value metrics...")
    player_model["Points_per_Million"] = player_model["Total_Points"] / player_model["Cost"]
    player_model["xGI_per90"] = player_model["Season_xGI"] / (player_model["Total_Minutes"] / 90)
    player_model["Form_Value"] = (player_model["Recent_Points"] * 0.7 + player_model["Form"] * 0.3) / player_model["Cost"]

    # === Step 4: Add fixture difficulty adjustment ===
    print("Adding fixture difficulty adjustment...")
    fixture_difficulty = (
        merged_gws.groupby("Opponent Team")
        .agg({
            "Team H Score": "mean",
            "Team A Score": "mean"
        })
        .reset_index()
    )
    fixture_difficulty["Opponent_Strength"] = (fixture_difficulty["Team H Score"] + fixture_difficulty["Team A Score"]) / 2

    team_strength_map = dict(zip(fixture_difficulty["Opponent Team"], fixture_difficulty["Opponent_Strength"]))
    merged_gws["Fixture_Difficulty"] = merged_gws["Opponent Team"].map(team_strength_map)

    fixture_adjustment = (
        merged_gws.groupby("Web name")["Fixture_Difficulty"].mean().reset_index()
    )

    player_model = pd.merge(player_model, fixture_adjustment, on="Web name", how="left")

    player_model["Fixture_Adjusted_Value"] = player_model["Form_Value"] * (1 + (5 - player_model["Fixture_Difficulty"]) * 0.05)

    # === Step 5: Best players function ===
    def best_players(position=None, top_n=15, adjusted=True):
        df = player_model.copy()
        if position:
            df = df[df["Position"] == position]
        sort_col = "Fixture_Adjusted_Value" if adjusted else "Form_Value"
        return df.sort_values(by=sort_col, ascending=False).head(top_n)[[
            "Web name", "Team", "Position", "Cost", "Total_Points", "Points_per_Million", "xGI_per90", sort_col
        ]]

    # === Step 6: Save datasets ===
    print("Saving datasets...")
    player_model.to_csv(os.path.join(output_dir, "player_model.csv"), index=False)

    midfielders = best_players("MID", top_n=10)
    forwards = best_players("FWD", top_n=10)
    defenders = best_players("DEF", top_n=10)
    goalkeepers = best_players("GKP", top_n=5)

    midfielders.to_csv(os.path.join(output_dir, "best_midfielders.csv"), index=False)
    forwards.to_csv(os.path.join(output_dir, "best_forwards.csv"), index=False)
    defenders.to_csv(os.path.join(output_dir, "best_defenders.csv"), index=False)
    goalkeepers.to_csv(os.path.join(output_dir, "best_goalkeepers.csv"), index=False)

    # Combined shortlist
    shortlist = pd.concat([midfielders, forwards, defenders, goalkeepers], ignore_index=True)
    shortlist.to_csv(os.path.join(output_dir, "best_shortlist.csv"), index=False)

    # === Step 7: Example Outputs ===
    print("\nTop 15 Overall (Fixture Adjusted):")
    print(best_players())

    print("\nTop 10 Midfielders (Fixture Adjusted):")
    print(midfielders)

    print("\nTop 10 Forwards (Fixture Adjusted):")
    print(forwards)

    print("\nTop 10 Defenders (Fixture Adjusted):")
    print(defenders)

    print("\nTop 5 Goalkeepers (Fixture Adjusted):")
    print(goalkeepers)

    print(f"\nMaster dataset saved as {os.path.join(output_dir, 'player_model.csv')}")
    print(f"Best positional picks saved as {os.path.join(output_dir, 'best_midfielders.csv')}, etc.")
    print(f"Combined shortlist saved as {os.path.join(output_dir, 'best_shortlist.csv')}")


if __name__ == "__main__":
    main()
