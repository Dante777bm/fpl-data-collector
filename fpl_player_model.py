import os
import sys
import pandas as pd


# ------------------------------
# Configuration
# ------------------------------
LAST_N_GWS = 5  # how many recent gameweeks to use
OUTPUT_DIR_NAME = "analysis"


# ------------------------------
# Utilities
# ------------------------------
def get_current_season_folder() -> str | None:
    """
    Returns the most recent folder matching FPL_Data_* (excluding 'Unknown').
    """
    candidates = [
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith("FPL_Data_") and "Unknown" not in d
    ]
    return max(candidates, default=None)


def check_required_columns(df: pd.DataFrame, required: set, df_name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {df_name}: {missing}")


def safe_div(numerator, denominator):
    try:
        return numerator / denominator if denominator else 0
    except ZeroDivisionError:
        return 0


# ------------------------------
# Core Processing
# ------------------------------
def build_player_model(merged_gws: pd.DataFrame, top_50_players: pd.DataFrame) -> pd.DataFrame:
    # --- Validate columns ---
    required_gws = {"Web name", "GW", "GW Points", "xGI", "xGC", "Minutes", "Starts",
                    "Opponent Team", "Team H Score", "Team A Score"}
    required_top = {"Web name", "Total_Points", "Cost", "Season_xGI",
                    "Total_Minutes", "Form", "Position", "Team"}

    check_required_columns(merged_gws, required_gws, "merged_gws")
    check_required_columns(top_50_players, required_top, "top_50_players")

    # --- Aggregate recent form ---
    latest_gw = merged_gws["GW"].max()
    recent_gws = merged_gws[merged_gws["GW"] >= latest_gw - (LAST_N_GWS - 1)]
    recent_form = (
        recent_gws.groupby("Web name")
        .agg({
            "GW Points": "mean",
            "xGI": "mean",
            "xGC": "mean",
            "Minutes": "mean",
            "Starts": "sum",
        })
        .reset_index()
        .rename(columns={"GW Points": "Recent_Points", "xGI": "Recent_xGI", "xGC": "Recent_xGC"})
    )

    # --- Merge ---
    player_model = pd.merge(top_50_players, recent_form, on="Web name", how="left")

    # --- Value metrics ---
    player_model["Points_per_Million"] = player_model.apply(
        lambda r: safe_div(r["Total_Points"], r["Cost"]), axis=1
    )
    player_model["xGI_per90"] = player_model.apply(
        lambda r: safe_div(r["Season_xGI"], r["Total_Minutes"] / 90) if r["Total_Minutes"] else 0,
        axis=1,
    )
    player_model["Form_Value"] = player_model.apply(
        lambda r: safe_div(r.get("Recent_Points", 0) * 0.7 + r.get("Form", 0) * 0.3, r["Cost"]),
        axis=1,
    )

    # --- Fixture difficulty ---
    fixture_difficulty = (
        merged_gws.groupby("Opponent Team")[["Team H Score", "Team A Score"]]
        .mean()
        .reset_index()
    )
    fixture_difficulty["Opponent_Strength"] = (
        fixture_difficulty["Team H Score"] + fixture_difficulty["Team A Score"]
    ) / 2

    team_strength_map = dict(zip(fixture_difficulty["Opponent Team"],
                                 fixture_difficulty["Opponent_Strength"]))

    merged_gws["Fixture_Difficulty"] = merged_gws["Opponent Team"].map(team_strength_map)
    fixture_adjustment = (
        merged_gws.groupby("Web name")["Fixture_Difficulty"].mean().reset_index()
    )

    player_model = pd.merge(player_model, fixture_adjustment, on="Web name", how="left")

    # Normalize fixture difficulty to ~1–5 scale if needed
    if player_model["Fixture_Difficulty"].max() > 6:
        max_val = player_model["Fixture_Difficulty"].max()
        player_model["Fixture_Difficulty"] = 5 * player_model["Fixture_Difficulty"] / max_val

    player_model["Fixture_Adjusted_Value"] = player_model.apply(
        lambda r: r["Form_Value"] * (1 + (5 - r.get("Fixture_Difficulty", 5)) * 0.05),
        axis=1,
    )
    return player_model


def best_players(player_model: pd.DataFrame, position=None, top_n=15, adjusted=True) -> pd.DataFrame:
    df = player_model.copy()
    if position:
        df = df[df["Position"] == position]
    sort_col = "Fixture_Adjusted_Value" if adjusted else "Form_Value"
    return df.sort_values(by=sort_col, ascending=False).head(top_n)[[
        "Web name", "Team", "Position", "Cost", "Total_Points",
        "Points_per_Million", "xGI_per90", sort_col
    ]]


# ------------------------------
# Main
# ------------------------------
def main():
    try:
        season_folder = get_current_season_folder()
        if not season_folder:
            print("❌ No valid season folder found.")
            return

        merged_gws_path = os.path.join(season_folder, "merged_gws.csv")
        top_50_path = os.path.join(season_folder, "top_50_players.csv")

        if not (os.path.exists(merged_gws_path) and os.path.exists(top_50_path)):
            print(f"❌ Required CSVs not found in {season_folder}.")
            return

        output_dir = os.path.join(season_folder, OUTPUT_DIR_NAME)
        os.makedirs(output_dir, exist_ok=True)

        print("📥 Loading data...")
        merged_gws = pd.read_csv(merged_gws_path)
        top_50 = pd.read_csv(top_50_path)

        print("⚙️ Building player model...")
        model = build_player_model(merged_gws, top_50)

        print("💾 Saving datasets...")
        model.to_csv(os.path.join(output_dir, "player_model.csv"), index=False)

        positions = {"MID": 10, "FWD": 10, "DEF": 10, "GKP": 5}
        outputs = {}
        for pos, n in positions.items():
            df = best_players(model, pos, top_n=n)
            df.to_csv(os.path.join(output_dir, f"best_{pos.lower()}s.csv"), index=False)
            outputs[pos] = df

        shortlist = pd.concat(outputs.values(), ignore_index=True)
        shortlist.to_csv(os.path.join(output_dir, "best_shortlist.csv"), index=False)

        print("\n🏆 Top 15 Overall:")
        print(best_players(model))

        for pos, df in outputs.items():
            print(f"\n🔹 Top {len(df)} {pos}:")
            print(df)

        print(f"\n✅ Master dataset: {os.path.join(output_dir, 'player_model.csv')}")
        print(f"✅ Positional picks saved in {output_dir}")
        print(f"✅ Combined shortlist saved as best_shortlist.csv")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
