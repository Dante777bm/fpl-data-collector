import pandas as pd
from fpl_community_detector import FPLCommunityDetector
import os

def main():
    # --- 1. Load and Preprocess Data ---

    # To analyze a single gameweek, specify the file path directly.
    # To analyze multiple gameweeks, you would typically load each CSV,
    # add a 'gameweek' column, and then concatenate them into a single DataFrame.
    # For this example, we'll use the latest available gameweek data.

    season_dir = 'FPL_Data_2025-26'
    gw_files = [f for f in os.listdir(season_dir) if f.startswith('FPL_Data_GW_') and f.endswith('.csv')]

    if not gw_files:
        print(f"No gameweek data found in {season_dir}.")
        return

    # Sort files to get the latest gameweek
    latest_gw_file = sorted(gw_files, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)[0]
    latest_gw_path = os.path.join(season_dir, latest_gw_file)

    print(f"Loading data from: {latest_gw_path}")
    fpl_data = pd.read_csv(latest_gw_path)

    # --- 2. Initialize the Community Detector ---
    detector = FPLCommunityDetector(fpl_data)

    # --- 3. Create Networks ---

    # Performance-based network
    print("\nCreating performance network...")
    detector.create_performance_network(threshold=0.6)

    # Price and form-based network
    print("Creating price-form network...")
    detector.create_price_form_network(cost_threshold=1.0, form_threshold=1.5)

    # --- 4. Detect Communities ---

    # Detect communities in the performance network
    print("\nDetecting communities in performance network...")
    detector.detect_communities('performance')

    # --- 5. Analyze and Visualize Communities ---

    # Analyze the detected communities
    print("\nAnalyzing performance communities...")
    analysis = detector.analyze_communities('performance')

    # Print analysis for the top 5 communities
    if analysis:
        for comm in analysis[:5]:
            print(f"\nCommunity {comm['community_id']} (Size: {comm['size']}):")
            print(f"  Players: {', '.join(comm['players'][:5])}{'...' if len(comm['players']) > 5 else ''}")
            print(f"  Positions: {comm['positions']}")
            print(f"  Avg Cost: £{comm['avg_cost']:.1f}m")
            print(f"  Avg Points: {comm['avg_points']:.1f}")

    # Visualize the communities
    print("\nVisualizing performance communities...")
    detector.visualize_communities('performance')

if __name__ == "__main__":
    main()
