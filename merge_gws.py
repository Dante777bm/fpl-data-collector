import os
import pandas as pd
import re

def get_current_season_folder():
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("FPL_Data_"):
            if "Unknown" not in item:
                return item
    return None

def find_gw_files(root_dir):
    gw_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("FPL_Data_GW_") and file.endswith(".csv"):
                gw_files.append(os.path.join(root, file))
    return gw_files

def merge_gw_files(gw_files):
    all_dfs = []
    for file in gw_files:
        df = pd.read_csv(file)
        gw_number = re.search(r'GW_(\d+)', file).group(1)
        df['GW'] = gw_number
        all_dfs.append(df)

    if not all_dfs:
        return None

    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

def main():
    current_season_folder = get_current_season_folder()
    if not current_season_folder:
        print("Current season folder not found.")
        return

    gw_files = find_gw_files(current_season_folder)

    if not gw_files:
        print("No gameweek CSV files found.")
        return

    merged_df = merge_gw_files(gw_files)

    if merged_df is not None:
        output_file = os.path.join(current_season_folder, "merged_gws.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"Merged data saved to {output_file}")

if __name__ == "__main__":
    main()
