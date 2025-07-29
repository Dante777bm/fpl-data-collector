import os
import pandas as pd

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
        all_dfs.append(df)

    if not all_dfs:
        return None

    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

def main():
    root_dir = "."
    gw_files = find_gw_files(root_dir)

    if not gw_files:
        print("No gameweek CSV files found.")
        return

    merged_df = merge_gw_files(gw_files)

    if merged_df is not None:
        output_file = "merged_gws.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"Merged data saved to {output_file}")

if __name__ == "__main__":
    main()
