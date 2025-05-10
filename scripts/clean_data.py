import pandas as pd
import os
from glob import glob

def clean_all_sites(input_folder="data", output_file="data/cleaned_all_sites.csv"):
    csv_files = sorted([f for f in glob(f"{input_folder}/*.csv") if "GroundwaterStations" not in f])

    combined = pd.DataFrame()

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip().str.upper()

            print(f"📄 {os.path.basename(file)} → Columns: {df.columns.tolist()}")

            # Try to auto-detect elevation column
            date_col = next((col for col in df.columns if 'DATE' in col), None)
            elev_col = next((col for col in df.columns if 'ELEV' in col or 'FEET' in col or 'RES' in col), None)

            if not date_col or not elev_col:
                print(f"⚠️ Skipping {file} — missing date or elevation-like column")
                continue

            df = df.rename(columns={date_col: 'date', elev_col: 'elevation'})
            df = df[['date', 'elevation']].copy()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['site'] = os.path.basename(file).replace(".csv", "")
            df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce')
            df = df.dropna(subset=['date', 'elevation'])


            combined = pd.concat([combined, df], ignore_index=True)
            print(f"✅ Processed {file} — {len(df)} rows")
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    if combined.empty:
        print("❌ No valid data found across all files.")
        return

    combined.to_csv(output_file, index=False)
    print(f"\n✅ Cleaned and saved to {output_file} — {len(combined)} total rows")
