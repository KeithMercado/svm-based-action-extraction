import pandas as pd
import os

# Define the paths clearly
base_path = r"C:\Users\John Keith Mercado\ThesisModel\data"
input_file = os.path.join(base_path, "ami_final_accurate_dataset.csv")
output_file = os.path.join(base_path, "ami_dataset_cleaned.csv")

# Check if file exists before reading
if not os.path.exists(input_file):
    print(f"[Error] Could not find the file at: {input_file}")
else:
    print(f"[System] Loading {input_file}...")
    df = pd.read_csv(input_file)

    # 1. Filter: Keep all Information Items
    # 2. Filter: Keep Action Items ONLY if they are longer than 5 words
    # This removes "You know .", "Okay .", "Right .", etc.
    df_clean = df[
        (df['label'] == 'information_item') | 
        ((df['label'] == 'action_item') & (df['text'].str.split().str.len() > 5))
    ]

    # Save the polished version
    df_clean.to_csv(output_file, index=False)

    print("-" * 40)
    print(f"Original Rows: {len(df)}")
    print(f"Cleaned Rows: {len(df_clean)}")
    print(f"Action Items remaining: {len(df_clean[df_clean['label'] == 'action_item'])}")
    print(f"Cleaned file saved at: {output_file}")
    print("-" * 40)