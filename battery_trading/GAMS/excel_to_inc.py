import pandas as pd
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
file_path = '/Users/inderpal/Documents/power_market_data/battery_trading/orderbook_2024-01-01.xlsx'
output_file = 'data.inc'

# LIMIT THE DATA SIZE HERE
# Set to None for full data, or an integer (e.g., 100) to limit rows
MAX_ORDERS = 200  

scalars = {
    'mtu': 0.25,
    'f_max': 10.0,
    'f_min': -10.0,
    's0': 0.0,
    's_max': 50.0,
    'nu': 0.5,
    'eta_plus': 0.95,
    'eta_minus': 0.95
}

def generate_gams_data():
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print("Reading Excel file...")
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        
        # --- SLICE DATA TO FIT LICENSE ---
        if MAX_ORDERS:
            print(f"Reducing dataset from {len(df)} to {MAX_ORDERS} orders to fit Community License.")
            df = df.head(MAX_ORDERS)
        
        # 1. Generate Order IDs
        df['order_id'] = ['o' + str(i+1) for i in range(len(df))]
        
        # 2. Force strings
        df['start_str'] = df['start'].astype(str).str.strip()
        df['side_str'] = df['side'].astype(str).str.strip()

    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    print(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("* --- AUTOMATICALLY GENERATED DATA ---\n\n")

        # --- SCALARS ---
        f.write("* 1. Scalars\n")
        for key, val in scalars.items():
            f.write(f"{key} = {val};\n")
        f.write("\n")

        # --- SETS ---
        f.write("* 2. Sets\n")
        
        # Set: Orders
        f.write("Set o /\n")
        for o_id in df['order_id']:
            f.write(f"  {o_id}\n")
        f.write("/;\n\n")

        # Set: Time (t)
        unique_times = sorted(df['start_str'].unique())
        f.write("Set t /\n")
        for t_val in unique_times:
            f.write(f"  '{t_val}'\n")
        f.write("/;\n\n")

        # Set: Sides
        unique_sides = sorted(df['side_str'].unique())
        f.write("Set sides /\n")
        for s_val in unique_sides:
            f.write(f"  {s_val}\n")
        f.write("/;\n\n")

        # --- MAPPINGS ---
        f.write("* 3. Mappings\n")
        
        # Map: Order -> Time
        f.write("Set map_ot(o,t) /\n")
        for i, row in df.iterrows():
            f.write(f"  {row['order_id']} . '{row['start_str']}'\n")
        f.write("/;\n\n")

        # Map: Order -> Side
        f.write("Set map_oside(o,sides) /\n")
        for i, row in df.iterrows():
            f.write(f"  {row['order_id']} . {row['side_str']}\n")
        f.write("/;\n\n")

        # --- PARAMETERS ---
        f.write("* 4. Parameters\n")

        # Price
        f.write("Parameter price(o) /\n")
        for i, row in df.iterrows():
            f.write(f"  {row['order_id']}  {row['price']}\n")
        f.write("/;\n\n")

        # Quantity
        f.write("Parameter quantity(o) /\n")
        for i, row in df.iterrows():
            f.write(f"  {row['order_id']}  {row['quantity']}\n")
        f.write("/;\n")

    print(f"Success! Reduced data written to {output_file}")

if __name__ == "__main__":
    generate_gams_data()