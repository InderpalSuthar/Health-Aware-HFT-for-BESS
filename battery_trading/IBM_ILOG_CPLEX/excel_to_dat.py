import pandas as pd

def generate_dat_file(excel_file, output_dat_file, max_orders=100):
    """
    Reads Excel file and generates OPL .dat file
    
    Parameters:
    excel_file: path to your Excel file
    output_dat_file: path for output .dat file
    max_orders: maximum number of orders to include (default 100 for Community Edition)
    """
    
    # Read Excel file
    print(f"Reading Excel file: {excel_file}")
    df = pd.read_excel(excel_file)
    
    # Limit to max_orders
    df = df.head(max_orders)
    print(f"Using first {len(df)} orders (limited for Community Edition)")
    
    # Extract unique products (time periods) and sort them chronologically
    products = sorted(df['start'].unique())
    print(f"Unique time periods: {len(products)}")
    
    # Battery parameters
    battery_params = {
        'mtu': 0.1,
        'f_max': 10.0,
        'f_min': -10.0,
        's0': 5.0,
        's_max': 10.0,
        'nu': 4.09,
        'eta_plus': 0.95,
        'eta_minus': 0.95
    }
    
    # Start writing the .dat file
    print(f"Generating .dat file: {output_dat_file}")
    with open(output_dat_file, 'w') as f:
        f.write("/*********************************************\n")
        f.write(" * Auto-generated Data File\n")
        f.write(f" * Total Orders: {len(df)}\n")
        f.write(f" * Time Periods: {len(products)}\n")
        f.write(" * LIMITED FOR CPLEX COMMUNITY EDITION\n")
        f.write(" *********************************************/\n\n")
        
        # Write battery parameters
        f.write("// Battery parameters\n")
        for param, value in battery_params.items():
            f.write(f"{param} = {value};\n")
        f.write("\n")
        
        # Write NumOrders
        f.write("// Number of orders\n")
        f.write(f"NumOrders = {len(df)};\n\n")
        
        # Write side array
        f.write("// Order Side (BUY/SELL)\n")
        f.write("side = [\n")
        for i, row in df.iterrows():
            side_val = row['side']
            if i < len(df) - 1:
                f.write(f'  "{side_val}",\n')
            else:
                f.write(f'  "{side_val}"\n')
        f.write("];\n\n")
        
        # Write product array (start time)
        f.write("// Order Product (Time Period)\n")
        f.write("product = [\n")
        for i, row in df.iterrows():
            prod_val = row['start']
            if i < len(df) - 1:
                f.write(f'  "{prod_val}",\n')
            else:
                f.write(f'  "{prod_val}"\n')
        f.write("];\n\n")
        
        # Write price array
        f.write("// Order Price\n")
        f.write("price = [\n")
        for i, row in df.iterrows():
            price_val = row['price']
            if i < len(df) - 1:
                f.write(f'  {price_val},\n')
            else:
                f.write(f'  {price_val}\n')
        f.write("];\n\n")
        
        # Write quantity array
        f.write("// Order Quantity\n")
        f.write("quantity = [\n")
        for i, row in df.iterrows():
            qty_val = row['quantity']
            if i < len(df) - 1:
                f.write(f'  {qty_val},\n')
            else:
                f.write(f'  {qty_val}\n')
        f.write("];\n")
    
    print(f"✓ Successfully generated {output_dat_file}")
    print(f"  - Orders included: {len(df)}")
    print(f"  - Unique time periods: {len(products)}")
    print(f"\nNOTE: CPLEX Community Edition limits:")
    print(f"  - Max 1000 variables")
    print(f"  - Max 1000 constraints")

# Run the script
if __name__ == "__main__":
    excel_file = "/Users/inderpal/Documents/power_market_data/battery_trading/orderbook_2024-01-01.xlsx"  # Your Excel file
    output_dat = "data.dat"  # Output .dat file
    
    # Adjust max_orders based on your needs (100 is safe for Community Edition)
    max_orders = 200  # Change this value to test different sizes
    
    try:
        generate_dat_file(excel_file, output_dat, max_orders)
    except FileNotFoundError:
        print(f"Error: Could not find file '{excel_file}'")
        print("Please make sure the Excel file is in the same directory")
    except Exception as e:
        print(f"Error: {e}")