
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))

from MILP import BatteryParams, MarketParams, DegradationModel, RollingIntrinsicBacktest, OrderBookLoader
from DP import DPRollingIntrinsicBacktest

def run_comparison():
    print("="*80)
    print("Optimization Strategy Comparison: MILP vs DP")
    print("="*80)
    
    # Parameters
    battery = BatteryParams()
    market = MarketParams()
    degradation = DegradationModel(battery)
    
    # Load Data
    csv_path = "code/strategy_comparison_data/orderbook_feb_01_14.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    loader = OrderBookLoader(csv_path)
    
    # Configuration
    initial_soc = 6.25
    horizon = 4
    
    # 1. Run MILP
    print("\n[1/2] Running Exact MILP Strategy...")
    milp_tester = RollingIntrinsicBacktest(battery, market, degradation)
    milp_results = milp_tester.run(loader, initial_soc, max_horizon_hours=horizon)
    
    # 2. Run DP (C++ Optimized)
    print("\n[2/2] Running Fast DP Strategy...")
    dp_tester = DPRollingIntrinsicBacktest(battery, market, degradation, dp_grid_size=201)
    dp_results = dp_tester.run(loader, initial_soc, max_horizon_hours=horizon)
    
    # 3. Analysis & Plotting
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS")
    print("="*80)
    
    metrics = ['Net Profit (€)', 'Degradation (€)', 'Revenue (€)', 'Avg Time (ms)', 'Total Time (s)']
    milp_vals = [
        milp_results['net_profit'], milp_results['degradation_cost'], 
        milp_results['gross_revenue'], milp_results['avg_solve_time_ms'], 
        milp_results['total_solve_time_s']
    ]
    dp_vals = [
        dp_results['net_profit'], dp_results['degradation_cost'], 
        dp_results['gross_revenue'], dp_results['avg_solve_time_ms'], 
        dp_results['total_solve_time_s']
    ]
    
    df = pd.DataFrame({'MILP': milp_vals, 'DP': dp_vals}, index=metrics)
    print(df)
    
    # Speedup
    if dp_vals[3] > 0:
        speedup = milp_vals[3] / dp_vals[3]
        print(f"\nSpeedup Factor (MILP/DP): {speedup:.2f}x")
    
    # --- PLOTTING ---
    print("\nGenerating Advanced Plots...")
    
    # Extract Timestamps
    timeline = [x['time'] for x in milp_results['soc_history']]
    
    # Create Figure with 3 Subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: Market Interaction (Price + Signals)
    # Extract Prices
    prices = [next((p['best_bid'] for p in dp_results.get('price_history', []) if p['time'] == t), float('nan')) for t in timeline]
    # If using stored history directly:
    if dp_results.get('price_history'):
        price_times = [p['time'] for p in dp_results['price_history']]
        price_vals = [p['best_ask'] for p in dp_results['price_history']] # Use Ask for reference
        best_bids = [p['best_bid'] for p in dp_results['price_history']] 
        # Plot mid price line
        mid_prices = [(p['best_bid'] + p['best_ask'])/2 for p in dp_results['price_history']]
        axes[0].plot(price_times, mid_prices, label='Mid Price', color='gray', alpha=0.5, linewidth=1)
    
    # Plot DP Trades: Buy (Green Triangle Up), Sell (Red Triangle Down)
    dp_trades = dp_results.get('trade_history', [])
    buy_times = [t['time'] for t in dp_trades if t['is_buy']]
    buy_prices = [t['price'] for t in dp_trades if t['is_buy']]
    sell_times = [t['time'] for t in dp_trades if not t['is_buy']]
    sell_prices = [t['price'] for t in dp_trades if not t['is_buy']]
    
    axes[0].scatter(buy_times, buy_prices, marker='^', color='green', s=50, label='DP Buy', zorder=5)
    axes[0].scatter(sell_times, sell_prices, marker='v', color='red', s=50, label='DP Sell', zorder=5)
    axes[0].set_ylabel('Price (€/MWh)')
    axes[0].set_title('Market Signal Analysis: DP Strategy Execution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: SoC Profile Comparison
    milp_soc = [x['soc'] for x in milp_results['soc_history']]
    dp_soc = [x['soc'] for x in dp_results['soc_history']]
    
    axes[1].plot(timeline, milp_soc, label='MILP (Exact)', linewidth=1.5, alpha=0.8)
    axes[1].plot(timeline, dp_soc, label='DP (Fast)', linestyle='--', linewidth=1.5, color='orange')
    axes[1].axhline(y=battery.energy_min, color='r', linestyle=':', alpha=0.5)
    axes[1].axhline(y=battery.energy_max, color='r', linestyle=':', alpha=0.5)
    axes[1].set_ylabel('State of Charge (MWh)')
    axes[1].set_title('Battery Inventory Management')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Solve Time (Log Scale)
    if 'solve_time_history' in milp_results and 'solve_time_history' in dp_results:
        # Align lengths if slightly different (can happen if solves skipped)
        min_len = min(len(milp_results['solve_time_history']), len(dp_results['solve_time_history']), len(timeline))
        # Use simple index for x-axis if lengths differ from timeline, or slice timeline
        plot_timeline = timeline[:min_len]
        
        milp_times = milp_results['solve_time_history'][:min_len]
        dp_times = dp_results['solve_time_history'][:min_len]
        
        axes[2].plot(plot_timeline, milp_times, label='MILP Time', alpha=0.7)
        axes[2].plot(plot_timeline, dp_times, label='DP Time', alpha=0.7)
        axes[2].set_yscale('log')
        axes[2].set_ylabel('Solve Time (Seconds)')
        axes[2].set_title('Computational Efficiency (Log Scale)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel('Time')
    else:
        axes[2].text(0.5, 0.5, 'Solve Time Data Not Available', ha='center')
    
    plt.tight_layout()
    plt.savefig('comparison_analysis.png', dpi=300)
    print("\nAdvanced Plot saved to comparison_analysis.png")

if __name__ == "__main__":
    run_comparison()
