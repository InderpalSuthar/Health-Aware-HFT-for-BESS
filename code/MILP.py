"""
Health-Aware High-Frequency Trading for Battery Energy Storage Systems
Implements MILP (Gurobi) approach 

Usage:
    python script.py path/to/orderbook.csv

CSV Format Required:
    - side: BUY or SELL
    - start: Delivery period start (ISO 8601)
    - transaction: Order submission time (ISO 8601)
    - validity: Order expiry time (ISO 8601, optional)
    - price: Order price (€/MWh)
    - quantity: Order quantity (MWh)
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class BatteryParams:
    """Battery system parameters"""
    power_max: float = 20.0  # MW (charging limit) - from 20 MW rating : fbar
    power_min: float = -20.0  # MW (discharging limit) : funderbar
    energy_capacity: float = 12.5  # MWh - from 12.5 MWh capacity : Erate
    energy_min: float = 1.875  # MWh - 15% of 12.5 MWh : Emin
    energy_max: float = 11.875  # MWh - 95% of 12.5 MWh : Emax
    eta_charge: float = 0.95  # Charging efficiency - 95% : ita+
    eta_discharge: float = 0.95  # Discharging efficiency - 95% : ita-
    replacement_cost: float = 300000.0  # $/MWh - from $300,000/MWh : R
    num_segments: int = 16  # Piecewise linear segments for degradation : J
    shelf_life: float = 10.0  # years - from 10-year shelf life : used in deriving the 11v
    cycle_life: int = 3000  # cycles at 80% DoD : used in deriving the 11v


@dataclass
class MarketParams:
    """Market parameters"""
    trading_fee: float = 0.09  # €/MWh : nu 
    min_trading_unit: float = 0.1  # MWh : u = lot size 
    time_interval: float = 0.25  # hours : M (Quarter-Hourly Products)


@dataclass
class Order:
    """Limit order book entry"""
    price: float  # €/MWh : Pi
    quantity: float  # MWh : Qi
    is_buy: bool  # True for buy orders, False for sell orders : Bids and asks 


class DegradationModel:
    """Battery degradation model with piecewise linear approximation
    
    Implements the degradation model from the paper:
    "Health-Aware High-Frequency Trading for Battery Energy Storage Systems"
    
    Key equations:
    - Stress function: Φ(δ) = 5.24 X 10⁻⁴ X δ²·⁰³
    - Segment cost: c_j = (R X J / (η⁻ X E_rate)) X [Φ((j+1)/J) - Φ(j/J)]
    
    Where:
    - δ: Cycle depth (0-1)
    - R: Replacement cost (€/MWh)
    - J: Number of segments
    - η⁻: Discharging efficiency
    - E_rate: Rated energy capacity (MWh)
    """
    
    def __init__(self, battery):
        """
        Initialize degradation model
        
        Args:
            battery: BatteryParams object
        """
        self.battery = battery
        self.J = battery.num_segments
        
        # Compute segment costs
        self.segment_costs = self._compute_segment_costs()
    
    def stress_function(self, delta: float) -> float: # (11v)
        """
        Cycle depth stress function for Li(NiMnCo)O2 cells
        
        Equation (6) from the paper: Φ(δ) = 5.24 × 10⁻⁴ × δ²·⁰³
        
        Args:
            delta: Cycle depth (0-1)
        
        Returns:
            Stress value Φ(δ)
        """
        return 5.24e-4 * (delta ** 2.03)
    
    def _compute_segment_costs(self) -> np.ndarray: # (11u)
        """
        Compute marginal degradation cost for each cycle depth segment
        
        Equation (7) from the paper:
        c_j = (R × J / (η⁻ × Ē)) × [Φ((j+1)/J) - Φ(j/J)]
        
        Returns:
            Array of segment costs (€/MWh)
        """
        costs = np.zeros(self.J)
        R = self.battery.replacement_cost  # €/MWh
        eta_dis = self.battery.eta_discharge  # η⁻
        E_rated = self.battery.energy_capacity  # Ē
        
        for j in range(self.J):
            # Cycle depth bounds for segment j
            delta_upper = (j + 1) / self.J  # (j+1)/J
            delta_lower = j / self.J        # j/J
            
            # Stress function values
            phi_upper = self.stress_function(delta_upper)
            phi_lower = self.stress_function(delta_lower)
            
            # Segment marginal cost
            costs[j] = (R * self.J / (eta_dis * E_rated)) * (phi_upper - phi_lower)
        
        return costs


class HealthAwareMILP:
    """MILP formulation for health-aware battery trading"""
    def __init__(self, battery: BatteryParams, market: MarketParams, degradation: DegradationModel):
        self.battery = battery
        self.market = market
        self.degradation = degradation
    
    def solve(self, orders: Dict[int, List[Order]], initial_soc: float,
              time_horizon: List[int], committed_power: Optional[Dict[int, float]] = None,
              final_soc: Optional[float] = None,
              initial_segment_energy: Optional[Dict[int, float]] = None) -> Dict:
        """
        Solve the health-aware intrinsic optimization problem using MILP
        """
        T = len(time_horizon)
        J = self.degradation.J
        M = self.market.time_interval
        u = self.market.min_trading_unit
        nu = self.market.trading_fee
        
        if committed_power is None:
            committed_power = {t: 0.0 for t in time_horizon} # ft0 = 0
        
        # Initial segment energy : ej0 distribution
        if initial_segment_energy is None: 
            segment_width = self.battery.energy_capacity / J
            n_full = int(initial_soc / segment_width)
            frac_fill = (initial_soc / segment_width) - n_full
            
            initial_segment_energy = {}
            for j in range(J):
                if j < n_full:
                    initial_segment_energy[j] = segment_width
                elif j == n_full:
                    initial_segment_energy[j] = frac_fill * segment_width
                else:
                    initial_segment_energy[j] = 0.0
        
        model = gp.Model("HealthAwareBESS")
        model.setParam('OutputFlag', 0)
        model.setParam('MIPGap', 0.01)  # 1% optimality gap for speed
        model.setParam('TimeLimit', 30)  # Max 30 seconds per solve
        
        # ==================== DECISION VARIABLES ====================
        
        # Order execution variables
        k = {}
        q = {}
        for t in time_horizon:
            for i, order in enumerate(orders.get(t, [])):
                max_blocks = int(order.quantity / u)
                k[i, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_blocks,
                                       name=f"k_{i}_{t}") # (11w)
                q[i, t] = model.addVar(lb=0, ub=order.quantity, name=f"q_{i}_{t}")
        
        # Operating mode (1=charge, 0=discharge) : (11w)
        alpha = {t: model.addVar(vtype=GRB.BINARY, name=f"alpha_{t}") 
                 for t in time_horizon}
        
        # Segment-based power flows
        p_ch = {(t, j): model.addVar(lb=0, name=f"p_ch_{t}_{j}")
                for t in time_horizon for j in range(J)}
        p_dis = {(t, j): model.addVar(lb=0, name=f"p_dis_{t}_{j}")
                 for t in time_horizon for j in range(J)}
        
        # Energy in segments (virtual storage model)
        e_bar_j = self.battery.energy_capacity / J # ebar = Erate/J
        e = {(t, j): model.addVar(lb=0, ub=e_bar_j, name=f"e_{t}_{j}")
             for t in time_horizon for j in range(J)} # (11r)
        
        # Total state of charge : (11l)
        s = {t: model.addVar(lb=self.battery.energy_min, ub=self.battery.energy_max,
                            name=f"s_{t}") for t in time_horizon}
        
        # Total charging and discharging power : lb = 0 means both positive 
        i_var = {t: model.addVar(lb=0, name=f"i_{t}") for t in time_horizon}
        w_var = {t: model.addVar(lb=0, name=f"w_{t}") for t in time_horizon}
        
        # Buying and selling power
        f_plus = {t: model.addVar(lb=0, name=f"f_plus_{t}") for t in time_horizon}
        f_minus = {t: model.addVar(lb=0, name=f"f_minus_{t}") for t in time_horizon}
        
        # Net power exchange
        f = {t: model.addVar(lb=M*self.battery.power_min, 
                            ub=M*self.battery.power_max,
                            name=f"f_{t}") for t in time_horizon}
        
        model.update()
        
        # ==================== OBJECTIVE FUNCTION ====================
        obj = 0
        for t in time_horizon:
            # Revenue from selling to BUY orders (market buyers want to buy, battery sells)
            for i, order in enumerate(orders.get(t, [])):
                if order.is_buy:
                    obj += (order.price - nu) * q[i, t]
            
            # Cost from buying from SELL orders (market sellers want to sell, battery buys)
            for i, order in enumerate(orders.get(t, [])):
                if not order.is_buy:
                    obj -= (order.price + nu) * q[i, t]
            
            # Degradation cost (negative since it's a cost)
            # Only discharge degradation as per paper
            for j in range(J):
                obj -= M * self.degradation.segment_costs[j] * p_dis[t, j]
        
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # ==================== CONSTRAINTS ====================
        
        for t_idx, t in enumerate(time_horizon):
            
            # Order execution constraints : (11b, 11c)
            for i, order in enumerate(orders.get(t, [])):
                model.addConstr(q[i, t] >= 0)
                model.addConstr(q[i, t] <= order.quantity)
                model.addConstr(q[i, t] == k[i, t] * u)
            
            # Total buying and selling : (11d, 11e)
            # When market has BUY orders (buyers), battery SELLS to them (f_minus = power out)
            # When market has SELL orders (sellers), battery BUYS from them (f_plus = power in)
            buy_orders = [i for i, order in enumerate(orders.get(t, [])) if order.is_buy]
            if buy_orders:
                model.addConstr(f_minus[t] == gp.quicksum(q[i, t] for i in buy_orders))
            else:
                model.addConstr(f_minus[t] == 0)
            
            sell_orders = [i for i, order in enumerate(orders.get(t, [])) if not order.is_buy]
            if sell_orders:
                model.addConstr(f_plus[t] == gp.quicksum(q[i, t] for i in sell_orders))
            else:
                model.addConstr(f_plus[t] == 0)
            
            # Net power exchange : (11f)
            model.addConstr(f[t] == committed_power[t] + f_plus[t] - f_minus[t])
            
            # Power limits : (11g)
            model.addConstr(f[t] >= M * self.battery.power_min)
            model.addConstr(f[t] <= M * self.battery.power_max)
            
            # Charging/discharging decomposition : (11h)
            model.addConstr(f[t] == M * (i_var[t] - w_var[t]))
            
            # Charging limits : (11i)
            model.addConstr(i_var[t] >= 0)
            model.addConstr(i_var[t] <= alpha[t] * self.battery.power_max)
            
            # Discharging limits : (11j)
            model.addConstr(w_var[t] >= 0)
            model.addConstr(w_var[t] <= (1 - alpha[t]) * (-self.battery.power_min))

            # SoC dynamics : (11k)
            # Calculate previous SoC
            if t_idx == 0:
                s_prev = initial_soc
            else:
                s_prev = s[time_horizon[t_idx-1]]
                
            model.addConstr(
                s[t] == s_prev + M * (
                    self.battery.eta_charge * i_var[t] - 
                    w_var[t] / self.battery.eta_discharge
                )
            )
            
            # Segment power aggregation : (11m, 11n)
            model.addConstr(w_var[t] == gp.quicksum(p_dis[t, j] for j in range(J)))
            model.addConstr(i_var[t] == gp.quicksum(p_ch[t, j] for j in range(J)))
            
            # Segment energy dynamics : (11o, 11s)
            for j in range(J):
                if t_idx == 0:
                    e_prev = initial_segment_energy[j]
                else:
                    e_prev = e[time_horizon[t_idx-1], j]
                # Energy in segment after charge/discharge 
                model.addConstr(
                    e[t, j] == e_prev + M * (
                        p_ch[t, j] * self.battery.eta_charge - 
                        p_dis[t, j] / self.battery.eta_discharge
                    )
                )
                # Non-negative power : (11p)
                model.addConstr(p_ch[t, j] >= 0)
                model.addConstr(p_dis[t, j] >= 0)
            
            # Total SoC : (11q)
            model.addConstr(s[t] == gp.quicksum(e[t, j] for j in range(J)))
            
        # Final SoC constraint : (11t)
        if final_soc is not None:
            model.addConstr(s[time_horizon[-1]] >= final_soc)
        
        # ==================== SOLVE ====================
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        # Extract solution
        if model.status == GRB.OPTIMAL:
            solution = {
                'status': 'optimal',
                'objective': model.objVal,
                'solve_time': solve_time,
                'trades': {},
                'soc': {},
                'power': {},
                'segment_energy': {},
                'segment_discharge': {},
                'degradation_cost': 0.0
            }
            
            for t in time_horizon:
                solution['soc'][t] = s[t].X
                solution['power'][t] = f[t].X / M
                solution['trades'][t] = []
                solution['segment_energy'][t] = {}
                solution['segment_discharge'][t] = {}
                
                # Track segment-level information
                for j in range(J):
                    solution['segment_energy'][t][j] = e[t, j].X
                    solution['segment_discharge'][t][j] = p_dis[t, j].X
                
                # Track trades
                for i, order in enumerate(orders.get(t, [])):
                    if k[i, t].X > 0.1:  # Minimum trade size
                        solution['trades'][t].append({
                            'order_idx': i,
                            'blocks': int(round(k[i, t].X)),
                            'quantity': k[i, t].X * u,
                            'price': order.price,
                            'is_buy': order.is_buy
                        })
                
                # Track degradation cost (only from discharge)
                for j in range(J):
                    if p_dis[t, j].X > 0.001:  # Small tolerance
                        solution['degradation_cost'] += (
                            M * self.degradation.segment_costs[j] * p_dis[t, j].X
                        )
            
            return solution
        else:
            return {'status': 'infeasible', 'solve_time': solve_time}


class OrderBookLoader:
    """Load and process EPEX SPOT-style order book data from CSV"""
    
    def __init__(self, csv_path: str):
        """
        Initialize order book loader
        
        Args:
            csv_path: Path to CSV file with order book data
        """
        self.csv_path = csv_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load order book data from CSV"""
        import pandas as pd
        
        print(f"Loading order book data from {self.csv_path}...")
        
        # Read the CSV file
        self.data = pd.read_csv(self.csv_path)
        
        print(f"Original columns: {list(self.data.columns)}")
        
        # Handle the special format: empty first column + 'initial' column
        # Your format: ,initial,side,start,transaction,validity,price,quantity
        # The first column is usually an index that gets read as 'Unnamed: 0'
        
        # Option 1: If first column is empty/unnamed, drop it
        if self.data.columns[0].startswith('Unnamed'):
            print(f"Dropping unnamed column: {self.data.columns[0]}")
            self.data = self.data.drop(columns=[self.data.columns[0]])
        
        # Option 2: If first column has a name but we don't need it
        elif 'initial' in self.data.columns:
            # Keep only the columns we need
            required_columns = ['side', 'start', 'transaction', 'validity', 'price', 'quantity']
            if all(col in self.data.columns for col in required_columns):
                # We have all required columns, drop 'initial'
                print("Dropping 'initial' column")
                self.data = self.data[required_columns]
            else:
                # Try to rename columns if they're in different order
                print("Attempting to map columns...")
                column_mapping = {}
                if len(self.data.columns) >= 7:  # Expecting at least 7 columns
                    # Map based on position
                    column_mapping = {
                        self.data.columns[1]: 'side',
                        self.data.columns[2]: 'start',
                        self.data.columns[3]: 'transaction',
                        self.data.columns[4]: 'validity',
                        self.data.columns[5]: 'price',
                        self.data.columns[6]: 'quantity'
                    }
                    self.data = self.data.rename(columns=column_mapping)
                    # Keep only the mapped columns
                    self.data = self.data[list(column_mapping.values())]
        
        print(f"Processed columns: {list(self.data.columns)}")
        
        # Convert timestamps - ensure timezone awareness
        try:
            self.data['transaction'] = pd.to_datetime(self.data['transaction'], utc=True)
            self.data['start'] = pd.to_datetime(self.data['start'], utc=True)
        except KeyError as e:
            print(f"ERROR: Missing required column {e}")
            print("Available columns:", list(self.data.columns))
            raise
        
        # Handle validity (can be NaN) - also make timezone-aware
        if 'validity' in self.data.columns:
            self.data['validity'] = pd.to_datetime(self.data['validity'], errors='coerce', utc=True)
        
        print(f"Loaded {len(self.data)} orders")
        print(f"Date range: {self.data['start'].min()} to {self.data['start'].max()}")
        print(f"Products: {self.data['start'].nunique()} unique delivery periods")
        
        # Print sample of data
        print("\nSample of first 5 orders:")
        for i, row in self.data.head().iterrows():
            print(f"  {row['side']} {row['quantity']} MWh at {row['price']} €/MWh for {row['start']}")
    
    def get_orders_at_time(self, current_time: pd.Timestamp, 
                          delivery_start: pd.Timestamp) -> List[Order]:
        """
        Get active orders at a specific time for a delivery period
        
        Args:
            current_time: Current market time (when we're making decision)
            delivery_start: Delivery period start time
        
        Returns:
            List of Order objects
        """
        # Filter orders for this delivery period
        mask = self.data['start'] == delivery_start
        
        # Order must have been submitted before current time
        mask &= self.data['transaction'] <= current_time
        
        # Order must still be valid (if validity exists)
        if 'validity' in self.data.columns:
            # NaN validity means order stays active
            mask &= (self.data['validity'].isna() | (self.data['validity'] > current_time))
        
        active_orders = self.data[mask]
        
        # Convert to Order objects
        orders = []
        for _, row in active_orders.iterrows():
            is_buy = row['side'].upper() == 'BUY'  # Handle case sensitivity
            orders.append(Order(
                price=float(row['price']),
                quantity=float(row['quantity']),
                is_buy=is_buy
            ))
        
        return orders
    
    def get_order_book_snapshot(self, current_time: pd.Timestamp) -> Dict[pd.Timestamp, List[Order]]:
        """
        Get order book snapshot for all active products at current time
        
        Args:
            current_time: Current market time
        
        Returns:
            Dictionary mapping delivery start time to list of orders
        """
        # Get all delivery periods that haven't happened yet (or are happening now)
        future_products = self.data[self.data['start'] >= current_time]['start'].unique()
        
        order_book = {}
        for product in sorted(future_products):
            orders = self.get_orders_at_time(current_time, product)
            if orders:  # Only include if there are active orders
                order_book[product] = orders
        
        return order_book
    
    def get_trading_timeline(self) -> List[pd.Timestamp]:
        """Get all unique transaction times for backtesting"""
        return sorted(self.data['transaction'].unique())
    
    def get_best_bid_ask(self, orders: List[Order]) -> Tuple[float, float, float]:
        """
        Get best bid, best ask, and spread from order list
        
        Returns:
            (best_bid, best_ask, spread)
        """
        buy_orders = [o for o in orders if o.is_buy]
        sell_orders = [o for o in orders if not o.is_buy]
        
        best_bid = max([o.price for o in buy_orders]) if buy_orders else 0.0
        best_ask = min([o.price for o in sell_orders]) if sell_orders else 1e6
        
        if best_bid > 0 and best_ask < 1e6:
            spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
        else:
            spread = 0.0
        
        return best_bid, best_ask, spread


class RollingIntrinsicBacktest:
    """Backtest the Rolling Intrinsic policy on historical order book data"""
    
    def __init__(self, battery: BatteryParams, market: MarketParams,
                 degradation: DegradationModel, phi: float = 0.0):
        """
        Initialize backtesting framework
        
        Args:
            battery: Battery parameters
            market: Market parameters
            degradation: Degradation model
            phi: Bid-ask spread penalty parameter
        """
        self.battery = battery
        self.market = market
        self.degradation = degradation
        self.phi = phi
        
        self.solver = HealthAwareMILP(battery, market, degradation)
        
        # Tracking
        self.soc_history = []
        self.profit_history = []
        self.degradation_history = []
        self.solve_time_history = []
        self.trade_history = []
        self.price_history = []
    
    def run(self, order_book_loader: OrderBookLoader, 
            initial_soc: float = 5.0,
            max_horizon_hours: int = 24,
            update_frequency: str = 'hourly') -> Dict:
        """
        Run backtest on order book data
        
        Args:
            order_book_loader: Loaded order book data
            initial_soc: Initial battery state of charge (MWh)
            max_horizon_hours: Maximum optimization horizon in hours
            update_frequency: How often to re-optimize ('every_update', 'hourly', 'minutely')
        
        Returns:
            Dictionary with backtest results
        """
        import pandas as pd
        
        print(f"\n{'='*70}")
        print(f"Running Rolling Intrinsic Backtest")
        print(f"Solver: 'MILP'")
        print(f"Update frequency: {update_frequency}")
        print(f"Phi penalty: {self.phi}")
        print(f"{'='*70}\n")
        
        current_soc = initial_soc
        cumulative_revenue = 0.0  # Gross revenue from trading
        cumulative_trading_cost = 0.0  # Trading fees paid
        cumulative_degradation = 0.0  # Degradation cost incurred
        
        # Get trading timeline
        timeline = order_book_loader.get_trading_timeline()
        
        # Filter based on update frequency
        if update_frequency == 'hourly':
            timeline = [t for t in timeline if t.minute == 0]
        elif update_frequency == 'minutely':
            timeline = [t for t in timeline if t.second == 0]
        
        print(f"Timeline: {len(timeline)} decision points")
        print(f"Start: {timeline[0]}")
        print(f"End: {timeline[-1]}\n")
        
        total_solves = 0
        last_print_time = time.time()
        
        for idx, current_time in enumerate(timeline):
            # Print progress every 10 solves or every 5 seconds
            current_wall_time = time.time()
            if idx % 10 == 0 or (current_wall_time - last_print_time) > 5:
                pct = 100 * idx / len(timeline)
                net_profit = cumulative_revenue - cumulative_trading_cost - cumulative_degradation
                print(f"Progress: {idx}/{len(timeline)} ({pct:.1f}%) | "
                      f"SoC: {current_soc:.2f} MWh | "
                      f"Net Profit: €{net_profit:.2f} | "
                      f"Solves: {total_solves}")
                last_print_time = current_wall_time
            
            # Get current order book snapshot
            order_book = order_book_loader.get_order_book_snapshot(current_time)
            
            # --- Capture Price Data for Analysis ---
            # Get best bid/ask for the *current* delivery period (t=0)
            avg_price_bid = 0.0
            avg_price_ask = 0.0
            if current_time in order_book:
                # Typically we trade on the nearest products. 
                # Let's record the price of the product delivering at `current_time` (or next hour)
                # Actually, standard intrinsic trades forward.
                # Let's record the spot price proxy (product starting at current_time)
                # If current_time is 14:00, we look for product starting 14:00
                
                spot_orders = order_book[current_time]
                bids = [o.price for o in spot_orders if o.is_buy]
                asks = [o.price for o in spot_orders if not o.is_buy]
                best_bid = max(bids) if bids else float('nan')
                best_ask = min(asks) if asks else float('nan')
                
                self.price_history.append({
                    'time': current_time,
                    'best_bid': best_bid,
                    'best_ask': best_ask
                })
            # ---------------------------------------
            
            if not order_book:
                continue
            
            # Convert to time periods (hours from now)
            delivery_times = sorted(order_book.keys())
            
            horizon_limit = current_time + pd.Timedelta(hours=min(max_horizon_hours, 4))
            
            delivery_times = [t for t in delivery_times if t <= horizon_limit]
            
            if len(delivery_times) < 2:
                continue
            
            # Create time horizon (use hours as indices)
            time_horizon = list(range(len(delivery_times)))
            
            # Map orders to time periods and filter to top orders only
            orders = {}
            max_orders_per_product = 10  # Only consider top 10 buy and top 10 sell orders
            for i, delivery_time in enumerate(delivery_times):
                all_orders = order_book[delivery_time]
                
                # Separate buy and sell orders
                buy_orders = sorted([o for o in all_orders if o.is_buy], 
                                   key=lambda x: x.price, reverse=True)[:max_orders_per_product]
                sell_orders = sorted([o for o in all_orders if not o.is_buy], 
                                    key=lambda x: x.price)[:max_orders_per_product]
                
                # Combine filtered orders
                orders[i] = buy_orders + sell_orders
            
            # Solve optimization
            try:
                solution = self.solver.solve(
                    orders, current_soc, time_horizon,
                    committed_power=None, final_soc=None
                )
                
                if solution['status'] != 'optimal':
                    continue
                
                total_solves += 1
                
                # Extract first-period action
                first_period_soc = solution['soc'].get(0, current_soc)
                first_period_power = solution['power'].get(0, 0.0)
                
                # Calculate actual SoC change
                soc_before = current_soc
                soc_after = first_period_soc
                soc_change = soc_after - soc_before
                
                # Update state
                current_soc = first_period_soc
                
                # Track ACTUAL revenue and costs for first period only
                period_revenue = 0.0
                period_trading_cost = 0.0
                period_degradation = 0.0
                
                if 0 in solution.get('trades', {}):
                    for trade in solution['trades'][0]:
                        quantity = trade['quantity']
                        price = trade['price']
                        
                        if trade['is_buy']:
                            # Market BUY order: buyers want to buy, battery SELLS to them
                            period_revenue += price * quantity
                            period_trading_cost += self.market.trading_fee * quantity
                        else:
                            # Market SELL order: sellers want to sell, battery BUYS from them
                            period_revenue -= price * quantity
                            period_trading_cost += self.market.trading_fee * quantity
                
                # Calculate degradation cost for this period
                # Only count discharge degradation (as per the paper)
                if soc_change < 0:
                    # Discharged energy (positive value)
                    energy_discharged = -soc_change
                    
                    # Calculate degradation using rainflow-equivalent model
                    cycle_depth = energy_discharged / self.battery.energy_capacity
                    life_loss = self.degradation.stress_function(cycle_depth)
                    period_degradation = self.battery.replacement_cost * life_loss
                
                # Accumulate totals
                cumulative_revenue += period_revenue
                cumulative_trading_cost += period_trading_cost
                cumulative_degradation += period_degradation
                
                # Track metrics
                self.soc_history.append({
                    'time': current_time,
                    'soc': current_soc,
                    'power': first_period_power,
                    'soc_change': soc_change
                })
                
                self.profit_history.append({
                    'time': current_time,
                    'revenue': period_revenue,
                    'trading_cost': period_trading_cost,
                    'degradation': period_degradation,
                    'net_profit': period_revenue - period_trading_cost - period_degradation,
                    'cumulative_revenue': cumulative_revenue,
                    'cumulative_trading_cost': cumulative_trading_cost,
                    'cumulative_degradation': cumulative_degradation,
                    'cumulative_net_profit': cumulative_revenue - cumulative_trading_cost - cumulative_degradation
                })
                
                self.degradation_history.append({
                    'time': current_time,
                    'degradation_cost': period_degradation,
                    'energy_discharged': -soc_change if soc_change < 0 else 0.0,
                    'cycle_depth': cycle_depth if soc_change < 0 else 0.0
                })
                
                self.solve_time_history.append(solution['solve_time'])
                
                # Store trades
                if 0 in solution.get('trades', {}):
                    self.trade_history.extend([{
                        'time': current_time,
                        **trade
                    } for trade in solution['trades'][0]])
                
            except Exception as e:
                print(f"Error at {current_time}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compute statistics
        avg_solve_time = np.mean(self.solve_time_history) if self.solve_time_history else 0
        total_solve_time = np.sum(self.solve_time_history)
        
        net_profit = cumulative_revenue - cumulative_trading_cost - cumulative_degradation
        
        results = {
            'gross_revenue': cumulative_revenue,
            'trading_costs': cumulative_trading_cost,
            'degradation_cost': cumulative_degradation,
            'net_profit': net_profit,
            'final_soc': current_soc,
            'total_solves': total_solves,
            'avg_solve_time_ms': avg_solve_time * 1000,
            'total_solve_time_s': total_solve_time,
            'soc_history': self.soc_history,
            'profit_history': self.profit_history,
            'solve_time_history': self.solve_time_history,
            'trade_history': self.trade_history,
            'price_history': self.price_history,
            'degradation_history': self.degradation_history,
        }
        
        return results


# Example usage with CSV data
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/Users/inderpal/Documents/NGU_Project/orderbook_degradation.csv"
    
    print("="*70)
    print("Health-Aware BESS Trading - MILP Implementation")
    print("="*70)
    
    # Initialize with correct parameters from the paper
    battery = BatteryParams(
        power_max=20.0,           # 20 MW
        power_min=-20.0,          # -20 MW
        energy_capacity=12.5,     # 12.5 MWh
        energy_min=1.875,         # 15% of 12.5 MWh
        energy_max=11.875,        # 95% of 12.5 MWh
        eta_charge=0.95,          # 95% efficiency
        eta_discharge=0.95,       # 95% efficiency
        replacement_cost=300000.0,# $300,000/MWh
        num_segments=8,           # Piecewise linear segments (reduced for speed)
        shelf_life=10.0,          # 10 years
        cycle_life=3000           # 3000 cycles at 80% DoD
    )
    
    market = MarketParams(
        trading_fee=0.09,        # €/MWh
        min_trading_unit=0.1,    # MWh
        time_interval=1.0        # hours
    )
    
    degradation = DegradationModel(battery)
    
    # Print degradation model info
    print(f"\nDegradation Model:")
    print(f"  Number of segments: {degradation.J}")
    print(f"  Segment costs (€/MW):")
    for j in range(min(5, degradation.J)):
        print(f"    Segment {j+1}: €{degradation.segment_costs[j]:.2f}/MW")
    if degradation.J > 5:
        print(f"    ... ({degradation.J - 5} more segments)")
    
    # Load order book
    try:
        loader = OrderBookLoader(csv_path)
    except FileNotFoundError:
        print(f"\nError: Could not find {csv_path}")
        print(f"Usage: python {sys.argv[0]} path/to/orderbook.csv")
        sys.exit(1)
    
    # Run backtest
    print("\nRunning backtest with corrected parameters...")
    
    backtest = RollingIntrinsicBacktest(
        battery, market, degradation, phi=0.0
    )
    
    results = backtest.run(
        loader,
        initial_soc=6.25,  # Middle SoC to allow both charge and discharge
        max_horizon_hours=24,
        update_frequency='hourly'
    )
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Battery Specifications:")
    print(f"  Capacity: {battery.energy_capacity} MWh")
    print(f"  Power Rating: ±{battery.power_max} MW")
    print(f"  SoC Range: [{battery.energy_min:.1f}, {battery.energy_max:.1f}] MWh")
    print(f"  Efficiency: {battery.eta_charge*100:.1f}%")
    print(f"  Replacement Cost: ${battery.replacement_cost:,.0f}/MWh")
    print(f"\nPerformance:")
    print(f"  Gross Revenue: €{results['gross_revenue']:,.2f}")
    print(f"  Trading Costs: €{results['trading_costs']:,.2f}")
    print(f"  Degradation Cost: €{results['degradation_cost']:,.2f}")
    print(f"  Net Profit: €{results['net_profit']:,.2f}")
    print(f"  Final SoC: {results['final_soc']:.2f} MWh")
    print(f"  Total Optimizations: {results['total_solves']}")
    print(f"  Avg Solve Time: {results['avg_solve_time_ms']:.2f} ms")
    print(f"{'='*70}")
    
    # Detailed logging to verify degradation logic
    print(f"\n{'='*70}")
    print("DETAILED BATTERY OPERATIONS")
    print(f"{'='*70}")
    print(f"{'Time':<20} {'SoC (MWh)':<12} {'ΔSoC':<10} {'Trades':<8} {'Degrad €':<12}")
    print(f"{'-'*70}")
    
    for i, soc_entry in enumerate(results['soc_history'][:15]):
        time_str = soc_entry['time'].strftime('%Y-%m-%d %H:%M')
        soc = soc_entry['soc']
        soc_change = soc_entry['soc_change']
        
        # Find corresponding profit entry
        profit_entry = results['profit_history'][i] if i < len(results['profit_history']) else None
        degrad = profit_entry['degradation'] if profit_entry else 0.0
        
        # Count trades at this time
        trades_at_time = [t for t in results['trade_history'] if t['time'] == soc_entry['time']]
        num_trades = len(trades_at_time)
        
        arrow = "↑" if soc_change > 0 else "↓" if soc_change < 0 else "→"
        print(f"{time_str:<20} {soc:>6.2f} {arrow}    {soc_change:>+6.3f}   {num_trades:>3}      €{degrad:>8.4f}")
    
    if len(results['soc_history']) > 15:
        print(f"... ({len(results['soc_history']) - 15} more periods)")
    
    print(f"\n{'='*70}")
    print("DEGRADATION VERIFICATION")
    print(f"{'='*70}")
    
    total_discharge_events = sum(1 for entry in results['degradation_history'] if entry['energy_discharged'] > 0)
    total_energy_discharged = sum(entry['energy_discharged'] for entry in results['degradation_history'])
    total_degrad_calculated = sum(entry['degradation_cost'] for entry in results['degradation_history'])
    
    print(f"Discharge Events: {total_discharge_events}")
    print(f"Total Energy Discharged: {total_energy_discharged:.4f} MWh")
    print(f"Total Degradation Cost: €{total_degrad_calculated:.4f}")
    print(f"Match with results: {abs(total_degrad_calculated - results['degradation_cost']) < 0.01}")
    
    if total_discharge_events > 0:
        print(f"\nFirst 5 Discharge Events:")
        print(f"{'Time':<20} {'Energy (MWh)':<15} {'Cycle Depth':<15} {'Cost (€)':<12}")
        print(f"{'-'*70}")
        discharge_count = 0
        for entry in results['degradation_history']:
            if entry['energy_discharged'] > 0 and discharge_count < 5:
                time_str = entry['time'].strftime('%Y-%m-%d %H:%M')
                print(f"{time_str:<20} {entry['energy_discharged']:>10.4f}     {entry['cycle_depth']:>10.4f}     €{entry['degradation_cost']:>8.4f}")
                discharge_count += 1
    else:
        print(f"\n⚠️  NO DISCHARGE EVENTS DETECTED")
        print(f"   Battery is not physically discharging")
    
    print(f"{'='*70}")