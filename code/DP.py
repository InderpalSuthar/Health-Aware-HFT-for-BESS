"""
Health-Aware Dynamic Programming (DP) Solver for BESS Trading
Approximates the MILP formulation using the method from:
"Health-Aware High-Frequency Trading for BESS: A Multi-Objective DP Approach" (Section 5)
"""

import os
import ctypes
import time
import numpy as np
from typing import Dict, List, Optional
from ctypes import Structure, c_double, c_int, POINTER
from MILP import BatteryParams, MarketParams, DegradationModel, Order, RollingIntrinsicBacktest

# CTypes Definitions matching algorithm3.cpp
class CBatteryParams(Structure):
    _fields_ = [
        ("power_max", c_double),
        ("power_min", c_double),
        ("energy_capacity", c_double),
        ("energy_min", c_double),
        ("energy_max", c_double),
        ("eta_charge", c_double),
        ("eta_discharge", c_double),
        ("replacement_cost", c_double),
        ("num_segments", c_int)
    ]

class CMarketParams(Structure):
    _fields_ = [
        ("trading_fee", c_double),
        ("min_trading_unit", c_double),
        ("time_interval", c_double)
    ]

class COrder(Structure):
    _fields_ = [
        ("price", c_double),
        ("quantity", c_double),
        ("is_buy", c_int)
    ]

class CTimeStepData(Structure):
    _fields_ = [
        ("num_buy_orders", c_int),
        ("num_sell_orders", c_int),
        ("buy_orders", COrder * 50),
        ("sell_orders", COrder * 50)
    ]

class HealthAwareDP:
    """
    Dynamic Programming solver for health-aware intrinsic optimization.
    Implements Algorithm 3 from the paper.
    Uses C++ backend if available.
    """
    
    def __init__(self, battery: BatteryParams, market: MarketParams, 
                 degradation: DegradationModel, 
                 state_grid_size: int = 201,  
                 action_discretization: int = 1,
                 phi: float = 0.0): 
        self.battery = battery
        self.market = market
        self.degradation = degradation
        self.m = state_grid_size
        self.v = action_discretization
        self.phi = phi
        
        # Precompute state grid
        self.s_min = battery.energy_min
        self.s_max = battery.energy_max
        self.delta_s = (self.s_max - self.s_min) / (self.m - 1)
        self.output_G = np.linspace(self.s_min, self.s_max, self.m)
        self.segment_costs = self.degradation.segment_costs
        self.delta_e = self.battery.energy_capacity / self.degradation.J
        
        # Load C++ Library
        self.lib = None
        try:
            lib_path = os.path.join(os.path.dirname(__file__), "libdpc.so")
            if os.path.exists(lib_path):
                self.lib = ctypes.CDLL(lib_path)
                self.lib.run_dp.argtypes = [
                    CBatteryParams, CMarketParams, c_int, c_int, c_double, c_double,
                    POINTER(CTimeStepData), POINTER(c_double), 
                    POINTER(c_int), POINTER(c_double)
                ]
                self.lib.run_dp.restype = None
                print("Loaded C++ optimized DP solver backend.")
            else:
                print("C++ library not found, falling back to Python (slow).")
        except Exception as e:
            print(f"Failed to load C++ library: {e}")

    def solve(self, orders: Dict[int, List[Order]], initial_soc: float,
              time_horizon: List[int], committed_power: Optional[Dict[int, float]] = None,
              final_soc: Optional[float] = None,
              initial_segment_energy: Optional[Dict[int, float]] = None) -> Dict:
        """
        Main DP Solver execution.
        """
        start_time = time.time()
        T = len(time_horizon)
        M = self.market.time_interval
        u = self.market.min_trading_unit
        
        policy = np.zeros((T, self.m), dtype=np.int32)
        
        # Prepare Data for C++
        if self.lib:
            timeline_data = (CTimeStepData * T)()
            
            # Use pre-processed liquidity logic to populate structs
            liquidity = {} # Need this for forward pass later anyway
            
            for t_idx, t in enumerate(time_horizon):
                step_orders = orders.get(t, [])
                buy_qs = [o for o in step_orders if o.is_buy]
                sell_qs = [o for o in step_orders if not o.is_buy]
                
                # Sort for priority: Buys (Highest Price First), Sells (Lowest Price First)
                buy_qs.sort(key=lambda x: x.price, reverse=True)
                sell_qs.sort(key=lambda x: x.price)
                
                liquidity[t_idx] = (sum(o.quantity for o in sell_qs), sum(o.quantity for o in buy_qs), buy_qs, sell_qs)
                
                # Copy to C Struct
                # Note: MAX_ORDERS = 50 fixed in C++
                data = timeline_data[t_idx]
                data.num_buy_orders = min(len(buy_qs), 50)
                data.num_sell_orders = min(len(sell_qs), 50)
                
                for i in range(data.num_buy_orders):
                    data.buy_orders[i].price = buy_qs[i].price
                    data.buy_orders[i].quantity = buy_qs[i].quantity
                    data.buy_orders[i].is_buy = 1
                    
                for i in range(data.num_sell_orders):
                    data.sell_orders[i].price = sell_qs[i].price
                    data.sell_orders[i].quantity = sell_qs[i].quantity
                    data.sell_orders[i].is_buy = 0
            
            # Params
            c_batt = CBatteryParams(
                self.battery.power_max, self.battery.power_min,
                self.battery.energy_capacity, self.battery.energy_min, self.battery.energy_max,
                self.battery.eta_charge, self.battery.eta_discharge,
                self.battery.replacement_cost, self.battery.num_segments
            )
            c_mkt = CMarketParams(
                self.market.trading_fee, self.market.min_trading_unit, self.market.time_interval
            )
            
            # Buffers
            segment_costs_arr = (c_double * len(self.segment_costs))(*self.segment_costs)
            policy_flat = np.zeros(T * self.m, dtype=np.int32)
            policy_ptr = policy_flat.ctypes.data_as(POINTER(c_int))
            obj_val_c = c_double(0.0)
            
            # CALL C++ with PHI
            self.lib.run_dp(
                c_batt, c_mkt, T, self.m, initial_soc, c_double(self.phi),
                timeline_data, segment_costs_arr,
                policy_ptr, ctypes.byref(obj_val_c)
            )
            
            # Reshape policy
            policy = policy_flat.reshape((T, self.m))
            obj_val = obj_val_c.value
            
        else:
            # PYTHON FALLBACK (Simplified/Old Logic) - Ignore Phi for simplicity as user uses C++
            liquidity = {}
            for t_idx, t in enumerate(time_horizon):
                buy_orders = [o for o in orders.get(t, []) if o.is_buy]
                sell_orders = [o for o in orders.get(t, []) if not o.is_buy]
                liquidity[t_idx] = (sum(o.quantity for o in sell_orders), sum(o.quantity for o in buy_orders), buy_orders, sell_orders)
            
            # Initialize V
            raise RuntimeError("C++ Library required for high performance (and Phi penalty) mode.")

        # ========== Forward Pass (Common) ==========
        current_soc = initial_soc
        schedule_trades = {}
        schedule_soc = {}
        schedule_power = {}
        total_deg = 0.0
        
        for t_idx in range(T):
            t_real = time_horizon[t_idx]
            s_idx = (np.abs(self.output_G - current_soc)).argmin()
            best_k = policy[t_idx, s_idx]
            a_optimal = best_k * u
            
            if a_optimal > 0:
                s_next = current_soc + a_optimal * self.battery.eta_charge
            elif a_optimal < 0:
                s_next = current_soc + a_optimal / self.battery.eta_discharge
            else:
                s_next = current_soc
            
            schedule_soc[t_real] = s_next 
            schedule_power[t_real] = a_optimal / M 
            
            trades = []
            if best_k != 0:
                max_buy_vol, max_sell_vol, buy_orders, sell_orders = liquidity[t_idx]
                if best_k > 0: 
                    rem = a_optimal
                    sorted_asks = sorted(sell_orders, key=lambda x: x.price)
                    for order in sorted_asks:
                        filled = min(rem, order.quantity)
                        if filled > 0:
                            trades.append({'quantity': filled, 'price': order.price, 'is_buy': False})
                            rem -= filled
                        if rem <= 1e-9: break
                else: 
                    rem = abs(a_optimal)
                    sorted_bids = sorted(buy_orders, key=lambda x: x.price, reverse=True)
                    for order in sorted_bids:
                        filled = min(rem, order.quantity)
                        if filled > 0:
                            trades.append({'quantity': filled, 'price': order.price, 'is_buy': True})
                            rem -= filled
                        if rem <= 1e-9: break
            
            schedule_trades[t_real] = trades
            total_deg += self.calculate_degradation_cost(current_soc, a_optimal)
            current_soc = s_next
            
        solve_time = time.time() - start_time
        
        solution = {
                'status': 'optimal',
                'objective': obj_val if self.lib else 0.0,
                'solve_time': solve_time,
                'trades': schedule_trades,
                'soc': schedule_soc,
                'power': schedule_power,
                'segment_energy': {t: {} for t in time_horizon}, 
                'segment_discharge': {t: {} for t in time_horizon},
                'degradation_cost': total_deg
            }
            
        return solution

    # Helper methods (calculate_degradation_cost, transition) kept for Forward Pass usage
    def calculate_degradation_cost(self, s: float, a: float) -> float:
        """
        Algorithm 2: Degradation Cost Computation
        """
        if a >= 0:
            return 0.0
            
        W = abs(a) 
        J = self.degradation.J
        costs = self.segment_costs
        
        val = s / self.delta_e
        nt = int(val)
        ft = val - nt
        
        C_deg = 0.0
        W_rem = W
        
        for j in range(J):
            if j < nt:
                e_j = self.delta_e
            elif j == nt:
                e_j = ft * self.delta_e
            else:
                e_j = 0.0
                
            d_j = min(e_j, W_rem)
            C_deg += costs[j] * d_j
            W_rem -= d_j
            if W_rem <= 1e-9:
                break
                
        return C_deg

    def transition(self, s: float, a: float) -> float:
        if a > 0:
            return s + a * self.battery.eta_charge
        elif a < 0:
            return s + a / self.battery.eta_discharge 
        else:
            return s

class DPRollingIntrinsicBacktest(RollingIntrinsicBacktest):
    """Subclass to inject DP solver"""
    def __init__(self, battery, market, degradation, phi=0.0, dp_grid_size=201):
        super().__init__(battery, market, degradation, phi)
        # Override the solver with DP and pass Phi
        self.solver = HealthAwareDP(battery, market, degradation, state_grid_size=dp_grid_size, phi=phi)
