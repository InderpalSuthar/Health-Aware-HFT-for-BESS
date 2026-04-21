#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

// Market limits
#define MAX_ORDERS 50
#define INF 1.0e9

extern "C" {

struct BatteryParams {
    double power_max;
    double power_min;
    double energy_capacity;
    double energy_min;
    double energy_max;
    double eta_charge;
    double eta_discharge;
    double replacement_cost;
    int num_segments;
};

struct MarketParams {
    double trading_fee;
    double min_trading_unit;
    double time_interval;
};

struct Order {
    double price;
    double quantity;
    int is_buy; // 1 = buy, 0 = sell
};

// Represents one time step's order book
struct TimeStepData {
    int num_buy_orders;
    int num_sell_orders;
    Order buy_orders[MAX_ORDERS];
    Order sell_orders[MAX_ORDERS];
};

struct DPResult {
    double objective;
    double solve_time_ms;
    double final_soc;
    // We can add arrays for schedule if needed, but for now we just return summary
    // Python will reconstruct schedule by running a forward pass using the policy
    // Or we return the policy array. Returning policy is best.
};

// --- Helper Functions ---

double interpolate(double x, const std::vector<double>& x_grid, const std::vector<double>& y_grid) {
    // x_grid is sorted
    if (x <= x_grid.front()) return y_grid.front();
    if (x >= x_grid.back()) return y_grid.back();
    
    // Binary search for interval
    auto it = std::lower_bound(x_grid.begin(), x_grid.end(), x);
    int idx = std::distance(x_grid.begin(), it) - 1;
    
    double x0 = x_grid[idx];
    double x1 = x_grid[idx+1];
    double y0 = y_grid[idx];
    double y1 = y_grid[idx+1];
    
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

double calculate_degradation_cost(double s, double a, const BatteryParams& batt, 
                                  const std::vector<double>& segment_costs, double delta_e) {
    if (a >= 0) return 0.0;
    
    double W = std::abs(a);
    int J = batt.num_segments;
    
    double val = s / delta_e;
    int nt = (int)val;
    double ft = val - nt;
    
    double cost = 0.0;
    double W_rem = W;
    
    for (int j = 0; j < J; ++j) {
        double e_j = 0.0;
        if (j < nt) e_j = delta_e;
        else if (j == nt) e_j = ft * delta_e;
        
        double d_j = std::min(e_j, W_rem);
        cost += segment_costs[j] * d_j;
        W_rem -= d_j;
        
        if (W_rem <= 1e-9) break;
    }
    return cost;
}

// --- Main Solver ---

// Returns policy array pointer. Caller must free.
// policy[t * m + s_idx] contains optimal k
void run_dp(
    BatteryParams batt,
    MarketParams market,
    int T,
    int m,
    double initial_soc,
    double phi,                  // Parametric penalty factor
    TimeStepData* timeline_data, // Array of size T
    double* segment_costs_arr,   // Array of size J
    int* out_policy,             // Output buffer of size T * m
    double* out_objective        // Output scalar
) {
    // 1. Setup Grid
    std::vector<double> G(m);
    double delta_s = (batt.energy_max - batt.energy_min) / (m - 1);
    for (int i = 0; i < m; ++i) {
        G[i] = batt.energy_min + i * delta_s;
    }
    
    std::vector<double> segment_costs(segment_costs_arr, segment_costs_arr + batt.num_segments);
    double delta_e = batt.energy_capacity / batt.num_segments;
    
    // Value Function V[current][s_idx] and V[next][s_idx]
    std::vector<double> V_next(m, 0.0); // V_{T+1} = 0
    std::vector<double> V_curr(m);
    
    double M = market.time_interval;
    double u = market.min_trading_unit;
    double nu = market.trading_fee;
    
    // 2. Backward Pass
    for (int t = T - 1; t >= 0; --t) {
        TimeStepData& lob = timeline_data[t];
        
        // Calculate Spread for Phi Penalty
        // Spread = (BestAsk - BestBid) / MidPrice? 
        // Paper says: delta_{t*,t} = (P_ask - P_bid) / ((P_ask + P_bid)/2) (Relative spread)
        // Or absolute spread?
        // Eq 12 in first paper (p10): delta = (P_ask - P_bid) / AvgPrice.
        // Eq on p25 of second paper: "delta defined as the big-ask spread". 
        // Usually absolute spread is P_ask - P_bid. 
        // Let's use Relative Spread as per Eq 12/p10 definition.
        // Note: Prices can be negative! Relative spread might blow up or be negative.
        // Paper 2 p25 says "Phi... acts as penalty for products with large bid-ask spread".
        // "delta defined as the big-ask spread". Typo for Bid-Ask?
        // Let's assume Absolute Spread if Relative is unstable, but Eq 12 is explicit: Relative.
        // Warning: if AvgPrice ~ 0, Relative Spread -> Infinity.
        // Robust implementation: Absolute Spread is safer for negative prices.
        // But let's check Eq 12 again. (P_ask - P_bid) / ((P_ask + P_bid)/2).
        // If prices are 50, 40 -> Spread = 10 / 45 = 0.22.
        // If prices are -10, -20 -> Spread = 10 / -15 = -0.66?
        // A penalty should be positive. phi >= 0. |a| >= 0.
        // If delta is negative, penalty becomes reward!
        // Spread is always positive (Ask >= Bid).
        // Start simple: Absolute Spread. P_ask - P_bid.
        // Wait, text says "delta defined as the big-ask spread". "big" -> "bid"?
        // Eq 12 explicitly defines relative.
        // "delta = (Pass - Pbid) / ((Pask + Pbid)/2)".
        // I will implement Absolute Spread for stability unless prices are consistently > 0.
        // Given battery trading involves 0 prices, Relative is risky.
        // However, if the paper uses Relative, I should try.
        // Let's compute Best Ask / Best Bid.
        
        double best_ask = INF;
        double best_bid = -INF;
        
        // Find best ask (lowest sell price)
        for(int i=0; i<lob.num_sell_orders; ++i) {
            if (lob.sell_orders[i].price < best_ask) best_ask = lob.sell_orders[i].price;
        }
        // Find best bid (highest buy price)
        for(int i=0; i<lob.num_buy_orders; ++i) {
            if (lob.buy_orders[i].price > best_bid) best_bid = lob.buy_orders[i].price;
        }
        
        double spread_penalty_factor = 0.0;
        if (best_ask < INF && best_bid > -INF && phi > 1e-6) {
            double spread = best_ask - best_bid;
            // Use Relative Spread if Average Price is substantial, else Absolute
            double avg_price = (best_ask + best_bid) / 2.0;
            if (std::abs(avg_price) > 1.0) {
                spread_penalty_factor = spread / std::abs(avg_price); // Use Abs avg to keep spread pos
            } else {
                spread_penalty_factor = spread; // Fallback
            }
        }

        // Sum volumes
        double max_buy_vol = 0; // Buying from sellers
        for (int i=0; i<lob.num_sell_orders; ++i) max_buy_vol += lob.sell_orders[i].quantity;
        
        double max_sell_vol = 0; // Selling to buyers
        for (int i=0; i<lob.num_buy_orders; ++i) max_sell_vol += lob.buy_orders[i].quantity;
        
        // Optimize for each state
        #pragma omp parallel for
        for (int s_idx = 0; s_idx < m; ++s_idx) {
            double s = G[s_idx];
            
            // Feasible actions logic
            double E_ch_max = M * batt.power_max;
            double E_dis_max = M * (-batt.power_min);
            
            double a_max_e = (batt.energy_max - s) / batt.eta_charge;
            double a_min_e_mag = (s - batt.energy_min) * batt.eta_discharge;
            
            double limit_pos = std::min({E_ch_max, a_max_e, max_buy_vol});
            int k_max = (int)(limit_pos / u);
            
            double limit_neg = std::min({E_dis_max, a_min_e_mag, max_sell_vol});
            int k_min = -(int)(limit_neg / u);
            
            double best_val = -INF;
            int best_k = 0;
            
            // Iterate k
            for (int k = k_min; k <= k_max; ++k) {
                double a = k * u;
                
                // Profit
                double profit = 0.0;
                if (k > 0) { // Buying
                    double rem = a;
                    double cost = 0;
                    for(int i=0; i<lob.num_sell_orders; ++i) {
                        double filled = std::min(rem, lob.sell_orders[i].quantity);
                        cost += filled * (lob.sell_orders[i].price + nu);
                        rem -= filled;
                        if (rem <= 1e-9) break;
                    }
                    profit = -cost;
                } else if (k < 0) { // Selling
                    double rem = std::abs(a);
                    double rev = 0;
                    for(int i=0; i<lob.num_buy_orders; ++i) {
                        double filled = std::min(rem, lob.buy_orders[i].quantity);
                        rev += filled * (lob.buy_orders[i].price - nu);
                        rem -= filled;
                        if (rem <= 1e-9) break;
                    }
                    profit = rev;
                }
                
                // Apply Phi Penalty (Parametric Enhancement)
                // pi_hat = pi - (phi * delta * |a|)
                if (phi > 1e-6 && k != 0) {
                    profit -= phi * spread_penalty_factor * std::abs(a);
                }
                
                // Deg Cost
                double dc = calculate_degradation_cost(s, a, batt, segment_costs, delta_e);
                
                // Transition
                double s_next_val;
                if (a > 0) s_next_val = s + a * batt.eta_charge;
                else if (a < 0) s_next_val = s + a / batt.eta_discharge;
                else s_next_val = s;
                
                // Value
                double v_fut = interpolate(s_next_val, G, V_next);
                double total = profit - dc + v_fut;
                
                if (total > best_val) {
                    best_val = total;
                    best_k = k;
                }
            }
            
            V_curr[s_idx] = best_val;
            out_policy[t * m + s_idx] = best_k;
        }
        
        // Move to next step
        V_next = V_curr;
    }
    
    // Compute objective for initial state
    *out_objective = interpolate(initial_soc, G, V_next);
}

} // extern "C"
