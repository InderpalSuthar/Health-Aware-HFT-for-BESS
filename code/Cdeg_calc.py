import numpy as np

class DegradationCostCalculator:
    """
    Implements the Degradation Cost Computation (Algorithm 2) 
    from the Health-Aware High-Frequency Trading for BESS paper.
    """
    
    def __init__(self, energy_capacity=12.5, num_segments=16, 
                 replacement_cost=300000.0, eta_discharge=0.95):
        """
        Initialize the calculator with battery parameters.
        
        Args:
            energy_capacity (float): E_rate, total energy capacity in MWh.
            num_segments (int): J, number of cycle depth segments.
            replacement_cost (float): R, replacement cost in Euro/MWh.
            eta_discharge (float): eta_dis, discharging efficiency.
        """
        self.E_rate = energy_capacity
        self.J = num_segments
        self.R = replacement_cost
        self.eta_dis = eta_discharge
        
        # Calculate segment width (Delta e)
        self.delta_e = self.E_rate / self.J
        
        # Precompute segment marginal costs (c_j)
        self.c_j = self._precompute_segment_costs()
        
    def _phi(self, delta: float) -> float:
        """Cycle depth stress function for Li(NiMnCo)O2 cells."""
        return 5.24e-4 * (delta ** 2.03)

    def _precompute_segment_costs(self) -> np.ndarray:
        """
        Precomputes the marginal degradation cost for each segment (c_j).
        Returns an array of length J.
        """
        costs = np.zeros(self.J)
        for j in range(1, self.J + 1):
            delta_upper = j / self.J
            delta_lower = (j - 1) / self.J
            
            stress_diff = self._phi(delta_upper) - self._phi(delta_lower)
            costs[j-1] = (self.R * self.J / (self.eta_dis * self.E_rate)) * stress_diff
            
        return costs

    def compute_cost(self, s_t: float, W: float) -> float:
        """
        Computes the degradation cost for discharging energy W at State of Charge s_t.
        Implements the exact logic of Algorithm 2 using W_rem.
        
        Args:
            s_t (float): Current State of Charge in MWh.
            W (float): Discharge energy requested in MWh.
            
        Returns:
            float: The total degradation cost for this discharge action.
        """
        n_t = int(s_t // self.delta_e)
        f_t = (s_t / self.delta_e) - n_t
        
        C_deg = 0.0
        W_rem = W
        
        for j in range(1, self.J + 1):
            idx = j - 1  
            
            if j <= n_t:
                e_j = self.delta_e
            elif j == n_t + 1:
                e_j = f_t * self.delta_e
            else:
                e_j = 0.0
                
            d_j = min(e_j, W_rem)
            
            C_deg += self.c_j[idx] * d_j
            W_rem -= d_j
            
            if W_rem <= 1e-9:  
                break
                
        if W_rem > 1e-9:
            print(f"Warning: Not enough energy in battery to discharge {W} MWh. Shortfall: {W_rem} MWh")
            
        return C_deg


if __name__ == "__main__":
    calc = DegradationCostCalculator(energy_capacity=10.0, num_segments=10)
    
    s_t = 6.6
    W = 0.8
    
    total_cost = calc.compute_cost(s_t, W)
    print(f"Total Degradation Cost: €{total_cost:.2f}")
