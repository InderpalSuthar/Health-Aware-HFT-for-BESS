$title Battery Intraday Trading Model

*********************************************
* 1. DECLARATIONS
*********************************************

* Declare Sets first (but do not define them yet)
Sets
    o       "Orders"
    t       "Products / Time Steps"
    sides   "Sides (BUY/SELL)"
    map_ot(o,t)      "Mapping: Order o belongs to time t"
    map_oside(o,sides) "Mapping: Order o is side BUY/SELL"
;

* Declare Parameters
Parameters
    mtu, f_max, f_min, s0, s_max, nu, eta_plus, eta_minus
    price(o)     "Price of order"
    quantity(o)  "Max quantity of order"
    f0(t)        "Initial flows (default 0)"
;

*********************************************
* 2. DATA IMPORT (THE FIX)
*********************************************
* We use $include instead of $gdxin because it is a text file
* Make sure battery_data.inc is in the same folder as this .gms file
$include battery_data.inc

* Initialize derived data
f0(t) = 0;

*********************************************
* 3. DECISION VARIABLES
*********************************************

Variables
    z            "Objective Value"
    q(o)         "Accepted quantity"
    f_plus(t)    "Total Buy flow"
    f_minus(t)   "Total Sell flow"
    f(t)         "Net flow"
    i_t(t)       "Injection (Charging)"
    w_t(t)       "Withdrawal (Discharging)"
    s(t)         "State of Charge"
;

Integer Variable k(o) "Integer blocks of mtu";
Binary Variable alpha(t) "Charging state (1=Charge, 0=Discharge)";
Positive Variable q, f_plus, f_minus, i_t, w_t, s;

*********************************************
* 4. EQUATIONS
*********************************************

Equations
    eq_obj              "Objective Function"
    eq_q_limit(o)       "Limit quantity"
    eq_q_mtu(o)         "Discrete blocks constraint"
    
    eq_f_plus(t)        "Sum of Buy orders"
    eq_f_minus(t)       "Sum of Sell orders"
    
    eq_flow_bal(t)      "Flow balance"
    eq_flow_min(t)      "Min flow limit"
    eq_flow_max(t)      "Max flow limit"
    eq_flow_decomp(t)   "Decompose flow into Inj/With"
    
    eq_inj_logic(t)     "Injection logic with alpha"
    eq_with_logic(t)    "Withdrawal logic with alpha"
    
    eq_soc_first(t)     "SoC for first time step"
    eq_soc_dyn(t)       "SoC dynamics"
    eq_soc_max(t)       "SoC capacity limit"
;

* --- Objective ---
* Maximize: (Price - nu)*q for BUYs  MINUS  (Price + nu)*q for SELLs
eq_obj.. 
    z =e= sum(t, 
            sum(o$(map_ot(o,t) and map_oside(o,'BUY')),  (price(o) - nu) * q(o)) 
          - sum(o$(map_ot(o,t) and map_oside(o,'SELL')), (price(o) + nu) * q(o))
          );

* --- Order Constraints ---
eq_q_limit(o)..    q(o) =l= quantity(o);
eq_q_mtu(o)..      q(o) =e= k(o) * mtu;

* --- Flow Aggregation ---
eq_f_plus(t)..     f_plus(t)  =e= sum(o$(map_ot(o,t) and map_oside(o,'BUY')), q(o));
eq_f_minus(t)..    f_minus(t) =e= sum(o$(map_ot(o,t) and map_oside(o,'SELL')), q(o));

* --- Flow Balance & Limits ---
eq_flow_bal(t)..   f(t) =e= f0(t) + f_plus(t) - f_minus(t);
eq_flow_min(t)..   f(t) =g= f_min;
eq_flow_max(t)..   f(t) =l= f_max;
eq_flow_decomp(t).. f(t) =e= i_t(t) - w_t(t);

* --- Logical Constraints ---
eq_inj_logic(t)..  i_t(t) =l= alpha(t) * f_max;
* Note: using abs(f_min) to ensure positive RHS
eq_with_logic(t).. w_t(t) =l= (1 - alpha(t)) * abs(f_min);

* --- Storage Dynamics ---
* ord(t)=1 is the first time step in the sorted set t
eq_soc_first(t)$(ord(t)=1).. 
    s(t) =e= s0 + eta_plus * i_t(t) - (1/eta_minus) * w_t(t);

* For subsequent time steps (t-1 refers to previous element)
eq_soc_dyn(t)$(ord(t)>1)..   
    s(t) =e= s(t-1) + eta_plus * i_t(t) - (1/eta_minus) * w_t(t);

eq_soc_max(t)..    s(t) =l= s_max;

*********************************************
* 5. SOLVE
*********************************************

Option threads = 10;
Option mip = CPLEX;
Option limrow = 0, limcol = 0;

Model BatteryModel /all/;

Solve BatteryModel using MIP maximizing z;

Display z.l, q.l, s.l, f.l;
