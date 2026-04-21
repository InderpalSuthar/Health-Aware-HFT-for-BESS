/*********************************************
 * Battery Intraday Trading Model
 *********************************************/

// 1. SETS & PARAMETERS
int NumOrders = ...;
range ORDERS = 1..NumOrders;

// Order data arrays
string side[ORDERS] = ...;
string product[ORDERS] = ...;
float price[ORDERS] = ...;
float quantity[ORDERS] = ...;

// Extract unique products and sort chronologically
{string} PRODUCTS = {product[i] | i in ORDERS};
sorted {string} PRODUCTS_SORTED = PRODUCTS;

// BATTERY PARAMETERS
float mtu = ...;
float f_max = ...;
float f_min = ...;
float s0 = ...;
float s_max = ...;
float nu = ...;
float eta_plus = ...;
float eta_minus = ...;

// Initial flows
float f0[p in PRODUCTS_SORTED] = 0.0;

// Time ordering
int T = card(PRODUCTS_SORTED);
range TimeIndex = 1..T;
string productAtTime[t in TimeIndex] = item(PRODUCTS_SORTED, t-1);

// 2. DECISION VARIABLES
dvar float+ q[ORDERS];
dvar int+ k[ORDERS];
dvar float+ f_plus[PRODUCTS_SORTED];
dvar float+ f_minus[PRODUCTS_SORTED];
dvar float f[PRODUCTS_SORTED];
dvar float+ i_t[PRODUCTS_SORTED];
dvar float+ w_t[PRODUCTS_SORTED];
dvar boolean alpha[PRODUCTS_SORTED];
dvar float+ s[PRODUCTS_SORTED];

// 3. OBJECTIVE FUNCTION
maximize
  sum(t in PRODUCTS_SORTED) (
    sum(o in ORDERS: product[o] == t && side[o] == "BUY") (price[o] - nu) * q[o]
    -
    sum(o in ORDERS: product[o] == t && side[o] == "SELL") (price[o] + nu) * q[o]
  );

// 4. CONSTRAINTS
subject to {
  forall(i in ORDERS) {
    q[i] <= quantity[i];
    q[i] == k[i] * mtu;
    k[i] >= 0;
  }

  forall(t in PRODUCTS_SORTED) {
    f_plus[t] == sum(o in ORDERS: product[o] == t && side[o] == "BUY") q[o];
    f_minus[t] == sum(o in ORDERS: product[o] == t && side[o] == "SELL") q[o];

    f[t] == f0[t] + f_plus[t] - f_minus[t];
    f_min <= f[t] <= f_max;
    f[t] == i_t[t] - w_t[t];

    i_t[t] <= alpha[t] * f_max;
    w_t[t] <= (1 - alpha[t]) * (-f_min);

    i_t[t] >= 0;
    w_t[t] >= 0;
    0 <= s[t] <= s_max;
  }

  s[productAtTime[1]] == s0 + eta_plus * i_t[productAtTime[1]] - (1/eta_minus) * w_t[productAtTime[1]];

  forall(idx in 2..T) {
    s[productAtTime[idx]] == s[productAtTime[idx-1]] + eta_plus * i_t[productAtTime[idx]] - (1/eta_minus) * w_t[productAtTime[idx]];
  }
}
execute {
  cplex.threads = 10;  // Use all available cores (default)
}