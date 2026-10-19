# =====================================================================
# FULL CODE IN ONE CELL
# Multi-objective (cost-time) planning of hybrid AM / casting capacity
#   Part 1: Deterministic Pareto frontiers (6 cases: Model 1/2 x no LT / LT / LT -30%)
#   Part 2: Monte Carlo Experiment A (global cost-time cloud)
#   Part 3: Monte Carlo Experiment B (exceedance curves, common scenarios, Big-M check)
# All CSV files and figures are saved in the folder results/ (listed at the end of the run).
# Data and model logic are taken from MOO-Definitive.ipynb; a random seed is added and
# two rates that increased with volume are corrected (Binder 506 -> 486, PBF 922 -> 906).
# Every number used in the paper is printed, and figures/CSVs are saved.
# =====================================================================
import math
from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------
# 0) SETTINGS
# ---------------------------------------------------------------------
SEED = 42                     # fixes all random draws -> reproducible results
RUN_DETERMINISTIC = True
RUN_EXPERIMENT_A = True
RUN_EXPERIMENT_B = True

N_SIM_A = 100_000             # Experiment A scenarios (original notebook value)
N_SCEN_B = 5_000              # Experiment B scenarios
EPS_B = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000]
CAST_CAP_B = 260              # casting hours per line (same as the deterministic runs)
BIG_M_PENALTY = 1e8           # USD per excess hour in the Big-M check (10e7 in the notebook)
SHOW_TITLES = True            # titles on every plot (set False for journal figures, which use captions)

OUT_DIR = Path('results')     # every CSV and PNG is saved here
OUT_DIR.mkdir(exist_ok=True)
saved_files = []
def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight'); saved_files.append(path)
def save_csv(df, name, index=False):
    path = OUT_DIR / name
    df.to_csv(path, index=index); saved_files.append(path)

plt.rcParams.update({'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 17,
                     'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14})
fmt_h = mt.FuncFormatter(lambda x, _: f"{int(x):,}")
fmt_m = mt.FuncFormatter(lambda y, _: f"{y/1e6:.1f}M")

# ---------------------------------------------------------------------
# 1) BASE DATA (identical to the notebook)
# ---------------------------------------------------------------------
machines    = ['AM_hWAAM', 'AM_Binder', 'AM_slm', 'AM_PBF', 'TM_Casting']
am_machines = ['AM_hWAAM', 'AM_Binder', 'AM_slm', 'AM_PBF']
tm_machines = ['TM_Casting']
parts       = ['part_1', 'part_2']        # part_1 = original, part_2 = topology-optimized

batch_size = {
    ('AM_hWAAM','part_1'):6, ('AM_hWAAM','part_2'):6,
    ('AM_Binder','part_1'):8, ('AM_Binder','part_2'):8,
    ('AM_slm','part_1'):5,    ('AM_slm','part_2'):5,
    ('AM_PBF','part_1'):5,    ('AM_PBF','part_2'):5,
    ('TM_Casting','part_1'):1,('TM_Casting','part_2'):1,
}
base_time = {   # hours per part
    ('AM_hWAAM','part_1'):9.30,  ('AM_hWAAM','part_2'):9.30,
    ('AM_Binder','part_1'):8.74, ('AM_Binder','part_2'):7.74,
    ('AM_slm','part_1'):14.77,   ('AM_slm','part_2'):13.77,
    ('AM_PBF','part_1'):22.86,   ('AM_PBF','part_2'):21.86,
    ('TM_Casting','part_1'):4.5, ('TM_Casting','part_2'):24,
}
base_setup = {'AM_hWAAM':380000, 'AM_Binder':350000, 'AM_slm':400000,
              'AM_PBF':500000, 'TM_Casting':430000}
cap_hours  = {'AM_hWAAM':440, 'AM_Binder':460, 'AM_slm':460,
              'AM_PBF':450, 'TM_Casting':260}
setup_time = {'AM_hWAAM':12, 'AM_Binder':10, 'AM_slm':15, 'AM_PBF':20, 'TM_Casting':50}
lead_time  = {'AM_hWAAM':2*24, 'AM_Binder':2*24, 'AM_slm':3*24, 'AM_PBF':3*24,
              'TM_Casting':21*24}

# Model 1: incremental discounts (marginal rate per segment)
pwl_breakpoints = [0, 25, 50, 100, 200, 400, 800, 1200, 2200]
incr_rates = {
    'AM_hWAAM':   [1000, 950, 903, 879, 855, 831, 808, 784],
    'AM_Binder':  [626, 594, 566, 526, 522, 508, 496, 486],   # last rate corrected (was 506)
    'AM_slm':     [790, 750, 710, 693, 659, 640, 623, 603],
    'AM_PBF':     [960, 912, 906, 902, 814, 796, 774, 756],   # third rate corrected (was 922)
    'TM_Casting': [2500, 2375, 1583, 1188, 991, 792, 595, 396],
}
def cumulative(rates, bps=pwl_breakpoints):
    cum = [0.0]
    for k, r in enumerate(rates):
        cum.append(cum[-1] + (bps[k+1] - bps[k]) * r)
    return cum
pwl_costvals = {m: cumulative(incr_rates[m]) for m in machines}

# Model 2: all-units (bulk) discount brackets
bounds_bulk = {1:(0,25), 2:(25,50), 3:(50,100), 4:(100,200),
               5:(200,400), 6:(400,800), 7:(800,1400), 8:(1400,2200)}
rate_bulk_AM = {1:1000, 2:950, 3:903, 4:879, 5:855, 6:831, 7:808, 8:786}
rate_bulk_TM = {1:2500, 2:2375, 3:1583, 4:1188, 5:938, 6:792, 7:625, 8:417}

B_DET = 1e8                   # deterministic budget (non-binding)
B_MC  = 1e7                   # Monte Carlo budget
demand_values = [10, 30, 55, 80, 110, 160, 250, 300, 450, 600, 700,
                 900, 1000, 1200, 1500, 1700, 1800, 1900, 2100]
eps_values = list(range(300, 10000, 300))   # 300 ... 9,900 h

# ---------------------------------------------------------------------
# 2) DETERMINISTIC MODEL (Model 1 or Model 2, with/without lead time)
# ---------------------------------------------------------------------
def solve_det(demand, eps, model='M1', lead=False, setup_mult_am=1.0):
    setup = {m: base_setup[m] * (setup_mult_am if m in am_machines else 1.0) for m in machines}
    md = gp.Model(); md.Params.OutputFlag = 0
    n   = md.addVars(machines, vtype=GRB.INTEGER, lb=0)
    x   = md.addVars(machines, parts, vtype=GRB.INTEGER, lb=0)
    y   = md.addVars(machines, vtype=GRB.BINARY)
    vol = md.addVars(machines, lb=0)
    vc  = md.addVars(machines, lb=0)

    T = gp.quicksum(x[i,j]*batch_size[i,j]*base_time[i,j] for i in machines for j in parts)
    if lead:
        T += gp.quicksum(y[i]*lead_time[i] + n[i]*setup_time[i] for i in machines)
    C = gp.quicksum(setup[i]*n[i] + vc[i] for i in machines)

    md.setObjective(C, GRB.MINIMIZE)
    md.addConstr(C <= B_DET)
    md.addConstr(T <= eps)
    md.addConstr(gp.quicksum(x[i,j]*batch_size[i,j] for i in machines for j in parts) >= demand)

    bigM = 1e6 if model == 'M1' else 1e8
    for i in machines:
        md.addConstr(vol[i] == gp.quicksum(x[i,j]*batch_size[i,j] for j in parts))
        md.addConstr(gp.quicksum(x[i,j]*batch_size[i,j]*base_time[i,j] for j in parts) <= cap_hours[i]*n[i])
        for j in parts:
            md.addConstr(x[i,j] <= bigM*y[i])
        md.addConstr(n[i] >= y[i])
        if model == 'M1':
            md.addGenConstrPWL(vol[i], vc[i], pwl_breakpoints, pwl_costvals[i])
        else:
            z = md.addVars(range(1, 9), vtype=GRB.BINARY)
            rates = rate_bulk_AM if i in am_machines else rate_bulk_TM
            md.addConstr(gp.quicksum(z[k] for k in range(1, 9)) == 1)
            for k in range(1, 9):
                lb, ub = bounds_bulk[k]
                if k > 1:
                    md.addConstr(vol[i] >= lb + 1 - bigM*(1 - z[k]))
                md.addConstr(vol[i] <= ub + bigM*(1 - z[k]))
            md.addConstr(vc[i] == gp.quicksum(rates[k]*vol[i]*z[k] for k in range(1, 9)))
    md.optimize()
    if md.status != GRB.OPTIMAL:
        return None
    vA = sum(vol[i].X for i in am_machines); vT = sum(vol[i].X for i in tm_machines)
    return {'Demand': demand, 'Epsilon': eps, 'Total_Cost': C.getValue(), 'Total_Time': T.getValue(),
            'vol_AM': vA, 'vol_TM': vT,
            'n_AM': sum(n[i].X for i in am_machines), 'n_TM': sum(n[i].X for i in tm_machines),
            'Tech': 'AM Dominant' if vA > vT else 'TM Dominant'}

def pareto_filter(df):
    keep = []
    for D, g in df.groupby('Demand'):
        g = g.drop_duplicates(subset=['Total_Cost', 'Total_Time'])
        c, t = g['Total_Cost'].to_numpy(), g['Total_Time'].to_numpy()
        dom = [np.any((c <= c[k]) & (t <= t[k]) & ((c < c[k]) | (t < t[k]))) for k in range(len(g))]
        keep.append(g[~np.array(dom)])
    return pd.concat(keep, ignore_index=True)

def plot_pareto(df_p, fname, title):
    col = {'AM Dominant': 'orange', 'TM Dominant': 'blue'}
    fig, ax = plt.subplots(figsize=(10, 6))
    for D, g in df_p.groupby('Demand'):
        g = g.sort_values('Total_Time')
        ax.scatter(g['Total_Time'], g['Total_Cost'], c=g['Tech'].map(col), s=80,
                   edgecolor='k', linewidth=0.5, zorder=3)
        for k in range(len(g) - 1):
            ax.plot(g['Total_Time'].iloc[k:k+2], g['Total_Cost'].iloc[k:k+2],
                    color=col[g['Tech'].iloc[k]], lw=1.5, zorder=2)
        ax.text(g['Total_Time'].iloc[-1] + 100, g['Total_Cost'].iloc[-1], f"D={D}", fontsize=13, va='center')
    ax.legend(handles=[mpatches.Patch(color='orange', label='AM Dominant'),
                       mpatches.Patch(color='blue', label='TM Dominant')], loc='lower right')
    ax.set_xlabel('Total Time (Hrs)'); ax.set_ylabel('Total Cost (M USD)')
    ax.xaxis.set_major_formatter(fmt_h); ax.yaxis.set_major_formatter(fmt_m); ax.grid(True)
    if SHOW_TITLES: ax.set_title(title)
    plt.tight_layout(); save_fig(fig, fname); plt.show()

if RUN_DETERMINISTIC:
    case_titles = {
        'M1 noLT':  'Pareto frontiers: Model 1 (incremental), no lead time',
        'M1 LT':    'Pareto frontiers: Model 1 (incremental), with lead time',
        'M1 LT-30': 'Pareto frontiers: Model 1, lead time, AM setup cost -30%',
        'M2 noLT':  'Pareto frontiers: Model 2 (bulk), no lead time',
        'M2 LT':    'Pareto frontiers: Model 2 (bulk), with lead time',
        'M2 LT-30': 'Pareto frontiers: Model 2, lead time, AM setup cost -30%',
    }
    cases = {  # name: (model, lead, AM setup multiplier, figure file)
        'M1 noLT':  ('M1', False, 1.0, 'pareto_incremental.png'),
        'M1 LT':    ('M1', True,  1.0, 'pareto_incremental_lead_time.png'),
        'M1 LT-30': ('M1', True,  0.7, 'pareto_incremental_lead_time_reduced_cost.png'),
        'M2 noLT':  ('M2', False, 1.0, 'pareto_bulk.png'),
        'M2 LT':    ('M2', True,  1.0, 'pareto_bulk_lead_time.png'),
        'M2 LT-30': ('M2', True,  0.7, 'pareto_bulk_lead_time_reduced_cost.png'),
    }
    det, frontier, machines_tab = {}, {}, {}
    for name, (mod, lead, mult, fname) in cases.items():
        rows = [r for D in demand_values for e in eps_values
                if (r := solve_det(D, e, mod, lead, mult)) is not None]
        df = pd.DataFrame(rows); det[name] = df
        df_p = pareto_filter(df); frontier[name] = df_p
        machines_tab[name] = df.groupby('Demand')[['n_AM', 'n_TM']].mean().apply(np.ceil).astype(int)
        save_csv(df,   f"det_{name.replace(' ', '_')}_all.csv")
        save_csv(df_p, f"det_{name.replace(' ', '_')}_pareto.csv")
        plot_pareto(df_p, fname, case_titles[name])

    # --- summary per case: cheapest vs fastest Pareto point ---
    summ = {}
    for name, df_p in frontier.items():
        out = []
        for D, g in df_p.groupby('Demand'):
            ch = g.loc[g['Total_Cost'].idxmin()]; fa = g.loc[g['Total_Time'].idxmin()]
            mixed = ((g['vol_AM'] > 0) & (g['vol_TM'] > 0)).sum()
            out.append({'D': D, 'points': len(g), 'mixed': mixed,
                        'C_cheap': ch.Total_Cost, 'T_cheap': ch.Total_Time, 'tech_cheap': ch.Tech,
                        'TMvol_cheap': ch.vol_TM, 'C_fast': fa.Total_Cost, 'T_fast': fa.Total_Time,
                        'premium_%': 100*(fa.Total_Cost/ch.Total_Cost - 1),
                        'time_cut_%': 100*(1 - fa.Total_Time/ch.Total_Time),
                        'cost_per_part': ch.Total_Cost / D})
        summ[name] = pd.DataFrame(out).set_index('D')
        print(f"\n===== {name}: cheapest vs fastest Pareto point =====")
        print(summ[name].round(2).to_string())
        save_csv(summ[name], f"det_{name.replace(' ', '_')}_summary.csv", index=True)

    print("\n===== 30% AM setup-cost reduction (cheapest plan) =====")
    for base, red in [('M1 LT', 'M1 LT-30'), ('M2 LT', 'M2 LT-30')]:
        for D in [450, 600, 700, 900, 1000, 1200]:
            a, b = summ[base].loc[D, 'C_cheap'], summ[red].loc[D, 'C_cheap']
            print(f"{base:6s} D={D:5d}: {a/1e6:.2f}M -> {b/1e6:.2f}M  ({100*(b/a-1):+.1f}%)")

    print("\n===== Model 2 vs Model 1 (cheapest plan) =====")
    for lt in ['noLT', 'LT']:
        for D in [250, 600, 1000, 1200, 1500]:
            a, b = summ[f'M1 {lt}'].loc[D, 'C_cheap'], summ[f'M2 {lt}'].loc[D, 'C_cheap']
            print(f"{lt:5s} D={D:5d}: M1 {a/1e6:.2f}M  M2 {b/1e6:.2f}M  ({100*(b/a-1):+.1f}%)")

    print("\n===== Machine counts (mean over feasible eps, rounded up) =====")
    mtab = pd.concat({k: v for k, v in machines_tab.items()}, axis=1)
    print(mtab.to_string())
    save_csv(mtab, 'det_machine_counts.csv', index=True)

# ---------------------------------------------------------------------
# 3) MONTE CARLO HELPERS
# ---------------------------------------------------------------------
rng = np.random.default_rng(SEED)
def trunc_normal(mu, sigma, low):
    return max(rng.normal(mu, sigma), low)

def solve_mc(D, eps, setup, times, costvals, caps, budget, slack_penalty=None):
    md = gp.Model(); md.Params.OutputFlag = 0
    x   = md.addVars(machines, parts, vtype=GRB.INTEGER, lb=0)
    n   = md.addVars(machines, vtype=GRB.INTEGER, lb=0)
    vol = md.addVars(machines, lb=0)
    vc  = md.addVars(machines, lb=0)
    T = gp.quicksum(x[i,j]*batch_size[i,j]*times[i,j] for i in machines for j in parts)
    C = gp.quicksum(setup[i]*n[i] + vc[i] for i in machines)
    s = None
    if slack_penalty is None:
        md.setObjective(C, GRB.MINIMIZE); md.addConstr(T <= eps)
    else:
        s = md.addVar(lb=0, vtype=GRB.INTEGER)
        md.setObjective(C + slack_penalty*s, GRB.MINIMIZE); md.addConstr(T <= eps + s)
    md.addConstr(C <= budget)
    md.addConstr(gp.quicksum(x[i,j]*batch_size[i,j] for i in machines for j in parts) >= D)
    for i in machines:
        md.addConstr(vol[i] == gp.quicksum(x[i,j]*batch_size[i,j] for j in parts))
        md.addConstr(gp.quicksum(x[i,j]*batch_size[i,j]*times[i,j] for j in parts) <= caps[i]*n[i])
        md.addGenConstrPWL(vol[i], vc[i], pwl_breakpoints, costvals[i])
    md.optimize()
    if md.status != GRB.OPTIMAL:
        return {'Status': 'INFEASIBLE'}
    vT = vol['TM_Casting'].X; vTot = sum(vol[i].X for i in machines)
    out = {'Status': 'OPTIMAL', 'Cost': C.getValue(), 'Time': T.getValue(),
           'Casting_share': vT / vTot if vTot > 0 else 0.0}
    if s is not None:
        out['TimeSlack'] = s.X; out['CostWithPenalty'] = C.getValue() + slack_penalty*s.X
    return out

def exceedance_plots(df, title, fname_prefix, logscale=False, cost_col='Cost'):
    p_levels = [0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.90, 0.95]
    fig, ax = plt.subplots(figsize=(10, 6))
    allc = df[cost_col].to_numpy(float)
    grid = np.linspace(np.percentile(allc, 1), np.percentile(allc, 99), 200)
    for eps, g in df.groupby('Epsilon'):
        c = g[cost_col].to_numpy(float)
        ax.plot(grid, [(c > v).mean() for v in grid], marker='.', label=f"Time = {eps}")
    ax.set_xlabel('Cost Threshold (USD)'); ax.set_ylabel('P(Cost > threshold)')
    if SHOW_TITLES: ax.set_title(f"Exceedance Curves {title}")
    ax.grid(True); ax.legend(title='Time Budget')
    if logscale: ax.set_xscale('log')
    plt.tight_layout(); save_fig(fig, f"{fname_prefix}_exceedance.png"); plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    eps_sorted = sorted(df['Epsilon'].unique())
    for p in p_levels:
        q = [np.quantile(df.loc[df['Epsilon'] == e, cost_col], 1 - p) for e in eps_sorted]
        ax.plot(eps_sorted, q, marker='o', label=f"{int(p*100)}% exceed")
    ax.set_xlabel('Time Budget ε (hrs)'); ax.set_ylabel('Cost c such that P(Cost > c) = p (USD)')
    if SHOW_TITLES: ax.set_title(f"Cost vs Time Budget {title}")
    ax.grid(True); ax.legend(title='Exceedance Level')
    if logscale: ax.set_yscale('log')
    plt.tight_layout(); save_fig(fig, f"{fname_prefix}_thresholds.png"); plt.show()

def print_quantiles(df, label, csv_name, cost_col='Cost'):
    print(f"\n===== Quantiles: {label} =====")
    rows = []
    for eps, g in df.groupby('Epsilon'):
        c = g[cost_col].to_numpy(float)
        med, p95 = np.median(c), np.quantile(c, 0.95)
        rows.append({'Epsilon': eps, 'n': len(c), 'median': med, 'P95': p95, 'P95_over_median': p95/med})
        print(f"eps={eps:5d}  n={len(c):5d}  median={med:,.0f}  P95={p95:,.0f}  P95/median={p95/med:.2f}")
    save_csv(pd.DataFrame(rows), csv_name)

# ---------------------------------------------------------------------
# 4) EXPERIMENT A: global cost-time cloud
#    eps ~ U(300, 10000); one flat sampled unit rate per technology (no discounts)
# ---------------------------------------------------------------------
if RUN_EXPERIMENT_A:
    r_A, mean_A = 10, 1000
    p_A = r_A / (r_A + mean_A)
    base_rate = {m: incr_rates[m][0] for m in machines}
    recs = []
    for sim in range(N_SIM_A):
        D   = rng.negative_binomial(r_A, p_A)
        eps = rng.uniform(300, 10000)
        setup = {m: trunc_normal(base_setup[m], 0.20*base_setup[m], 1) for m in machines}
        times = {k: trunc_normal(t, 0.15*t, 0.1) for k, t in base_time.items()}
        costvals = {}
        for m in machines:
            rate = trunc_normal(base_rate[m], 0.20*base_rate[m], 0.1)
            costvals[m] = cumulative([rate]*8)
        sol = solve_mc(D, eps, setup, times, costvals, cap_hours, B_MC)
        if sol['Status'] == 'OPTIMAL':
            recs.append({'Demand': D, 'Epsilon': eps, 'Cost': sol['Cost'], 'Time': sol['Time']})
    df_A = pd.DataFrame(recs); save_csv(df_A, 'mc_experiment_A.csv')
    print(f"\n===== Experiment A: {len(df_A):,} feasible of {N_SIM_A:,} scenarios =====")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_A['Time'], df_A['Cost'], c='steelblue', alpha=0.3, s=20, edgecolor='none')
    ax.set_xlabel('Total Time (Hrs)'); ax.set_ylabel('Total Cost (USD)'); ax.grid(True)
    ax.xaxis.set_major_formatter(fmt_h); ax.yaxis.set_major_formatter(fmt_m)
    if SHOW_TITLES: ax.set_title('Monte Carlo Experiment A: cost vs. time (feasible scenarios)')
    plt.tight_layout(); save_fig(fig, 'general_montecarlo.png'); plt.show()

# ---------------------------------------------------------------------
# 5) EXPERIMENT B: fixed time budgets with common random numbers
#    The same scenarios are used for the hard-constraint and Big-M runs.
# ---------------------------------------------------------------------
if RUN_EXPERIMENT_B:
    r_B, mean_B = 5, 500
    p_B = r_B / (r_B + mean_B)
    caps_B = dict(cap_hours); caps_B['TM_Casting'] = CAST_CAP_B
    scenarios = []
    for s in range(N_SCEN_B):
        scenarios.append({
            'Demand': rng.negative_binomial(r_B, p_B),
            'setup':  {m: trunc_normal(base_setup[m], 0.20*base_setup[m], 1) for m in machines},
            'times':  {k: trunc_normal(t, 0.15*t, 0.1) for k, t in base_time.items()},
        })
    recs_hard, recs_soft = [], []
    for s, sc in enumerate(scenarios):
        for eps in EPS_B:
            h = solve_mc(sc['Demand'], eps, sc['setup'], sc['times'], pwl_costvals, caps_B, B_MC)
            recs_hard.append({'scenario': s, 'Epsilon': eps, 'Demand': sc['Demand'], **h})
            k = solve_mc(sc['Demand'], eps, sc['setup'], sc['times'], pwl_costvals, caps_B, 1e9,
                         slack_penalty=BIG_M_PENALTY)
            recs_soft.append({'scenario': s, 'Epsilon': eps, 'Demand': sc['Demand'], **k})
    df_raw  = pd.DataFrame(recs_hard); save_csv(df_raw, 'mc_experiment_B_hard.csv')
    df_soft = pd.DataFrame(recs_soft); save_csv(df_soft, 'mc_experiment_B_bigM.csv')
    df_crn  = df_raw[df_raw['Status'] == 'OPTIMAL'].copy()

    n_tot, n_ok = len(df_raw), len(df_crn)
    print(f"\n===== Experiment B feasibility =====")
    print(f"Runs: {n_tot:,}   optimal: {n_ok:,} ({100*n_ok/n_tot:.1f}%)   infeasible: {n_tot-n_ok:,} ({100*(n_tot-n_ok)/n_tot:.1f}%)")
    print(df_raw.assign(ok=df_raw['Status'] == 'OPTIMAL').groupby('Epsilon')['ok'].mean().mul(100).round(1)
          .rename('% feasible').to_string())
    print("\nAverage casting share of parts by eps:")
    print(df_crn.groupby('Epsilon')['Casting_share'].mean().mul(100).round(1).to_string())

    # pooled (feasible at each budget)
    print_quantiles(df_crn, 'pooled, feasible runs at each budget', 'mc_B_quantiles_pooled.csv')
    exceedance_plots(df_crn, '(feasible runs)', 'exc_pooled')

    # common scenarios (feasible at every budget)
    ok = df_crn.groupby('scenario')['Epsilon'].nunique()
    common_ids = ok[ok == len(EPS_B)].index
    df_common = df_crn[df_crn['scenario'].isin(common_ids)].copy()
    print(f"\nCommon scenarios: {len(common_ids):,} of {df_crn['scenario'].nunique():,} scenarios feasible at least once")
    print_quantiles(df_common, 'common scenarios (feasible for all eps)', 'mc_B_quantiles_common.csv')
    exceedance_plots(df_common, '(scenarios feasible for all ε)', 'exc_common')

    # Big-M check (all scenarios, penalized cost)
    df_soft_ok = df_soft[df_soft['Status'] == 'OPTIMAL'].copy()
    print(f"\nBig-M runs solved: {len(df_soft_ok):,} of {len(df_soft):,}")
    print(df_soft_ok.assign(slack=df_soft_ok['TimeSlack'] > 0).groupby('Epsilon')['slack'].mean().mul(100).round(1)
          .rename('% runs using slack').to_string())
    print_quantiles(df_soft_ok, 'Big-M, penalized cost', 'mc_B_quantiles_bigM.csv', cost_col='CostWithPenalty')
    exceedance_plots(df_soft_ok, '(Big-M penalty, log scale)', 'exc_bigM', logscale=True, cost_col='CostWithPenalty')

    # scatter: feasible runs only (no slack)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_crn['Epsilon'], df_crn['Cost'], c='steelblue', alpha=0.3, s=20, edgecolor='none')
    ax.set_xlabel('Time Budget ε (hrs)'); ax.set_ylabel('Total Cost (USD)'); ax.grid(True)
    ax.xaxis.set_major_formatter(fmt_h); ax.yaxis.set_major_formatter(fmt_m)
    if SHOW_TITLES: ax.set_title('Feasible runs only (no slack)')
    plt.tight_layout(); save_fig(fig, 'feasibles.png'); plt.show()

    # scatter: all runs with Big-M penalty
    fig, ax = plt.subplots(figsize=(10, 6))
    used = df_soft_ok['TimeSlack'] > 0
    ax.scatter(df_soft_ok.loc[~used, 'Epsilon'], df_soft_ok.loc[~used, 'CostWithPenalty'],
               c='steelblue', alpha=0.3, s=20, label='No slack')
    ax.scatter(df_soft_ok.loc[used, 'Epsilon'], df_soft_ok.loc[used, 'CostWithPenalty'],
               c='crimson', alpha=0.3, s=20, label='Slack used')
    ax.set_yscale('log'); ax.set_xlabel('Time Budget ε (hrs)'); ax.set_ylabel('Total Cost (USD)')
    ax.legend(); ax.grid(True); ax.xaxis.set_major_formatter(fmt_h)
    if SHOW_TITLES: ax.set_title('All runs with Big-M penalty (log scale)')
    plt.tight_layout(); save_fig(fig, 'bigM_scatter.png'); plt.show()

print(f"\nDone. {len(saved_files)} files saved in: {OUT_DIR.resolve()}")
for f in saved_files:
    print("  ", f.name)
