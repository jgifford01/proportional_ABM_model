import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

# --------------------------- Style ---------------------------
sns.set(style="whitegrid")
fontlabelsize = 10
linewidth = 1.5

# --------------------------- Step 1: Define Parameters and File Paths ---------------------------
agent_number = 200
random_seed = 3145

data_dir = f"data/{agent_number}agents_seed{random_seed}"
plots_dir = f"plots/{agent_number}agents_seed{random_seed}"
os.makedirs(plots_dir, exist_ok=True)

# --------------------------- Helpers ---------------------------
def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def read_tv0_collapsed(path):
    """
    Read watervalue-style file (P, TV0), coerce numeric, collapse duplicates by mean TV0 for each P.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if not {"P", "TV0"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns ['P','TV0']. Found: {list(df.columns)}")

    df = _to_numeric(df, ["P", "TV0"]).dropna(subset=["P", "TV0"])
    df = df.groupby("P", as_index=False).agg(TV0=("TV0", "mean")).sort_values("P").reset_index(drop=True)
    return df

def safe_divide(numer, denom):
    """
    elementwise division with denom==0 -> NaN (avoids warnings)
    """
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.full_like(numer, np.nan, dtype=float)
    mask = denom != 0
    out[mask] = numer[mask] / denom[mask]
    return out

def collapse_trading(path, count_col):
    """
    Read trading file (P, count_col), coerce numeric, collapse duplicates by mean per P.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if not {"P", count_col}.issubset(df.columns):
        raise ValueError(f"{path} must have columns ['P','{count_col}']. Found: {list(df.columns)}")

    df = _to_numeric(df, ["P", count_col]).dropna(subset=["P", count_col])
    df = df.groupby("P", as_index=False).agg(**{count_col: (count_col, "mean")}).sort_values("P").reset_index(drop=True)
    return df

def align_to_grid(dfP, dfY, P_grid):
    """
    Nearest-neighbor alignment of (dfP, dfY) to P_grid.
    """
    P_src = np.asarray(dfP, dtype=float)
    Y_src = np.asarray(dfY, dtype=float)
    idx = np.abs(P_src[:, None] - P_grid[None, :]).argmin(axis=0)
    return Y_src[idx]

# --------------------------- Step 2: Load and Process Data ---------------------------

# ---- 2.1 Load GFT mean outputs (headered) ----
gft_path = os.path.join(data_dir, "GFT_final_mean_array.csv")
gft_df = pd.read_csv(gft_path)
gft_df.columns = [c.strip() for c in gft_df.columns]

required = [
    "Central Planning Mean",
    "Smart Market Mean",
    "Smart Market Proportional Mean",
]
missing = [c for c in required if c not in gft_df.columns]
if missing:
    raise ValueError(f"Missing columns in {gft_path}: {missing}\nFound: {list(gft_df.columns)}")

gft_df = _to_numeric(gft_df, required)

GFT_cp  = gft_df["Central Planning Mean"].to_numpy(dtype=float)
GFT_sm  = gft_df["Smart Market Mean"].to_numpy(dtype=float)
GFT_smp = gft_df["Smart Market Proportional Mean"].to_numpy(dtype=float)

n = len(GFT_cp)
P_grid = np.linspace(0, 1, n)

# ---- 2.2 Load pre-trade values (collapse duplicates) ----
wv_path = os.path.join(data_dir, "watervalue.csv")
wv_prop_path = os.path.join(data_dir, "watervalue_SM_proportional.csv")

wv = read_tv0_collapsed(wv_path)
wv_prop = read_tv0_collapsed(wv_prop_path)

TV0 = align_to_grid(wv["P"].to_numpy(), wv["TV0"].to_numpy(), P_grid)
TV0_prop = align_to_grid(wv_prop["P"].to_numpy(), wv_prop["TV0"].to_numpy(), P_grid)

# Benchmark for normalization: pre-trade value at full availability (δ=1)
TVP1 = TV0[-1]

# ---- 2.3 Compute ratios (safe against TV0==0) ----
GFT_cp_over_TVP1  = safe_divide(GFT_cp,  TVP1)
GFT_sm_over_TVP1  = safe_divide(GFT_sm,  TVP1)
GFT_smp_over_TVP1 = safe_divide(GFT_smp, TVP1)

GFT_cp_over_TV0   = safe_divide(GFT_cp,  TV0)
GFT_sm_over_TV0   = safe_divide(GFT_sm,  TV0)
GFT_smp_over_TV0  = safe_divide(GFT_smp, TV0_prop)

# ---- 2.4 Trading counts (collapse duplicates) ----
cp_trade_path = os.path.join(data_dir, "num_trading_agents_CPP.csv")
sm_trade_path = os.path.join(data_dir, "num_trading_agents_SM.csv")
smp_trade_path = os.path.join(data_dir, "num_trading_agents_SM_proportional.csv")

cp_trade = collapse_trading(cp_trade_path, "num_trading_agents_CPP")
sm_trade = collapse_trading(sm_trade_path, "num_trading_agents_SM")
smp_trade = collapse_trading(smp_trade_path, "num_trading_agents_SM_proportional")

cp_frac = align_to_grid(cp_trade["P"].to_numpy(), cp_trade["num_trading_agents_CPP"].to_numpy(), P_grid) / agent_number
sm_frac = align_to_grid(sm_trade["P"].to_numpy(), sm_trade["num_trading_agents_SM"].to_numpy(), P_grid) / agent_number
smp_frac = align_to_grid(smp_trade["P"].to_numpy(), smp_trade["num_trading_agents_SM_proportional"].to_numpy(), P_grid) / agent_number

# ---- 2.5 Final value (post-trade): TV1 = TV0 + GFT ----
TV1_cp = TV0 + GFT_cp
TV1_sm = TV0 + GFT_sm
TV1_smp = TV0_prop + GFT_smp

# --------------------------- NORMALIZE right two panels by TV0(δ=1) ---------------------------
TV0_norm = safe_divide(TV0, TVP1)
TV0_prop_norm = safe_divide(TV0_prop, TVP1)

TV1_cp_norm = safe_divide(TV1_cp, TVP1)
TV1_sm_norm = safe_divide(TV1_sm, TVP1)
TV1_smp_norm = safe_divide(TV1_smp, TVP1)

# --------------------------- Step 3: Create Combined Figure (1x5) ---------------------------
fig, axes = plt.subplots(1, 5, figsize=(10.6, 2.8), constrained_layout=True)

# Canonical legend labels/colors you want everywhere
label_map = {
    "CPP": {"color": "#A0A09F"},
    "SM": {"color": "#F6DB8C"},
    "SM proportional": {"color": "#4C70D4"},
    "CPP/SM": {"color": "#4D4D4D"},
}

# --------------------------- Panel 1: GFT / TV(δ=1) ---------------------------
ax1 = axes[0]
sns.lineplot(x=P_grid, y=GFT_cp_over_TVP1,  label="CPP",            color=label_map["CPP"]["color"], ax=ax1, legend=False, linewidth=linewidth)
sns.lineplot(x=P_grid, y=GFT_sm_over_TVP1,  label="SM",             color=label_map["SM"]["color"], ax=ax1, legend=False, linewidth=linewidth)
sns.lineplot(x=P_grid, y=GFT_smp_over_TVP1, label="SM proportional", color=label_map["SM proportional"]["color"], ax=ax1, legend=False, linewidth=linewidth)

ax1.set_xlim(0, 1)
ax1.set_xticks(np.linspace(0, 1, num=6))
ax1.set_xticklabels([f"{x:.1f}" for x in ax1.get_xticks()], fontsize=fontlabelsize)
ax1.tick_params(axis="y", labelsize=fontlabelsize, pad=0)
ax1.set_xlabel("Water availability index (δ)", color="black", fontsize=fontlabelsize)
ax1.set_title("GFT / TV0(δ=1)", fontsize=fontlabelsize)
ax1.grid(True, color="lightgray", linestyle="-", linewidth=0.5)

# --------------------------- Panel 2: GFT / TV0(δ) ---------------------------
ax2 = axes[1]
sns.lineplot(x=P_grid, y=GFT_cp_over_TV0,  label="CPP",            color=label_map["CPP"]["color"], ax=ax2, legend=False, linewidth=linewidth)
sns.lineplot(x=P_grid, y=GFT_sm_over_TV0,  label="SM",             color=label_map["SM"]["color"], ax=ax2, legend=False, linewidth=linewidth)
sns.lineplot(x=P_grid, y=GFT_smp_over_TV0, label="SM proportional", color=label_map["SM proportional"]["color"], ax=ax2, legend=False, linewidth=linewidth)

ax2.set_xlim(0, 1)
ax2.set_xticks(np.linspace(0, 1, num=6))
ax2.set_xticklabels([f"{x:.1f}" for x in ax2.get_xticks()], fontsize=fontlabelsize)
ax2.tick_params(axis="y", labelsize=fontlabelsize, pad=0)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax2.set_yticks(np.linspace(0, 2, num=6))
ax2.set_xlabel("Water availability index (δ)", color="black", fontsize=fontlabelsize)
ax2.set_title("GFT / TV0(δ)", fontsize=fontlabelsize)
ax2.grid(True, color="lightgray", linestyle="-", linewidth=0.5)

# --------------------------- Panel 3: Fraction trading ---------------------------
ax3 = axes[2]
sns.lineplot(x=P_grid, y=cp_frac,  label="CPP",            color=label_map["CPP"]["color"], ax=ax3, legend=False, linewidth=linewidth)
sns.lineplot(x=P_grid, y=sm_frac,  label="SM",             color=label_map["SM"]["color"], ax=ax3, legend=False, linewidth=linewidth)
sns.lineplot(x=P_grid, y=smp_frac, label="SM proportional", color=label_map["SM proportional"]["color"], ax=ax3, legend=False, linewidth=linewidth)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1.2)
ax3.set_xticks(np.linspace(0, 1, num=6))
ax3.set_xticklabels([f"{x:.1f}" for x in ax3.get_xticks()], fontsize=fontlabelsize)
ax3.tick_params(axis="y", labelsize=fontlabelsize, pad=0)
ax3.set_xlabel("Water availability index (δ)", color="black", fontsize=fontlabelsize)
ax3.set_title("Fraction of agents trading", fontsize=fontlabelsize)
ax3.grid(True, color="lightgray", linestyle="-", linewidth=0.5)

# --------------------------- Panel 4: Pre-trade value | δ (normalized) ---------------------------
ax4 = axes[3]
sns.lineplot(x=P_grid, y=TV0_norm,      label="CPP/SM",           color=label_map["CPP/SM"]["color"], ax=ax4, legend=False, linewidth=linewidth)
sns.lineplot(x=P_grid, y=TV0_prop_norm, label="SM proportional",  color=label_map["SM proportional"]["color"], ax=ax4, legend=False, linewidth=linewidth)

ax4.set_xlim(0, 1)
ax4.set_xticks(np.linspace(0, 1, num=6))
ax4.set_xticklabels([f"{x:.1f}" for x in ax4.get_xticks()], fontsize=fontlabelsize)
ax4.tick_params(axis="y", labelsize=fontlabelsize, pad=0)
ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax4.set_xlabel("Water availability index (δ)", color="black", fontsize=fontlabelsize)
ax4.set_title("TV0(δ) / TV0(δ=1)", fontsize=fontlabelsize)
ax4.grid(True, color="lightgray", linestyle="-", linewidth=0.5)

# --------------------------- Panel 5: Final value | δ (normalized) ---------------------------
ax5 = axes[4]
sns.lineplot(x=P_grid, y=TV1_cp_norm,  label="CPP",            color=label_map["CPP"]["color"], ax=ax5, legend=False, linewidth=linewidth)
sns.lineplot(x=P_grid, y=TV1_sm_norm,  label="SM",             color=label_map["SM"]["color"], ax=ax5, legend=False, linewidth=linewidth)
sns.lineplot(x=P_grid, y=TV1_smp_norm, label="SM proportional", color=label_map["SM proportional"]["color"], ax=ax5, legend=False, linewidth=linewidth)

ax5.set_xlim(0, 1)
ax5.set_xticks(np.linspace(0, 1, num=6))
ax5.set_xticklabels([f"{x:.1f}" for x in ax5.get_xticks()], fontsize=fontlabelsize)
ax5.tick_params(axis="y", labelsize=fontlabelsize, pad=0)
ax5.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax5.set_xlabel("Water availability index (δ)", color="black", fontsize=fontlabelsize)
ax5.set_title("TV1(δ) / TV0(δ=1)", fontsize=fontlabelsize)
ax5.grid(True, color="lightgray", linestyle="-", linewidth=0.5)

# --------------------------- Shared Legend (ONLY: CPP, SM, SM proportional, CPP/SM) ---------------------------
want = ["CPP", "SM", "SM proportional", "CPP/SM"]
handle_by_label = {}

for ax in axes:
    h, l = ax.get_legend_handles_labels()
    for hh, ll in zip(h, l):
        if ll in want and ll not in handle_by_label:
            handle_by_label[ll] = hh

handles = [handle_by_label[k] for k in want if k in handle_by_label]
labels  = [k for k in want if k in handle_by_label]

fig.legend(
    handles, labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=fontlabelsize,
    bbox_to_anchor=(0.5, -0.20),
)

# --------------------------- Spines ---------------------------
for ax in axes:
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    ax.tick_params(axis="y", pad=0)

plt.tight_layout()

# Give extra room at bottom so legend isn't clipped
plt.subplots_adjust(left=0.0, right=1, wspace=0.25, bottom=0.28)

# --------------------------- Save ---------------------------
png_path = os.path.join(plots_dir, "5_panel_CPP_SM_SMP_TV0_TV1_normed.png")
svg_path = os.path.join(plots_dir, "5_panel_CPP_SM_SMP_TV0_TV1_normed.svg")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(svg_path, format="svg", dpi=1000, bbox_inches="tight")

print(f"Saved: {png_path}")
print(f"Saved: {svg_path}")
print(f"Mean fraction trading (CPP): {np.nanmean(cp_frac):.3f}")
print(f"Max fraction trading (CPP):  {np.nanmax(cp_frac):.3f}")
