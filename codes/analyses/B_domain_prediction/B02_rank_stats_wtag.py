#!/usr/bin/env python3
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse,re

# Helpers
def _layer_from_tag(tag:str) -> str:
    """Map a tag string to layer label L1/L2/L3; return 'none' if not matched."""
    if not isinstance(tag,str) or not tag:
        return "none"
    head = tag.split("_",1)[0].upper()  # e.g., L1_exact -> 'L1'
    return head if head in {"L1","L2","L3"} else "none"

def _pick_most_severe_layer(tags: pd.Series) -> str:
    """
    If multiple tags exist for the same (Cluster, Original position),
    pick the most severe layer by priority: L1 < L2 < L3 < none.
    """
    priority = {"L1":1,"L2":2,"L3":3,"none":4}
    # Map each tag to a layer then to priority, and choose the min
    layers = tags.map(_layer_from_tag)
    best = layers.map(priority).min()
    # If all were NaN, default to 'none'
    if pd.isna(best):
        return "none"
    # Map back to layer
    inv = {v:k for k,v in priority.items()}
    return inv.get(int(best),"none")

def compute_rank_stats(
    df: pd.DataFrame,
    rank_col: str = "True token rank",
    layer_col: str = "layer",
    layers_order = ("L1","L2","L3","none"),
) -> pd.DataFrame:
    """
    Compute rank statistics by layer and overall total.

    Stats:
      - mean,std,median,min,max of ranks
      - #1 ratio:         P(rank == 1)
      - within #10 ratio: P(rank <= 10)
      - above #1000 ratio P(rank > 1000)
    """
    def _one(group:pd.Series) -> pd.Series:
        g = group.dropna().astype(float)
        n = len(g)
        if n == 0:
            return pd.Series({
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
                "#1_ratio": np.nan,
                "≤10_ratio": np.nan,
                ">1000_ratio": np.nan,
                "#1_%": np.nan,
                "≤10_%": np.nan,
                ">1000_%": np.nan,
            })
        one = (g == 1).mean()
        le10 = (g <= 10).mean()
        gt1000 = (g > 1000).mean()
        return pd.Series({
            "n": n,
            "mean": g.mean(),
            "std": g.std(ddof=1) if n > 1 else 0.0,
            "median": g.median(),
            "min": g.min(),
            "max": g.max(),
            "#1_ratio": one,
            "≤10_ratio": le10,
            ">1000_ratio": gt1000,
            "#1_%": one * 100.0,
            "≤10_%": le10 * 100.0,
            ">1000_%": gt1000 * 100.0,
        })

    if rank_col not in df.columns:
        raise KeyError(f"rank_col '{rank_col}' not in DataFrame.")
    if layer_col not in df.columns:
        raise KeyError(f"layer_col '{layer_col}' not in DataFrame.")

    per_layer = (
        df.groupby(layer_col,dropna=False)[rank_col]
          .apply(_one)
          .unstack(1)
    )

    # Reindex to desired order, keep unexpected labels at the end just in case
    present = [x for x in layers_order if x in per_layer.index]
    unexpected = [x for x in per_layer.index if x not in layers_order]
    per_layer = per_layer.loc[present + unexpected]

    total_row = _one(df[rank_col])
    total_row.name = "total"

    cols_order = [
        "n", "mean", "std", "median", "min", "max",
        "#1_ratio", "≤10_ratio", ">1000_ratio",
        "#1_%", "≤10_%", ">1000_%",
    ]
    out = pd.concat([per_layer, total_row.to_frame().T], axis=0)
    out = out.reindex(columns=cols_order)

    return out.round({
        "mean": 3, "std": 3, "median": 3, "min": 0, "max": 0,
        "#1_ratio": 4, "≤10_ratio": 4, ">1000_ratio": 4,
        "#1_%": 2, "≤10_%": 2, ">1000_%": 2,
    })

# Main
def main():
    ap = argparse.ArgumentParser(description="Output statistics of prediction")
    ap.add_argument("--pred",required=True,help="Predicted result csv")
    ap.add_argument("--leak",required=True,help="Probe layer csv")
    ap.add_argument("--out",default="./",help="Output directory")
    args = ap.parse_args()

    name = Path(args.pred).name
    m = re.match(r"^every_domain_prediction\.(.+)\.csv$",name)
    title = m.group(1)

    # Read inputs
    df_pred = pd.read_csv(args.pred)
    df_leak = pd.read_csv(args.leak)

    # Basic column checks
    pred_cols = {"Cluster","Original position","True token rank"}
    missing = pred_cols - set(df_pred.columns)
    if missing:
        raise KeyError(f"pred missing columns: {missing}")

    leak_cols = {"Cluster","Original position","tag"}
    missing = leak_cols - set(df_leak.columns)
    if missing:
        raise KeyError(f"leak missing columns: {missing}")

    # Build per-(Cluster,Original position) layer from leakage
    # If multiple rows exist per position, pick most severe layer.
    pos_layer = (
        df_leak.groupby(["Cluster","Original position"])["tag"]
               .apply(_pick_most_severe_layer)
               .rename("layer")
               .reset_index()
    )

    # Merge layer onto predictions; positions not present in leakage -> 'none'
    df = df_pred.merge(pos_layer,on=["Cluster","Original position"],how="left")
    df["layer"] = df["layer"].fillna("none")

    # Compute stats
    stats = compute_rank_stats(df,rank_col="True token rank",layer_col="layer")

    # Save outputs
    outdir = Path(args.out)
    stats.to_csv(outdir / f"o02.rank_stats.{title}.csv")
    df.to_csv(outdir / f"o02.every_domain_prediction_wtag.{title}.csv",index=False)

    return stats,df

if __name__ == "__main__":
    stats_df,merged_df = main()
    print(stats_df.to_string())

#---end---

