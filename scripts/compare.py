"""Aggregate per-run metrics into results.{csv,md,png}.

Reads every ``<runs_dir>/<model>/metrics.csv`` (last row) + ``meta.json``,
sorts by GAUC descending, and writes a CSV table, a Markdown table, and a
grouped AUC/GAUC bar chart.

Usage:
  PYTHONPATH=src python scripts/compare.py --runs-dir experiments/runs --out-dir experiments
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def run(runs_dir: Path, out_csv: Path, out_md: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for run in sorted(Path(runs_dir).iterdir()):
        m = run / "metrics.csv"; meta = run / "meta.json"
        if not (m.exists() and meta.exists()):
            continue
        with open(m) as f:
            last = list(csv.DictReader(f))[-1]
        md = json.loads(meta.read_text())
        rows.append({
            "model": run.name,
            "params_backbone": int(md["params_backbone"]),
            "params_emb":      int(md["params_emb"]),
            "mflops_per_sample": float(md["mflops_per_sample"]),
            "train_time_sec":  float(md["train_time_sec"]),
            "AUC": float(last["auc"]),
            "GAUC": float(last["gauc"]),
            "LogLoss": float(last["logloss"]),
        })
    if not rows:
        raise ValueError(
            f"No valid runs found under {runs_dir} (each run needs metrics.csv + meta.json)."
        )
    rows.sort(key=lambda r: r["GAUC"], reverse=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    md_lines = ["| model | backbone | mflops | AUC | GAUC | LogLoss | train_s |",
                "|---|---|---|---|---|---|---|"]
    for r in rows:
        md_lines.append(f"| {r['model']} | {r['params_backbone']:,} | {r['mflops_per_sample']:.1f} "
                        f"| {r['AUC']:.4f} | {r['GAUC']:.4f} | {r['LogLoss']:.4f} | {r['train_time_sec']:.0f} |")
    Path(out_md).write_text("\n".join(md_lines))
    fig, ax = plt.subplots(figsize=(9,4))
    labels = [r["model"] for r in rows]
    auc = [r["AUC"] for r in rows]; gauc = [r["GAUC"] for r in rows]
    x = range(len(labels)); w = 0.35
    ax.bar([i-w/2 for i in x], auc, w, label="AUC")
    ax.bar([i+w/2 for i in x], gauc, w, label="GAUC")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=20); ax.legend()
    ax.set_ylim(0.5, max(max(auc), max(gauc)) + 0.02); ax.set_title("RecFlow CTR — AUC / GAUC")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    run(Path(a.runs_dir), out / "results.csv", out / "results.md", out / "results.png")

if __name__ == "__main__":
    main()
