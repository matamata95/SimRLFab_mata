import html
import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# DATA_DIR = Path("./log/80 states throughput test/log_visualization")
# OUTPUT_HTML = DATA_DIR / "dashboard.html"

COLORS = {
    "blue":   "#2563eb",
    "cyan":   "#0891b2",
    "green":  "#16a34a",
    "amber":  "#d97706",
    "red":    "#dc2626",
    "purple": "#7c3aed",
    "slate":  "#475569",
    "pink":   "#db2777",
}


# ── SVG helpers ───────────────────────────────────────────────────────────────

def format_number(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (int, float)):
        if abs(value) >= 1000:
            return f"{value:,.{digits}f}"
        return f"{value:.{digits}f}"
    return str(value)


def _sample(x: List[float], y: List[float], max_pts: int) -> Tuple[List[float], List[float]]:
    if len(x) <= max_pts or max_pts <= 0:
        return x, y
    step = max(1, math.ceil(len(x) / max_pts))
    sx, sy = x[::step], y[::step]
    if sx[-1] != x[-1]:
        sx.append(x[-1])
        sy.append(y[-1])
    return sx, sy


def _clean(x_series, y_series, max_pts: int) -> Tuple[List[float], List[float]]:
    pairs = []
    for xv, yv in zip(x_series, y_series):
        try:
            xf, yf = float(xv), float(yv)
        except (TypeError, ValueError):
            continue
        if not (math.isnan(xf) or math.isnan(yf)):
            pairs.append((xf, yf))
    if not pairs:
        return [], []
    return _sample([p[0] for p in pairs], [p[1] for p in pairs], max_pts)


def svg_line_chart(title: str, series, max_points: int = 1500,
                   width: int = 900, height: int = 320) -> str:
    filtered = []
    for item in series:
        xs, ys = _clean(item["x"], item["y"], max_points)
        if xs:
            filtered.append({**item, "x": xs, "y": ys})

    if not filtered:
        return f'<section class="panel"><h2>{html.escape(title)}</h2><p>No data.</p></section>'

    mt, mr, mb, ml = 26, 24, 34, 56
    pw = width - ml - mr
    ph = height - mt - mb

    all_x = [v for it in filtered for v in it["x"]]
    all_y = [v for it in filtered for v in it["y"]]
    x0, x1 = min(all_x), max(all_x)
    y0, y1 = min(all_y), max(all_y)
    if x0 == x1: x1 = x0 + 1.0
    if y0 == y1:
        pad = abs(y0) * 0.1 if y0 else 1.0
        y0 -= pad; y1 += pad
    else:
        pad = (y1 - y0) * 0.08
        y0 -= pad; y1 += pad

    def px(v): return ml + (v - x0) / (x1 - x0) * pw
    def py(v): return mt + ph - (v - y0) / (y1 - y0) * ph

    grid = []
    for t in range(5):
        yv = y0 + (y1 - y0) * t / 4
        yp = py(yv)
        grid.append(
            f'<line x1="{ml}" y1="{yp:.2f}" x2="{width-mr}" y2="{yp:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
            f'<text x="{ml-8}" y="{yp+4:.2f}" text-anchor="end" fill="#64748b" font-size="11">'
            f'{html.escape(format_number(yv, 2))}</text>'
        )

    paths, legend = [], []
    for i, it in enumerate(filtered):
        pts = " ".join(f"{px(xv):.2f},{py(yv):.2f}" for xv, yv in zip(it["x"], it["y"]))
        paths.append(f'<polyline fill="none" stroke="{it["color"]}" stroke-width="2.2" points="{pts}"/>')
        lx = ml + i * 170
        legend.append(
            f'<rect x="{lx}" y="0" width="14" height="14" rx="3" fill="{it["color"]}"/>'
            f'<text x="{lx+20}" y="11" fill="#0f172a" font-size="12">{html.escape(str(it["label"]))}</text>'
        )

    x_labels = (
        f'<text x="{ml}" y="{height-8}" fill="#64748b" font-size="11">{html.escape(format_number(x0, 0))}</text>'
        f'<text x="{width-mr}" y="{height-8}" text-anchor="end" fill="#64748b" font-size="11">'
        f'{html.escape(format_number(x1, 0))}</text>'
    )
    return (
        f'<section class="panel"><h2>{html.escape(title)}</h2>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'{"".join(legend)}'
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#fff" stroke="#cbd5e1" stroke-width="1"/>'
        f'{"".join(grid)}{"".join(paths)}{x_labels}</svg></section>'
    )


def svg_hbar_chart(title: str, labels: List[str], values: List[float],
                   color: str, width: int = 900) -> str:
    cleaned = []
    for label, value in zip(labels, values):
        try:
            n = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(n):
            cleaned.append((str(label), n))

    if not cleaned:
        return f'<section class="panel"><h2>{html.escape(title)}</h2><p>No data.</p></section>'

    bh, gap = 22, 10
    mt, mb, ml, mr = 16, 18, 230, 80
    h = mt + mb + len(cleaned) * (bh + gap)
    pw = width - ml - mr
    max_v = max(v for _, v in cleaned) or 1.0

    parts = [f'<section class="panel"><h2>{html.escape(title)}</h2>'
             f'<svg viewBox="0 0 {width} {h}">']
    for i, (label, value) in enumerate(cleaned):
        yp = mt + i * (bh + gap)
        bw = (value / max_v) * pw
        parts.append(
            f'<text x="{ml-12}" y="{yp+15}" text-anchor="end" fill="#0f172a" font-size="12">'
            f'{html.escape(label)}</text>'
            f'<rect x="{ml}" y="{yp}" width="{bw:.2f}" height="{bh}" rx="4" fill="{color}"/>'
            f'<text x="{ml+bw+8:.2f}" y="{yp+15}" fill="#334155" font-size="12">'
            f'{html.escape(format_number(value, 2))}</text>'
        )
    parts.append("</svg></section>")
    return "".join(parts)


# ── Dashboard builder ─────────────────────────────────────────────────────────

def build_dashboard(data_dir: Path) -> str:
    mach_actions   = pd.read_csv(data_dir / "machine_actions.csv")
    mach_util      = pd.read_csv(data_dir / "machine_utilization_per_episode.csv")
    transp_actions = pd.read_csv(data_dir / "transport_actions.csv")
    transp_util    = pd.read_csv(data_dir / "transport_utilization_per_episode.csv")
    agent_count    = pd.read_csv(data_dir / "agent_actions_count.csv")
    agent_reward   = pd.read_csv(data_dir / "agent_reward.csv")

    panels = []

    # ── 0. Agent mean reward per episode ─────────────────────────────────────
    agent_reward["reward"] = pd.to_numeric(agent_reward["reward"], errors="coerce")
    ep_reward = (
        agent_reward.groupby("episode")["reward"]
        .agg(mean_reward=lambda r: r.sum() / len(r), sum_reward="sum")
        .reset_index()
        .sort_values("episode")
    )

    panels.append(svg_line_chart(
        "Total Episode Reward",
        [
            {
                "label": "sum reward",
                "x": ep_reward["episode"].tolist(),
                "y": ep_reward["sum_reward"].tolist(),
                "color": COLORS["blue"],
            }
        ],
    ))

    # ── 1. Overall machine utilization per episode ───────────────────────────
    mach_util_all = pd.read_csv(data_dir / "machine_utilization_all.csv")
    mach_util_all = mach_util_all.sort_values("episode_counter")
    ep_all = mach_util_all["episode_counter"].tolist()

    panels.append(svg_line_chart(
        "Overall Machine Utilization per Episode",
        [
            {"label": "utilization", "x": ep_all, "y": mach_util_all["utilization"].tolist(), "color": COLORS["green"]},
            {"label": "broken",      "x": ep_all, "y": mach_util_all["broken"].tolist(),      "color": COLORS["red"]},
            {"label": "idle",        "x": ep_all, "y": mach_util_all["idle"].tolist(),         "color": COLORS["slate"]},
        ],
    ))
    panels.append(svg_line_chart(
        "Finished Orders per Episode",
        [
            {"label": "finished_orders", "x": ep_all, "y": mach_util_all["finished_orders"].tolist(), "color": COLORS["blue"]},
        ],
    ))
    panels.append(svg_line_chart(
        "Order Waiting Time per Episode",
        [
            {"label": "order_wait_time", "x": ep_all, "y": mach_util_all["order_wait_time"].tolist(), "color": COLORS["amber"]},
        ]
    ))

    # ── 2. Machine actions: one chart per action type, normalized per machine ─
    count_cols   = [c for c in mach_actions.columns if c.endswith("_count")]
    dur_cols     = [c for c in mach_actions.columns if c.endswith("_total_duration")]
    action_names = [c.replace("_count", "") for c in count_cols]
    machine_labels = mach_actions["machine"].tolist()

    # Normalize each machine's values to sum to 1 across its own actions
    norm_counts = []
    norm_durs   = []
    for _, row in mach_actions.iterrows():
        counts = [float(row[c]) for c in count_cols]
        durs   = [float(row[c]) for c in dur_cols]
        cs = sum(counts) or 1.0
        ds = sum(durs)   or 1.0
        norm_counts.append([v / cs for v in counts])
        norm_durs.append([v / ds for v in durs])

    for i, action in enumerate(action_names):
        panels.append(svg_hbar_chart(
            f"{action} — Count per Machine (Normalized per Machine)",
            machine_labels,
            [norm_counts[m][i] for m in range(len(machine_labels))],
            color=COLORS["blue"],
        ))
        panels.append(svg_hbar_chart(
            f"{action} — Duration per Machine (Normalized per Machine)",
            machine_labels,
            [norm_durs[m][i] for m in range(len(machine_labels))],
            color=COLORS["amber"],
        ))

    # ── 3. Machine utilization per episode — three charts per machine ─────────
    for machine in mach_util["machine"].unique():
        mdf = mach_util[mach_util["machine"] == machine].sort_values("episode_counter")
        ep = mdf["episode_counter"].tolist()
        panels.append(svg_line_chart(
            f"{machine} — Utilization per Episode",
            [{"label": "utilization", "x": ep, "y": mdf["utilization"].tolist(), "color": COLORS["green"]}],
        ))
        panels.append(svg_line_chart(
            f"{machine} — Breakdown per Episode",
            [{"label": "broken", "x": ep, "y": mdf["broken"].tolist(), "color": COLORS["red"]}],
        ))
        panels.append(svg_line_chart(
            f"{machine} — Idle per Episode",
            [{"label": "idle", "x": ep, "y": mdf["idle"].tolist(), "color": COLORS["slate"]}],
        ))

    # ── 4. Transport actions: count + duration bar per transport ──────────────
    t_count_cols = [c for c in transp_actions.columns if c.endswith("_count")]
    t_dur_cols   = [c for c in transp_actions.columns if c.endswith("_total_duration")]
    t_count_names = [c.replace("_count", "") for c in t_count_cols]
    t_dur_names   = [c.replace("_total_duration", "") for c in t_dur_cols]

    for _, row in transp_actions.iterrows():
        transport = row["transport"]
        t_counts = [float(row[c]) for c in t_count_cols]
        t_durs   = [float(row[c]) for c in t_dur_cols]
        t_count_sum = sum(t_counts) or 1.0
        t_dur_sum   = sum(t_durs)   or 1.0
        panels.append(svg_hbar_chart(
            f"{transport} — Action Counts (Normalized)",
            t_count_names,
            [v / t_count_sum for v in t_counts],
            color=COLORS["purple"],
        ))
        panels.append(svg_hbar_chart(
            f"{transport} — Action Durations (Normalized)",
            t_dur_names,
            [v / t_dur_sum for v in t_durs],
            color=COLORS["cyan"],
        ))

    # ── 5. Transport utilization per episode — one line chart per transport ───
    for transport in transp_util["transport"].unique():
        tdf = transp_util[transp_util["transport"] == transport].sort_values("episode_counter")
        ep = tdf["episode_counter"].tolist()
        panels.append(svg_line_chart(
            f"{transport} — Utilization per Episode",
            [
                {"label": "working",  "x": ep, "y": tdf["working"].tolist(),  "color": COLORS["blue"]},
                {"label": "walking",  "x": ep, "y": tdf["walking"].tolist(),  "color": COLORS["cyan"]},
                {"label": "handling", "x": ep, "y": tdf["handling"].tolist(), "color": COLORS["purple"]},
                {"label": "idle",     "x": ep, "y": tdf["idle"].tolist(),     "color": COLORS["slate"]},
            ],
        ))

    # ── 6. Agent action frequencies bar chart ────────────────────────────────
    # ! action mapping needs to match the one used in the simulation
    action_mapping = pd.read_csv(
        Path(ACTION_OUTPUT_PATH)
    )
    id_to_label = dict(zip(action_mapping["action_index"].astype(str), action_mapping["action_label"]))

    row = agent_count.iloc[0]
    action_labels = [
        id_to_label.get(c.replace("action_", "").replace("_freq", ""), c.replace("_freq", ""))
        for c in agent_count.columns
    ]
    action_freqs = [float(row[c]) for c in agent_count.columns]
    panels.append(svg_hbar_chart(
        "Agent Action Frequencies",
        action_labels,
        action_freqs,
        color=COLORS["green"],
    ))

    css = """
    :root { color-scheme: light; --bg: #f8fafc; --panel: #ffffff; --border: #dbe2ea;
            --text: #0f172a; --muted: #64748b; --shadow: 0 10px 30px rgba(15,23,42,.06); }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Arial, Helvetica, sans-serif; background: var(--bg);
           color: var(--text); line-height: 1.45; }
    header { padding: 28px 32px 12px; }
    header h1 { margin: 0 0 8px; font-size: 30px; }
    header p  { margin: 0; color: var(--muted); max-width: 900px; }
    main { padding: 0 24px 32px; display: grid; gap: 18px; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
             box-shadow: var(--shadow); padding: 18px; overflow-x: auto; }
    .panel h2 { margin: 0 0 14px; font-size: 19px; }
    svg { width: 100%; min-width: 760px; height: auto; display: block; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Simulation Log Dashboard</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>Simulation Log Dashboard</h1>
    <p>Source: <code>{html.escape(str(data_dir))}</code></p>
  </header>
  <main>{"".join(panels)}</main>
</body>
</html>"""


if __name__ == "__main__":
    NUM_STATES = 47
    DATA_DIR = Path(f"./log/{NUM_STATES} states viper/log_visualization")
    ACTION_OUTPUT_PATH = DATA_DIR / "action_mapping.csv"
    OUTPUT_HTML = DATA_DIR / "dashboard.html"

    dashboard = build_dashboard(DATA_DIR)
    OUTPUT_HTML.write_text(dashboard, encoding="utf-8")
    print(f"Dashboard written to: {OUTPUT_HTML}")