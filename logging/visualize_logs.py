import argparse
import html
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_LOG_DIR = Path("./log/66 states test")
NORMALIZATION = True
FIRST_N = 1000

COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#16a34a",
    "amber": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "slate": "#475569",
    "pink": "#db2777",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML dashboard for a simulator log directory."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=str(DEFAULT_LOG_DIR),
        help="Directory containing the CSV log files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the dashboard and summary CSV files will be written.",
    )
    parser.add_argument(
        "--max-line-points",
        type=int,
        default=1500,
        help="Maximum points to draw per line chart series after downsampling.",
    )
    return parser.parse_args()


def format_number(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (int, float)):
        if abs(value) >= 1000:
            return f"{value:,.{digits}f}"
        return f"{value:.{digits}f}"
    return str(value)


def sample_points(x_values: List[float], y_values: List[float], max_points: int) -> Tuple[List[float], List[float]]:
    if len(x_values) <= max_points or max_points <= 0:
        return x_values, y_values
    step = max(1, math.ceil(len(x_values) / max_points))
    sampled_x = x_values[::step]
    sampled_y = y_values[::step]
    if sampled_x[-1] != x_values[-1]:
        sampled_x.append(x_values[-1])
        sampled_y.append(y_values[-1])
    return sampled_x, sampled_y


def clean_pairs(x_series: pd.Series, y_series: pd.Series, max_points: int) -> Tuple[List[float], List[float]]:
    pairs = []
    x_list = x_series if isinstance(x_series, list) else x_series.tolist()
    y_list = y_series if isinstance(y_series, list) else y_series.tolist()
    for x_val, y_val in zip(x_list, y_list):
        try:
            x_num = float(x_val)
            y_num = float(y_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(x_num) or math.isnan(y_num):
            continue
        pairs.append((x_num, y_num))
    if not pairs:
        return [], []
    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    return sample_points(x_values, y_values, max_points=max_points)


def normalize_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    minimum = valid.min()
    maximum = valid.max()
    if maximum == minimum:
        return numeric.apply(lambda value: 0.0 if pd.notna(value) else value)
    return (numeric - minimum) / (maximum - minimum)


def maybe_normalize_series(series: pd.Series) -> pd.Series:
    if NORMALIZATION:
        return normalize_series(series)
    return pd.to_numeric(series, errors="coerce")


def svg_line_chart(
    title: str,
    series,
    max_points: int,
    width: int = 900,
    height: int = 320,
) -> str:
    filtered = []
    for item in series:
        x_values, y_values = clean_pairs(item["x"], item["y"], max_points=max_points)
        if x_values and y_values:
            filtered.append(
                {
                    "label": item["label"],
                    "color": item["color"],
                    "x": x_values,
                    "y": y_values,
                }
            )

    if not filtered:
        return f'<section class="panel"><h2>{html.escape(title)}</h2><p>No data available.</p></section>'

    margin_top = 26
    margin_right = 24
    margin_bottom = 34
    margin_left = 56
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_x = [value for item in filtered for value in item["x"]]
    all_y = [value for item in filtered for value in item["y"]]
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)

    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        padding = abs(min_y) * 0.1 if min_y else 1.0
        min_y -= padding
        max_y += padding
    else:
        padding = (max_y - min_y) * 0.08
        min_y -= padding
        max_y += padding

    def project_x(value: float) -> float:
        return margin_left + ((value - min_x) / (max_x - min_x)) * plot_width

    def project_y(value: float) -> float:
        return margin_top + plot_height - ((value - min_y) / (max_y - min_y)) * plot_height

    grid = []
    for tick in range(5):
        frac = tick / 4 if 4 else 0
        y_val = min_y + (max_y - min_y) * frac
        y_pos = project_y(y_val)
        grid.append(
            f'<line x1="{margin_left}" y1="{y_pos:.2f}" x2="{width - margin_right}" y2="{y_pos:.2f}" '
            f'stroke="#e2e8f0" stroke-width="1" />'
            f'<text x="{margin_left - 8}" y="{y_pos + 4:.2f}" text-anchor="end" fill="#64748b" font-size="11">'
            f"{html.escape(format_number(y_val, 2))}</text>"
        )

    paths = []
    legend = []
    for index, item in enumerate(filtered):
        coords = " ".join(
            f"{project_x(x_val):.2f},{project_y(y_val):.2f}"
            for x_val, y_val in zip(item["x"], item["y"])
        )
        paths.append(
            f'<polyline fill="none" stroke="{item["color"]}" stroke-width="2.2" points="{coords}" />'
        )
        legend_x = margin_left + index * 170
        legend.append(
            f'<rect x="{legend_x}" y="0" width="14" height="14" rx="3" fill="{item["color"]}" />'
            f'<text x="{legend_x + 20}" y="11" fill="#0f172a" font-size="12">{html.escape(str(item["label"]))}</text>'
        )

    x_labels = (
        f'<text x="{margin_left}" y="{height - 8}" fill="#64748b" font-size="11">{html.escape(format_number(min_x, 0))}</text>'
        f'<text x="{width - margin_right}" y="{height - 8}" text-anchor="end" fill="#64748b" font-size="11">'
        f"{html.escape(format_number(max_x, 0))}</text>"
    )

    return (
        f'<section class="panel"><h2>{html.escape(title)}</h2>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'{"".join(legend)}'
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" '
        f'fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />'
        f'{"".join(grid)}'
        f'{"".join(paths)}'
        f"{x_labels}"
        f"</svg></section>"
    )


def svg_hbar_chart(
    title: str,
    labels: List[str],
    values: List[float],
    color: str,
    width: int = 900,
) -> str:
    cleaned = []
    for label, value in zip(labels, values):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        cleaned.append((str(label), numeric))

    if not cleaned:
        return f'<section class="panel"><h2>{html.escape(title)}</h2><p>No data available.</p></section>'

    bar_height = 22
    gap = 10
    margin_top = 16
    margin_bottom = 18
    margin_left = 230
    margin_right = 80
    height = margin_top + margin_bottom + len(cleaned) * (bar_height + gap)
    plot_width = width - margin_left - margin_right
    max_value = max(value for _, value in cleaned)
    if max_value == 0:
        max_value = 1.0

    parts = [f'<section class="panel"><h2>{html.escape(title)}</h2><svg viewBox="0 0 {width} {height}">']
    for index, (label, value) in enumerate(cleaned):
        y_pos = margin_top + index * (bar_height + gap)
        bar_width = (value / max_value) * plot_width
        parts.append(
            f'<text x="{margin_left - 12}" y="{y_pos + 15}" text-anchor="end" fill="#0f172a" font-size="12">'
            f"{html.escape(label)}</text>"
        )
        parts.append(
            f'<rect x="{margin_left}" y="{y_pos}" width="{bar_width:.2f}" height="{bar_height}" rx="4" fill="{color}" />'
        )
        parts.append(
            f'<text x="{margin_left + bar_width + 8:.2f}" y="{y_pos + 15}" fill="#334155" font-size="12">'
            f"{html.escape(format_number(value, 2))}</text>"
        )
    parts.append("</svg></section>")
    return "".join(parts)


def html_table(title: str, frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return f'<section class="panel"><h2>{html.escape(title)}</h2><p>No data available.</p></section>'

    preview = frame.head(max_rows).copy()
    for column in preview.columns:
        if pd.api.types.is_numeric_dtype(preview[column]):
            preview[column] = preview[column].map(lambda value: format_number(value, 3))

    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in preview.columns)
    body_rows = []
    for _, row in preview.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        f'<section class="panel"><h2>{html.escape(title)}</h2>'
        f'<div class="table-wrap"><table><thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table></div></section>'
    )


def render_cards(cards: List[Tuple[str, str]]) -> str:
    parts = ['<section class="card-grid">']
    for label, value in cards:
        parts.append(
            '<div class="metric-card">'
            f'<div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-value">{html.escape(value)}</div>'
            "</div>"
        )
    parts.append("</section>")
    return "".join(parts)


def read_episode_log(log_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(log_dir / "episode_log.csv")
    frame["episode_counter"] = pd.to_numeric(frame["episode_counter"], errors="coerce")
    frame = frame.dropna(subset=["episode_counter"]).sort_values("episode_counter")
    frame["episode_counter"] = frame["episode_counter"].astype(int)
    return frame


def read_agent_reward_log(log_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(
        log_dir / "agent_reward_log.csv",
        usecols=["episode", "sim_step", "sim_time", "reward", "action_valid"],
        low_memory=False,
    )
    frame["episode"] = pd.to_numeric(frame["episode"], errors="coerce")
    frame["sim_step"] = pd.to_numeric(frame["sim_step"], errors="coerce")
    frame["sim_time"] = pd.to_numeric(frame["sim_time"], errors="coerce")
    frame["reward"] = pd.to_numeric(frame["reward"], errors="coerce")
    frame["action_valid"] = frame["action_valid"].astype(str).str.lower().eq("true")
    frame = frame.dropna(subset=["episode", "sim_step", "reward"]).sort_values(["episode", "sim_step"])

    filtered = frame[frame["episode"] >= 1].copy()
    episode_summary = (
        filtered.groupby("episode", as_index=False)
        .agg(
            agent_total_reward=("reward", "sum"),
            agent_mean_reward=("reward", "mean"),
            decision_count=("reward", "size"),
            valid_action_count=("action_valid", "sum"),
        )
        .sort_values("episode")
    )
    episode_summary["episode"] = episode_summary["episode"].astype(int)
    episode_summary["invalid_action_count"] = (
        episode_summary["decision_count"] - episode_summary["valid_action_count"]
    )
    episode_summary["valid_action_rate"] = (
        episode_summary["valid_action_count"] / episode_summary["decision_count"]
    )
    return frame, episode_summary


def read_machine_log(log_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = log_dir / "machine_log.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    machine_ids = sorted(
        {
            int(match.group(1))
            for column in header
            for match in [re.match(r"machine_(\d+)_action", column)]
            if match
        }
    )

    frame = pd.read_csv(path, engine="python", dtype=str)

    per_machine_rows = []
    action_count_totals = {}  # type: Dict[str, float]
    action_duration_totals = {}  # type: Dict[str, float]

    for machine_id in machine_ids:
        action_col = f"machine_{machine_id}_action"
        duration_col = f"machine_{machine_id}_duration"
        subset = frame[[action_col, duration_col]].dropna(subset=[action_col]).copy()
        subset = subset.rename(columns={action_col: "action", duration_col: "duration"})
        subset["duration_numeric"] = pd.to_numeric(subset["duration"], errors="coerce").fillna(0.0)

        action_counts = subset["action"].value_counts()
        action_durations = subset.groupby("action")["duration_numeric"].sum()

        for action, count in action_counts.items():
            action_count_totals[action] = action_count_totals.get(action, 0.0) + float(count)
        for action, duration in action_durations.items():
            action_duration_totals[action] = action_duration_totals.get(action, 0.0) + float(duration)

        per_machine_rows.append(
            {
                "machine_id": machine_id,
                "processing_time": float(action_durations.get("processing", 0.0)),
                "breakdown_time": float(action_durations.get("breakdown", 0.0)),
                "idle_starvation_time": float(action_durations.get("idle_starvation", 0.0)),
                "processing_events": int(action_counts.get("processing", 0)),
                "breakdown_events": int(action_counts.get("breakdown", 0)),
                "idle_starvation_events": int(action_counts.get("idle_starvation", 0)),
                "changeover_events": int(action_counts.get("changeover", 0)),
            }
        )

    action_summary = pd.DataFrame(
        [
            {
                "action": action,
                "event_count": action_count_totals.get(action, 0.0),
                "numeric_duration_total": action_duration_totals.get(action, 0.0),
            }
            for action in sorted(action_count_totals.keys())
        ]
    ).sort_values("event_count", ascending=False)

    per_machine_summary = pd.DataFrame(per_machine_rows).sort_values("machine_id")
    return action_summary, per_machine_summary


def read_transport_log(log_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = log_dir / "transport_log.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    transport_ids = sorted(
        {
            int(match.group(1))
            for column in header
            for match in [re.match(r"transp_(\d+)_action", column)]
            if match
        }
    )

    usecols = []
    for transport_id in transport_ids:
        usecols.extend(
            [
                f"transp_{transport_id}_action",
                f"transp_{transport_id}_sim_time",
                f"transp_{transport_id}_from_at",
                f"transp_{transport_id}_to_at",
                f"transp_{transport_id}_duration",
            ]
        )

    frame = pd.read_csv(path, usecols=usecols, low_memory=False)

    action_count_totals = {}  # type: Dict[str, float]
    action_duration_totals = {}  # type: Dict[str, float]
    route_totals = {}  # type: Dict[str, float]

    for transport_id in transport_ids:
        subset = frame[
            [
                f"transp_{transport_id}_action",
                f"transp_{transport_id}_from_at",
                f"transp_{transport_id}_to_at",
                f"transp_{transport_id}_duration",
            ]
        ].dropna(subset=[f"transp_{transport_id}_action"]).copy()
        subset.columns = ["action", "from_at", "to_at", "duration"]
        subset["duration"] = pd.to_numeric(subset["duration"], errors="coerce").fillna(0.0)

        counts = subset["action"].value_counts()
        durations = subset.groupby("action")["duration"].sum()

        for action, count in counts.items():
            action_count_totals[action] = action_count_totals.get(action, 0.0) + float(count)
        for action, duration in durations.items():
            action_duration_totals[action] = action_duration_totals.get(action, 0.0) + float(duration)

        routes = subset[pd.to_numeric(subset["from_at"], errors="coerce") != pd.to_numeric(subset["to_at"], errors="coerce")]
        if not routes.empty:
            route_counts = (
                routes.groupby(["from_at", "to_at"], dropna=True)
                .size()
                .sort_values(ascending=False)
            )
            for (from_at, to_at), count in route_counts.items():
                route_label = f"{int(float(from_at))} -> {int(float(to_at))}"
                route_totals[route_label] = route_totals.get(route_label, 0.0) + float(count)

    action_summary = pd.DataFrame(
        [
            {
                "action": action,
                "event_count": action_count_totals.get(action, 0.0),
                "duration_total": action_duration_totals.get(action, 0.0),
            }
            for action in sorted(action_count_totals.keys())
        ]
    ).sort_values("event_count", ascending=False)

    route_summary = pd.DataFrame(
        [{"route": route, "count": count} for route, count in route_totals.items()]
    ).sort_values("count", ascending=False)

    return action_summary, route_summary


def build_dashboard(
    log_dir: Path,
    output_dir: Path,
    episode_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    agent_episode_df: pd.DataFrame,
    machine_action_df: pd.DataFrame,
    machine_machine_df: pd.DataFrame,
    transport_action_df: pd.DataFrame,
    transport_route_df: pd.DataFrame,
    max_points: int,
) -> str:
    merged = episode_df.merge(
        agent_episode_df,
        left_on="episode_counter",
        right_on="episode",
        how="left",
    )

    episode_normalized = episode_df.copy()
    for column in [
        "total_reward",
        "finished_orders",
        "order_waiting_time",
    ]:
        episode_normalized[column] = maybe_normalize_series(episode_normalized[column])

    agent_episode_normalized = agent_episode_df.copy()
    for column in [
        "agent_mean_reward",
        "decision_count",
        "valid_action_rate",
    ]:
        agent_episode_normalized[column] = maybe_normalize_series(agent_episode_normalized[column])

    _transp_cols = ["transp_working", "transp_walking", "transp_handling", "transp_idle"]
    _machine_cols = ["machines_working", "machines_broken", "machines_idle"]

    machine_action_normalized = machine_action_df.copy()
    _ec = pd.to_numeric(machine_action_df["event_count"], errors="coerce")
    machine_action_normalized["event_count"] = (_ec / _ec.sum()) if NORMALIZATION else _ec

    _dur = pd.to_numeric(machine_action_df["numeric_duration_total"], errors="coerce")
    machine_action_normalized["numeric_duration_total"] = (_dur / _dur.sum()) if NORMALIZATION else _dur

    machine_machine_normalized = machine_machine_df.copy()
    machine_machine_normalized = machine_machine_df.copy()
    _total_time = (
        pd.to_numeric(machine_machine_df["processing_time"], errors="coerce").fillna(0.0)
        + pd.to_numeric(machine_machine_df["breakdown_time"], errors="coerce").fillna(0.0)
        + pd.to_numeric(machine_machine_df["idle_starvation_time"], errors="coerce").fillna(0.0)
    ).replace(0.0, float("nan"))
    if NORMALIZATION:
        machine_machine_normalized["processing_time"] = pd.to_numeric(machine_machine_df["processing_time"], errors="coerce") / _total_time
        machine_machine_normalized["breakdown_time"] = pd.to_numeric(machine_machine_df["breakdown_time"], errors="coerce") / _total_time
        _transp_sum = sum(pd.to_numeric(episode_df[col], errors="coerce").fillna(0.0) for col in _transp_cols).replace(0.0, float("nan"))
        for col in _transp_cols:
            episode_normalized[col] = pd.to_numeric(episode_df[col], errors="coerce") / _transp_sum
        
        _machine_sum = sum(pd.to_numeric(episode_df[col], errors="coerce").fillna(0.0) for col in _machine_cols).replace(0.0, float("nan"))
        for col in _machine_cols:
            episode_normalized[col] = pd.to_numeric(episode_df[col], errors="coerce") / _machine_sum
    else:
        machine_machine_normalized["processing_time"] = pd.to_numeric(machine_machine_df["processing_time"], errors="coerce")
        machine_machine_normalized["breakdown_time"] = pd.to_numeric(machine_machine_df["breakdown_time"], errors="coerce")
        for col in _transp_cols:
            episode_normalized[col] = pd.to_numeric(episode_df[col], errors="coerce")
        for col in _machine_cols:
            episode_normalized[col] = pd.to_numeric(episode_df[col], errors="coerce")

    transport_action_normalized = transport_action_df.copy()
    _tec = pd.to_numeric(transport_action_df["event_count"], errors="coerce")
    transport_action_normalized["event_count"] = (_tec / _tec.sum()) if NORMALIZATION else _tec

    _tdur = pd.to_numeric(transport_action_df["duration_total"], errors="coerce")
    transport_action_normalized["duration_total"] = (_tdur / _tdur.sum()) if NORMALIZATION else _tdur

    transport_route_normalized = transport_route_df.copy()
    if not transport_route_normalized.empty:
        _rc = pd.to_numeric(transport_route_df["count"], errors="coerce")
        transport_route_normalized["count"] = (_rc / _rc.sum()) if NORMALIZATION else _rc

    normalization_suffix = " (Normalized 0-1)" if NORMALIZATION else ""
    normalization_note = (
        "<li>All plotted charts use per-series min-max normalization, so each visualized metric is scaled to the "
        "<code>0..1</code> range for comparison. Summary cards and tables remain in raw units.</li>"
        if NORMALIZATION
        else "<li>Plotted charts currently use raw values. Set <code>NORMALIZATION = True</code> near the top of the script to switch to normalized charts.</li>"
    )

    cards = [
        ("Log folder", str(log_dir.name)),
        ("Episodes", format_number(float(len(episode_df)), 0)),
        ("Agent decisions", format_number(float(len(agent_df)), 0)),
        ("Mean episode reward", format_number(float(episode_df["total_reward"].mean()), 2)),
        ("Mean valid action rate", format_number(float(agent_episode_df["valid_action_rate"].mean() * 100.0), 2) + "%"),
        ("Mean finished orders / episode", format_number(float(episode_df["finished_orders"].mean()), 2)),
        ("Mean order waiting time", format_number(float(episode_df["order_waiting_time"].mean()), 2)),
        ("Avg machine working share", format_number(float(episode_df["machines_working"].mean()), 3)),
        ("Avg transport working share", format_number(float(episode_df["transp_working"].mean()), 3)),
    ]

    _reward_y = agent_episode_normalized["agent_total_reward"][:FIRST_N]
    _reward_x = agent_episode_normalized["episode"][:FIRST_N]
    _reward_mean = float(pd.to_numeric(_reward_y, errors="coerce").mean())

    _fo_x = episode_df["episode_counter"][:FIRST_N]
    _fo_y = pd.to_numeric(episode_df["finished_orders"][:FIRST_N], errors="coerce")
    _fo_mean = float(_fo_y.mean())

    _wt_x = episode_df["episode_counter"][:FIRST_N]
    _wt_y = pd.to_numeric(episode_df["order_waiting_time"][:FIRST_N], errors="coerce")
    _wt_mean = float(_wt_y.mean())

    panels = [
        render_cards(cards),
        svg_line_chart(
            "Episode Reward Comparison",
            [
                {
                    "label": "agent_reward_log summed reward",
                    "x": _reward_x,
                    "y": _reward_y,
                    "color": COLORS["amber"],
                },
                {
                    "label": f"mean ({format_number(_reward_mean, 2)})",
                    "x": [_reward_x.iloc[0], _reward_x.iloc[-1]],
                    "y": [_reward_mean, _reward_mean],
                    "color": COLORS["slate"],
                },
            ],
            max_points=max_points,
        ),
        svg_line_chart(
            "Finished Orders per Episode",
            [
                {"label": "finished_orders", "x": _fo_x, "y": _fo_y, "color": COLORS["green"]},
                {
                    "label": f"mean ({format_number(_fo_mean, 2)})",
                    "x": [_fo_x.iloc[0], _fo_x.iloc[-1]],
                    "y": [_fo_mean, _fo_mean],
                    "color": COLORS["slate"],
                },
            ],
            max_points=max_points,
        ),
        svg_line_chart(
            "Order Waiting Time per Episode",
            [
                {"label": "order_waiting_time", "x": _wt_x, "y": _wt_y, "color": COLORS["red"]},
                {
                    "label": f"mean ({format_number(_wt_mean, 2)})",
                    "x": [_wt_x.iloc[0], _wt_x.iloc[-1]],
                    "y": [_wt_mean, _wt_mean],
                    "color": COLORS["slate"],
                },
            ],
            max_points=max_points,
        ),
        svg_line_chart(
            "Machine Utilization Shares by Episode" + normalization_suffix,
            [
                {
                    "label": "working",
                    "x": episode_normalized["episode_counter"][:FIRST_N],
                    "y": episode_normalized["machines_working"][:FIRST_N],
                    "color": COLORS["green"],
                },
                {
                    "label": "broken",
                    "x": episode_normalized["episode_counter"][:FIRST_N],
                    "y": episode_normalized["machines_broken"][:FIRST_N],
                    "color": COLORS["red"],
                },
                {
                    "label": "idle",
                    "x": episode_normalized["episode_counter"][:FIRST_N],
                    "y": episode_normalized["machines_idle"][:FIRST_N],
                    "color": COLORS["slate"],
                },
            ],
            max_points=max_points,
            # height=480,
        ),
        svg_line_chart(
            "Transport Utilization Shares by Episode" + normalization_suffix,
            [
                {
                    "label": "working",
                    "x": episode_normalized["episode_counter"][:FIRST_N],
                    "y": episode_normalized["transp_working"][:FIRST_N],
                    "color": COLORS["blue"],
                },
                {
                    "label": "walking",
                    "x": episode_normalized["episode_counter"][:FIRST_N],
                    "y": episode_normalized["transp_walking"][:FIRST_N],
                    "color": COLORS["cyan"],
                },
                {
                    "label": "handling",
                    "x": episode_normalized["episode_counter"][:FIRST_N],
                    "y": episode_normalized["transp_handling"][:FIRST_N],
                    "color": COLORS["purple"],
                },
                {
                    "label": "idle",
                    "x": episode_normalized["episode_counter"][:FIRST_N],
                    "y": episode_normalized["transp_idle"][:FIRST_N],
                    "color": COLORS["slate"],
                },
            ],
            max_points=max_points,
            # height=480,
        ),

        svg_hbar_chart(
            "Machine Events by Action" + normalization_suffix,
            machine_action_normalized["action"].astype(str).tolist(),
            machine_action_normalized["event_count"].astype(float).tolist(),
            color=COLORS["blue"],
        ),
        svg_hbar_chart(
            "Machine Numeric Duration by Action" + normalization_suffix,
            machine_action_normalized["action"].astype(str).tolist(),
            machine_action_normalized["numeric_duration_total"].astype(float).tolist(),
            color=COLORS["amber"],
        ),
        svg_hbar_chart(
            "Processing Time by Machine" + normalization_suffix,
            [f"machine_{int(machine_id)}" for machine_id in machine_machine_normalized["machine_id"].tolist()],
            machine_machine_normalized["processing_time"].astype(float).tolist(),
            color=COLORS["green"],
        ),
        svg_hbar_chart(
            "Breakdown Time by Machine" + normalization_suffix,
            [f"machine_{int(machine_id)}" for machine_id in machine_machine_normalized["machine_id"].tolist()],
            machine_machine_normalized["breakdown_time"].astype(float).tolist(),
            color=COLORS["red"],
        ),
        svg_hbar_chart(
            "Transport Events by Action" + normalization_suffix,
            transport_action_normalized["action"].astype(str).tolist(),
            transport_action_normalized["event_count"].astype(float).tolist(),
            color=COLORS["purple"],
        ),
        svg_hbar_chart(
            "Transport Duration by Action" + normalization_suffix,
            transport_action_normalized["action"].astype(str).tolist(),
            transport_action_normalized["duration_total"].astype(float).tolist(),
            color=COLORS["cyan"],
        ),
        svg_hbar_chart(
            "Top Transport Routes" + normalization_suffix,
            transport_route_normalized.head(12)["route"].astype(str).tolist(),
            transport_route_normalized.head(12)["count"].astype(float).tolist(),
            color=COLORS["slate"],
        ),
        svg_hbar_chart(
            "Least Used Transport Routes" + normalization_suffix,
            transport_route_normalized.tail(12)["route"].astype(str).tolist(),
            transport_route_normalized.tail(12)["count"].astype(float).tolist(),
            color=COLORS["slate"],
        ),
        html_table(
            "Episode and Agent Summary Preview",
            merged[
                [
                    "episode_counter",
                    "total_reward",
                    "finished_orders",
                    "order_waiting_time",
                    "agent_total_reward",
                    "decision_count",
                    "valid_action_rate",
                ]
            ],
        ),
        html_table("Machine Summary Preview", machine_machine_df),
        html_table("Transport Route Preview", transport_route_df),
    ]

    notes = (
        '<section class="panel">'
        + "<h2>Notes</h2>"
        + "<ul>"
        + "<li>Dashboard source files: <code>agent_reward_log.csv</code>, <code>episode_log.csv</code>, "
        + "<code>machine_log.csv</code>, and <code>transport_log.csv</code>.</li>"
        + normalization_note
        + "<li>Machine <code>changeover</code> rows in the current log format store a transition descriptor in the "
        + "<code>duration</code> column, so duration charts count only numeric durations.</li>"
        + f"<li>Full outputs are written to <code>{html.escape(str(output_dir))}</code>.</li>"
        + "</ul>"
        + "</section>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Simulation Log Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --border: #dbe2ea;
      --text: #0f172a;
      --muted: #64748b;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 12px;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
      max-width: 900px;
    }}
    main {{
      padding: 0 24px 32px;
      display: grid;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 18px;
      overflow-x: auto;
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 19px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric-value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
      word-break: break-word;
    }}
    svg {{
      width: 100%;
      min-width: 760px;
      height: auto;
      display: block;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-top: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      background: #f8fafc;
    }}
    code {{
      font-family: Consolas, Monaco, monospace;
      font-size: 0.95em;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Simulation Log Dashboard</h1>
    <p>Generated from <code>{html.escape(str(log_dir))}</code>. This dashboard focuses on reward behavior,
    episode-level KPIs, machine activity, and transport activity for the selected run.</p>
  </header>
  <main>
    {''.join(panels)}
    {notes}
  </main>
</body>
</html>
"""


def ensure_required_files(log_dir: Path) -> None:
    required = [
        "agent_reward_log.csv",
        "episode_log.csv",
        "machine_log.csv",
        "transport_log.csv",
    ]
    missing = [name for name in required if not (log_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {log_dir}: {', '.join(missing)}"
        )


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (log_dir / "visualizations")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_required_files(log_dir)

    episode_df = read_episode_log(log_dir)
    agent_df, agent_episode_df = read_agent_reward_log(log_dir)
    machine_action_df, machine_machine_df = read_machine_log(log_dir)
    transport_action_df, transport_route_df = read_transport_log(log_dir)

    merged_episode_agent = episode_df.merge(
        agent_episode_df,
        left_on="episode_counter",
        right_on="episode",
        how="left",
    )

    merged_episode_agent.to_csv(output_dir / "episode_agent_summary.csv", index=False)
    machine_action_df.to_csv(output_dir / "machine_action_summary.csv", index=False)
    machine_machine_df.to_csv(output_dir / "machine_per_machine_summary.csv", index=False)
    transport_action_df.to_csv(output_dir / "transport_action_summary.csv", index=False)
    transport_route_df.to_csv(output_dir / "transport_route_summary.csv", index=False)

    dashboard = build_dashboard(
        log_dir=log_dir,
        output_dir=output_dir,
        episode_df=episode_df,
        agent_df=agent_df,
        agent_episode_df=agent_episode_df,
        machine_action_df=machine_action_df,
        machine_machine_df=machine_machine_df,
        transport_action_df=transport_action_df,
        transport_route_df=transport_route_df,
        max_points=args.max_line_points,
    )

    dashboard_path = output_dir / "simulation_dashboard.html"
    dashboard_path.write_text(dashboard, encoding="utf-8")

    print(f"Dashboard written to: {dashboard_path}")
    print(f"Summary files written to: {output_dir}")


if __name__ == "__main__":
    main()
