"""
Plausibility assessment and HTML report generation.

Classifies each biomarker trajectory as:
- ok (✓) Safe: within expected range/direction — fully acceptable
- warn (⚠) Marginal: unexpected magnitude or direction — barely acceptable
- fail (✗) Hazardous: outside limits or pharmacologically inconsistent — abnormal

Outputs a single HTML file with red/black design, summary cards, filterable table,
and Chart.js time-series charts per metric (each with ✓/⚠/✗ badge).
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

from utils.lab_reference_ranges import (
    LAB_RANGES,
    CRITICAL_RULES,
    get_lab_range,
    validate_lab_value,
)
from utils.constants import get_metric_display_name


# Thresholds for "marginal" vs "implausible" when no ref range (e.g. LBDSCASI)
MAX_REASONABLE_DELTA_PCT = 80  # percent change from baseline — above = warn/fail
MAX_REASONABLE_DELTA_PCT_FAIL = 200  # above this = fail


def classify_plausibility(
    metric_name: str,
    baseline: float,
    final_delta: float,
    series_values: List[float],
) -> Tuple[str, str]:
    """
    Classify a single metric as 'ok', 'warn', or 'fail' and return a short note.

    Returns:
        (status, note) where status is 'ok' | 'warn' | 'fail'
    """
    final_val = baseline + final_delta
    all_vals = [baseline, final_val] + [v for v in series_values if v == v]  # exclude NaN

    # Must be positive (CRITICAL_RULES)
    if metric_name in CRITICAL_RULES.get("MUST_BE_POSITIVE", []):
        for v in all_vals:
            if v < 0:
                return (
                    "fail",
                    f"Value must be positive (got {v:.3f}). Pharmacologically inconsistent.",
                )

    # Use LAB_RANGES if available
    if metric_name in LAB_RANGES:
        info = LAB_RANGES[metric_name]
        phys = info.get("physiological_limit")
        normal = info.get("normal_range")
        clinical = info.get("clinical_range")

        if phys:
            lo, hi = phys
            for v in all_vals:
                if v < lo or v > hi:
                    return (
                        "fail",
                        f"Outside physiological limits ({lo}–{hi}). Pharmacologically inconsistent.",
                    )

        # All within phys; check normal/clinical
        if normal:
            lo_n, hi_n = normal
            any_outside_normal = any(v < lo_n or v > hi_n for v in all_vals)
            if any_outside_normal and clinical:
                lo_c, hi_c = clinical
                any_outside_clinical = any(v < lo_c or v > hi_c for v in all_vals)
                if any_outside_clinical:
                    return ("fail", "Outside clinical range. Hazardous.")
                return ("warn", "Outside normal range but within clinical. Unexpected magnitude or direction.")
            if any_outside_normal:
                return ("warn", "Outside normal range. Unexpected magnitude or direction.")

        # Optional: large delta as % of baseline → warn
        if baseline != 0:
            pct = abs(final_delta / baseline) * 100
            if pct > MAX_REASONABLE_DELTA_PCT_FAIL:
                return ("fail", f"Change too large ({pct:.0f}% of baseline). Pharmacologically inconsistent.")
            if pct > MAX_REASONABLE_DELTA_PCT:
                return ("warn", f"Large change ({pct:.0f}% of baseline). Verify magnitude.")

        return ("ok", "Within expected range/direction.")

    # No LAB_RANGES (e.g. LBDSCASI): use delta % only
    if baseline != 0:
        pct = abs(final_delta / baseline) * 100
        if pct > MAX_REASONABLE_DELTA_PCT_FAIL:
            return ("fail", "Change way too large. Abnormal.")
        if pct > MAX_REASONABLE_DELTA_PCT:
            return ("warn", "Unexpected magnitude. Barely acceptable.")
    return ("ok", "Within expected range/direction.")


def build_plausibility_meta(
    time_series_df,
    timepoint_cols: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], int, int, int]:
    """
    Build per-metric metadata: status, note, ref range, etc.
    Returns (meta_dict, cnt_ok, cnt_warn, cnt_fail).
    """
    meta = {}
    cnt_ok = cnt_warn = cnt_fail = 0

    for metric_name in time_series_df["metric_name"].unique():
        row = time_series_df[time_series_df["metric_name"] == metric_name].iloc[0]
        baseline = float(row["baseline"])
        final_delta = float(row["final_delta"])
        series_vals = []
        for c in timepoint_cols:
            if c in row:
                v = row[c]
                if v == v:  # not NaN
                    series_vals.append(float(v))

        status, note = classify_plausibility(metric_name, baseline, final_delta, series_vals)
        if status == "ok":
            cnt_ok += 1
        elif status == "warn":
            cnt_warn += 1
        else:
            cnt_fail += 1

        ref_lo = ref_hi = None
        ref_str = "—"
        unit = "—"
        if metric_name in LAB_RANGES:
            unit = LAB_RANGES[metric_name].get("unit", "—")
            nr = get_lab_range(metric_name, "normal_range")
            if nr:
                ref_lo, ref_hi = nr
                ref_str = f"{ref_lo}–{ref_hi} {unit}"

        meta[metric_name] = {
            "name": get_metric_display_name(metric_name),
            "unit": unit,
            "ref": ref_str,
            "ref_lo": ref_lo,
            "ref_hi": ref_hi,
            "status": status,
            "note": note,
        }

    return meta, cnt_ok, cnt_warn, cnt_fail


def write_plausibility_html(
    out_path: Path,
    time_series_df,
    timepoint_cols: List[str],
    *,
    drug_name: str = "Unknown",
    drug_smiles: str = "",
    age: int = 0,
    sex: str = "M",
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    total_seconds: int = 10800,
    interval_seconds: int = 10,
    dosage_mg: float = 10.0,
    patient_id: int = 1,
) -> None:
    """
    Write a single HTML file with red/black design: summary cards (✓/⚠/✗),
    filter tabs, detailed table, and Chart.js time-series charts per metric.
    """
    timepoint_cols = sorted(
        [c for c in timepoint_cols if c.endswith("sec")],
        key=lambda x: int(x.replace("sec", "")),
    )
    times_sec = [int(c.replace("sec", "")) for c in timepoint_cols]
    meta, cnt_ok, cnt_warn, cnt_fail = build_plausibility_meta(time_series_df, timepoint_cols)

    # Build rawData and meta for JS (match reference structure)
    raw_data_js = []
    meta_js = {}
    for metric_name in time_series_df["metric_name"].unique():
        row = time_series_df[time_series_df["metric_name"] == metric_name].iloc[0]
        baseline = float(row["baseline"])
        final_delta = float(row["final_delta"])
        final_val = baseline + final_delta
        series = [baseline]
        for c in timepoint_cols:
            if c in row:
                v = row[c]
                series.append(float(v) if v == v else baseline)
        # Ensure series length matches [0] + times_sec (1 + len(times_sec) points)
        n_expected = len(times_sec) + 1
        if len(series) < n_expected:
            series = [baseline] + [baseline + (final_delta * (t / max(times_sec[-1], 1))) for t in times_sec]
        series = series[:n_expected]
        m = meta[metric_name]
        raw_data_js.append({
            "id": metric_name,
            "base": round(baseline, 4),
            "final": round(final_val, 4),
            "delta": round(final_delta, 4),
            "series": [round(x, 4) for x in series[: len(times_sec) + 1]],
        })
        meta_js[metric_name] = {
            "name": m["name"],
            "unit": m["unit"],
            "ref": m["ref"],
            "refLo": m["ref_lo"],
            "refHi": m["ref_hi"],
            "status": m["status"],
            "note": m["note"],
        }

    time_labels_js = []
    for t in [0] + times_sec:
        if t == 0:
            time_labels_js.append("0s")
        elif t % 60 == 0:
            time_labels_js.append(f"{t // 60}m")
        else:
            time_labels_js.append(f"{t}s")

    total_hours = total_seconds / 3600
    interval_str = f"{interval_seconds} sec" if interval_seconds < 60 else f"{interval_seconds // 60} min"
    patient_parts = [f"ID #{patient_id}"]
    if age:
        patient_parts.append(str(age))
    patient_parts.append(sex)
    if height_cm is not None:
        patient_parts.append(f"{height_cm:.0f} cm")
    if weight_kg is not None:
        patient_parts.append(f"{weight_kg:.1f} kg")
    patient_line = " — ".join(patient_parts)

    # Red/black theme (user requested cool red/black)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Plausibility Dashboard — {drug_name}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0a0a;
    --surface: #141414;
    --border: #2a1515;
    --text: #e8d4d4;
    --muted: #806060;
    --green: #2ecc8a;
    --yellow: #f0b429;
    --red: #ff4d6d;
    --blue: #ff6b6b;
    --accent: #cc3333;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    min-height: 100vh;
  }}
  body::before {{
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; position: relative; z-index: 1; }}

  .header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 40px; border-bottom: 1px solid var(--border); padding-bottom: 28px; }}
  .header-title h1 {{ font-family: 'Fraunces', serif; font-size: 28px; font-weight: 700; letter-spacing: -0.5px; color: #fff; }}
  .header-title p {{ color: var(--muted); margin-top: 6px; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }}
  .patient-card {{ background: var(--surface); border: 1px solid var(--border); padding: 14px 20px; border-radius: 6px; text-align: right; }}
  .patient-card .label {{ color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; }}
  .patient-card .value {{ color: #fff; font-size: 12px; margin-top: 3px; }}

  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 36px; }}
  .summary-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 18px; }}
  .summary-card .sc-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 8px; }}
  .summary-card .sc-value {{ font-size: 26px; font-weight: 700; font-family: 'Fraunces', serif; }}
  .summary-card.ok .sc-value {{ color: var(--green); }}
  .summary-card.warn .sc-value {{ color: var(--yellow); }}
  .summary-card.fail .sc-value {{ color: var(--red); }}
  .summary-card .sc-sub {{ font-size: 10px; color: var(--muted); margin-top: 4px; }}

  .filter-bar {{ display: flex; gap: 8px; margin-bottom: 24px; flex-wrap: wrap; }}
  .filter-btn {{ background: var(--surface); border: 1px solid var(--border); color: var(--muted); padding: 6px 14px; border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; transition: all 0.2s; }}
  .filter-btn:hover, .filter-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}

  .plausibility-table {{ width: 100%; border-collapse: collapse; margin-bottom: 48px; }}
  .plausibility-table th {{ text-align: left; padding: 8px 12px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); border-bottom: 1px solid var(--border); }}
  .plausibility-table td {{ padding: 10px 12px; border-bottom: 1px solid #1a1010; vertical-align: middle; }}
  .plausibility-table tr:hover td {{ background: #1a1212; }}
  .metric-name {{ color: var(--blue); font-weight: 600; cursor: pointer; }}
  .metric-name:hover {{ text-decoration: underline; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; }}
  .badge-ok {{ background: rgba(46,204,138,0.15); color: var(--green); }}
  .badge-warn {{ background: rgba(240,180,41,0.15); color: var(--yellow); }}
  .badge-fail {{ background: rgba(255,77,109,0.15); color: var(--red); }}
  .delta-pos {{ color: var(--red); }}
  .delta-neg {{ color: var(--green); }}
  .ref-range {{ color: var(--muted); font-size: 11px; }}
  .note-cell {{ color: var(--muted); font-size: 11px; max-width: 280px; line-height: 1.5; }}
  .note-cell.critical {{ color: #ff8fab; }}
  .hide {{ display: none; }}

  .section-title {{ font-family: 'Fraunces', serif; font-size: 18px; color: #fff; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }}
  .section-title::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; margin-bottom: 48px; }}
  .chart-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px; }}
  .chart-card.card-fail {{ border-color: rgba(255,77,109,0.5); }}
  .chart-card.card-warn {{ border-color: rgba(240,180,41,0.4); }}
  .chart-card.card-ok {{ border-color: var(--border); }}
  .chart-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .chart-title {{ font-weight: 600; font-size: 12px; color: #fff; }}
  .chart-subtitle {{ font-size: 10px; color: var(--muted); margin-top: 2px; }}
  .chart-delta {{ font-size: 11px; font-weight: 700; }}
  canvas {{ max-height: 160px; }}

  .footer {{ border-top: 1px solid var(--border); padding-top: 20px; color: var(--muted); font-size: 10px; display: flex; justify-content: space-between; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <div class="header-title">
      <h1>Model Plausibility Assessment</h1>
      <p>Drug: {drug_name} ({dosage_mg} mg) · Simulation: {total_hours:.1f} h · Checkpoints: {interval_str}</p>
    </div>
    <div class="patient-card">
      <div class="label">Patient</div>
      <div class="value">{patient_line}</div>
      <div class="label" style="margin-top:8px">Drug SMILES</div>
      <div class="value" style="font-size:10px;max-width:280px;word-break:break-all">{drug_smiles or "—"}</div>
    </div>
  </div>

  <div class="summary-grid">
    <div class="summary-card ok">
      <div class="sc-label">Safe ✓</div>
      <div class="sc-value" id="cnt-ok">{cnt_ok}</div>
      <div class="sc-sub">within expected range/direction</div>
    </div>
    <div class="summary-card warn">
      <div class="sc-label">Marginal ⚠</div>
      <div class="sc-value" id="cnt-warn">{cnt_warn}</div>
      <div class="sc-sub">unexpected magnitude or direction</div>
    </div>
    <div class="summary-card fail">
      <div class="sc-label">Hazardous ✗</div>
      <div class="sc-value" id="cnt-fail">{cnt_fail}</div>
      <div class="sc-sub">pharmacologically inconsistent</div>
    </div>
  </div>

  <div class="filter-bar">
    <button class="filter-btn active" data-filter="all" onclick="filterTable(this,'all')">All metrics</button>
    <button class="filter-btn" data-filter="ok" onclick="filterTable(this,'ok')">✓ Safe</button>
    <button class="filter-btn" data-filter="warn" onclick="filterTable(this,'warn')">⚠ Marginal</button>
    <button class="filter-btn" data-filter="fail" onclick="filterTable(this,'fail')">✗ Hazardous</button>
  </div>

  <div class="section-title">Detailed Biomarker Assessment</div>
  <table class="plausibility-table" id="plausTable">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Full Name</th>
        <th>Baseline</th>
        <th>Final</th>
        <th>Δ (absolute)</th>
        <th>Ref. Range</th>
        <th>Status</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>

  <div class="section-title">Time-Series Trajectories</div>
  <div class="charts-grid" id="chartsGrid"></div>

  <div class="footer">
    <span>Reference ranges: NHANES / clinical laboratory standards. Assessment based on physiological limits and expected drug response.</span>
    <span>BloodTwin · Plausibility Report</span>
  </div>
</div>

<script>
const times = {json.dumps([0] + times_sec)};
const timeLabels = {json.dumps(time_labels_js)};
const rawData = {json.dumps(raw_data_js)};
const meta = {json.dumps(meta_js)};

const tbody = document.getElementById('tableBody');
rawData.forEach(d => {{
  const m = meta[d.id];
  if(!m) return;
  const pct = (d.base !== 0 ? ((d.delta/d.base)*100).toFixed(1) : '—');
  const deltaClass = d.delta > 0 ? 'delta-pos' : 'delta-neg';
  const badgeClass = `badge-${{m.status}}`;
  const badgeText = m.status==='ok' ? '✓ Safe' : m.status==='warn' ? '⚠ Marginal' : '✗ Hazardous';
  const tr = document.createElement('tr');
  tr.dataset.status = m.status;
  tr.innerHTML = `
    <td><span class="metric-name" onclick="scrollToChart('${{d.id}}')">${{d.id}}</span></td>
    <td style="color:#fff">${{m.name}}</td>
    <td>${{d.base.toFixed(3)}}</td>
    <td>${{d.final.toFixed(3)}}</td>
    <td class="${{deltaClass}}">${{d.delta>0?'+':''}}${{d.delta.toFixed(3)}} (${{d.delta>0?'+':''}}${{pct}}%)</td>
    <td class="ref-range">${{m.ref}} ${{m.unit}}</td>
    <td><span class="badge ${{badgeClass}}">${{badgeText}}</span></td>
    <td class="note-cell ${{m.status==='fail'?'critical':''}}">${{m.note}}</td>
  `;
  tbody.appendChild(tr);
}});

function filterTable(btn, filter) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#tableBody tr').forEach(tr => {{
    tr.classList.toggle('hide', filter !== 'all' && tr.dataset.status !== filter);
  }});
  document.querySelectorAll('.chart-card').forEach(card => {{
    const s = card.dataset.status;
    card.classList.toggle('hide', filter !== 'all' && s !== filter);
  }});
}}

const grid = document.getElementById('chartsGrid');
const chartInstances = {{}};

rawData.forEach(d => {{
  const m = meta[d.id];
  if(!m) return;
  const cardClass = `card-${{m.status}}`;
  const card = document.createElement('div');
  card.className = `chart-card ${{cardClass}}`;
  card.id = `chart-${{d.id}}`;
  card.dataset.status = m.status;

  const pct = (d.base !== 0 ? ((d.delta/d.base)*100).toFixed(1) : '—');
  const deltaColor = d.delta > 0 ? '#ff4d6d' : '#2ecc8a';
  const badgeClass = `badge-${{m.status}}`;
  const badgeText = m.status==='ok' ? '✓' : m.status==='warn' ? '⚠' : '✗';

  card.innerHTML = `
    <div class="chart-header">
      <div>
        <div class="chart-title">${{d.id}} <span class="badge ${{badgeClass}}" style="font-size:9px">${{badgeText}}</span></div>
        <div class="chart-subtitle">${{m.name}} · ${{m.unit}}</div>
      </div>
      <div class="chart-delta" style="color:${{deltaColor}}">${{d.delta>0?'+':''}}${{pct}}%</div>
    </div>
    <canvas id="canvas-${{d.id}}"></canvas>
  `;
  grid.appendChild(card);

  const refLoArr = m.refLo != null ? Array(times.length).fill(m.refLo) : [];
  const refHiArr = m.refHi != null ? Array(times.length).fill(m.refHi) : [];
  const lineColor = m.status === 'fail' ? '#ff4d6d' : m.status === 'warn' ? '#f0b429' : '#4da6ff';

  const datasets = [
    {{ label: d.id, data: d.series, borderColor: lineColor, backgroundColor: lineColor + '15', fill: true, tension: 0.4, pointRadius: 2, borderWidth: 2 }}
  ];
  if (m.refLo != null && m.refHi != null) {{
    datasets.push({{ label: 'Ref Hi', data: refHiArr, borderColor: '#2ecc8a40', borderDash: [4,4], borderWidth: 1, pointRadius: 0, fill: false }});
    datasets.push({{ label: 'Ref Lo', data: refLoArr, borderColor: '#2ecc8a40', borderDash: [4,4], borderWidth: 1, pointRadius: 0, fill: false }});
  }}

  chartInstances[d.id] = new Chart(document.getElementById(`canvas-${{d.id}}`), {{
    type: 'line',
    data: {{ labels: timeLabels, datasets }},
    options: {{
      animation: {{ duration: 400 }},
      plugins: {{ legend: {{ display: false }}, tooltip: {{
        backgroundColor: '#111520',
        borderColor: '#1e2535',
        borderWidth: 1,
        titleColor: '#4da6ff',
        bodyColor: '#d4daf0',
        callbacks: {{ label: ctx => ` ${{ctx.dataset.label === d.id ? ctx.parsed.y.toFixed(4) + ' ' + m.unit : ''}}` }}
      }}}},
      scales: {{
        x: {{ ticks: {{ color: '#5a6280', font: {{ size: 9, family: 'JetBrains Mono' }} }}, grid: {{ color: '#161b26' }} }},
        y: {{ ticks: {{ color: '#5a6280', font: {{ size: 9, family: 'JetBrains Mono' }} }}, grid: {{ color: '#161b26' }} }}
      }}
    }}
  }});
}});

function scrollToChart(id) {{
  const el = document.getElementById(`chart-${{id}}`);
  if(el) el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
}}
</script>
</body>
</html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
