#!/usr/bin/env python3
"""Build self-contained QB chance-creation dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = Path(__file__).resolve().parent / "output"


def build_qb_games() -> pd.DataFrame:
    routes = pd.read_csv(OUTPUT / "route_chances.csv")
    plays = pd.read_csv(DATA / "plays.csv")
    scouting = pd.read_csv(DATA / "pffScoutingData.csv")
    players = pd.read_csv(DATA / "players.csv")
    games = pd.read_csv(DATA / "games.csv")

    play_chances = (
        routes.groupby(["gameId", "playId"], as_index=False)
        .agg(
            team_play_window=("team_play_window", "max"),
            actionable_chance=("actionable_chance", "max"),
            pressured=("pressured", "max"),
        )
    )
    passers = scouting.loc[
        scouting["pff_role"].eq("Pass"), ["gameId", "playId", "nflId"]
    ].rename(columns={"nflId": "qbId"})
    passers = passers.drop_duplicates(["gameId", "playId"])

    play_data = plays.merge(play_chances, on=["gameId", "playId"], how="inner")
    play_data = play_data.merge(passers, on=["gameId", "playId"], how="inner")
    play_data["first_down_success"] = (
        play_data["prePenaltyPlayResult"] >= play_data["yardsToGo"]
    ).astype(int)

    qb_games = (
        play_data.groupby(
            ["gameId", "qbId", "possessionTeam", "defensiveTeam"], as_index=False
        )
        .agg(
            dropbacks=("playId", "size"),
            receiver_windows=("team_play_window", "sum"),
            actionable_chances=("actionable_chance", "sum"),
            pressured_dropbacks=("pressured", "sum"),
            successful_dropbacks=("first_down_success", "sum"),
            total_yards=("prePenaltyPlayResult", "sum"),
        )
    )
    qb_games["window_rate"] = 100 * qb_games["receiver_windows"] / qb_games["dropbacks"]
    qb_games["actionable_chance_rate"] = (
        100 * qb_games["actionable_chances"] / qb_games["dropbacks"]
    )
    qb_games["actionability_rate"] = (
        100
        * qb_games["actionable_chances"]
        / qb_games["receiver_windows"].replace(0, np.nan)
    )
    qb_games["team_chances"] = qb_games["actionable_chances"]
    qb_games["chance_rate"] = qb_games["actionable_chance_rate"]
    qb_games["first_down_rate"] = (
        100 * qb_games["successful_dropbacks"] / qb_games["dropbacks"]
    )
    qb_games["yards_per_dropback"] = qb_games["total_yards"] / qb_games["dropbacks"]

    qb_games = qb_games.merge(
        players[["nflId", "displayName"]].rename(
            columns={"nflId": "qbId", "displayName": "quarterback"}
        ),
        on="qbId",
        how="left",
    ).merge(games[["gameId", "week"]], on="gameId", how="left")

    qb_games["qb_key"] = qb_games["qbId"].astype(int).astype(str)
    columns = [
        "qb_key",
        "quarterback",
        "possessionTeam",
        "defensiveTeam",
        "gameId",
        "week",
        "dropbacks",
        "receiver_windows",
        "window_rate",
        "actionable_chances",
        "actionable_chance_rate",
        "actionability_rate",
        "pressured_dropbacks",
        "team_chances",
        "chance_rate",
        "first_down_rate",
        "yards_per_dropback",
    ]
    return qb_games[columns].sort_values(["quarterback", "week"])


def build_defense_summary() -> pd.DataFrame:
    """Aggregate opponent chances and passing outcomes by defense."""
    team_games = pd.read_csv(OUTPUT / "team_game_summary.csv")
    defense = (
        team_games.groupby("defensiveTeam", as_index=False)
        .agg(
            games=("gameId", "nunique"),
            dropbacks_faced=("dropbacks", "sum"),
            windows_allowed=("team_windows", "sum"),
            actionable_chances_allowed=("actionable_chances", "sum"),
            first_downs_conceded=("successful_dropbacks", "sum"),
            touchdowns_conceded=("touchdowns", "sum"),
            yards_allowed=("total_passing_yards", "sum"),
        )
    )
    defense["window_rate_allowed"] = (
        100 * defense["windows_allowed"] / defense["dropbacks_faced"]
    )
    defense["actionable_chance_rate_allowed"] = (
        100 * defense["actionable_chances_allowed"] / defense["dropbacks_faced"]
    )
    defense["actionability_rate_allowed"] = (
        100
        * defense["actionable_chances_allowed"]
        / defense["windows_allowed"].replace(0, np.nan)
    )
    defense["chances_allowed"] = defense["actionable_chances_allowed"]
    defense["chance_rate_allowed"] = defense["actionable_chance_rate_allowed"]
    defense["first_down_rate_conceded"] = (
        100 * defense["first_downs_conceded"] / defense["dropbacks_faced"]
    )
    defense["touchdown_rate_conceded"] = (
        100 * defense["touchdowns_conceded"] / defense["dropbacks_faced"]
    )
    defense["yards_per_dropback_allowed"] = (
        defense["yards_allowed"] / defense["dropbacks_faced"]
    )
    return defense.sort_values("actionable_chance_rate_allowed")


def dashboard_html(records: list[dict], default_qb: str) -> str:
    data_json = json.dumps(records, separators=(",", ":"), allow_nan=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QB Chance Conversion</title>
<style>
:root {{ --ink:#17222d; --muted:#617181; --paper:#f5f1e8; --card:#fffdf7;
  --blue:#176b87; --orange:#d26a3a; --grid:#d7d2c8; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.4 system-ui,sans-serif; }}
main {{ max-width:1180px; margin:auto; padding:28px 22px 48px; }}
h1 {{ margin:0 0 4px; font-size:30px; }}
.subtitle {{ color:var(--muted); margin-bottom:20px; }}
.controls {{ display:flex; gap:16px; align-items:end; flex-wrap:wrap; margin-bottom:18px; }}
label {{ display:grid; gap:5px; font-weight:650; }}
select {{ min-width:280px; padding:9px 12px; border:1px solid #a8ada9; border-radius:7px; background:white; font-size:15px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:16px; }}
.card,.plot {{ background:var(--card); border:1px solid #ded9cf; border-radius:10px; box-shadow:0 2px 8px #25313b10; }}
.card {{ padding:13px 15px; }}
.card small {{ display:block; color:var(--muted); }}
.card strong {{ display:block; font-size:22px; margin-top:2px; }}
.plots {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.plot {{ padding:14px; min-width:0; }}
.plot h2 {{ font-size:17px; margin:0 0 4px; }}
.plot p {{ color:var(--muted); margin:0 0 4px; font-size:13px; }}
svg {{ width:100%; height:auto; overflow:visible; }}
.axis {{ stroke:var(--ink); stroke-width:1; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.tick,.label {{ fill:var(--muted); font-size:11px; }}
.point {{ fill:var(--blue); stroke:white; stroke-width:1.5; cursor:pointer; }}
.point:hover {{ fill:var(--orange); r:7; }}
.trend {{ stroke:var(--orange); stroke-width:2.2; stroke-dasharray:6 4; }}
.empty {{ color:var(--muted); padding:80px 20px; text-align:center; }}
.note {{ margin-top:14px; color:var(--muted); font-size:13px; }}
@media(max-width:800px) {{ .plots {{ grid-template-columns:1fr; }} .cards {{ grid-template-columns:1fr 1fr; }} }}
</style>
</head>
<body><main>
<h1>QB Chance Conversion</h1>
<div class="subtitle">One point per QB-game. Chance Created = qualifying receiver window. Actionable Chance = chance plus clean pocket.</div>
<div class="controls">
  <label>Quarterback<select id="qb"></select></label>
  <label>X-axis metric<select id="metric"><option value="window_rate">Chance Created</option><option value="actionable_chance_rate" selected>Actionable Chance</option></select></label>
</div>
<section class="cards" id="cards"></section>
<section class="plots">
  <div class="plot"><h2 id="first-title"></h2><p id="first-meta"></p><div id="first-plot"></div></div>
  <div class="plot"><h2 id="yards-title"></h2><p id="yards-meta"></p><div id="yards-plot"></div></div>
</section>
<div class="note">Backup and partial-game samples can be tiny. Correlation describes association, not pure QB causality.</div>
</main>
<script>
const DATA={data_json};
const DEFAULT_QB={json.dumps(default_qb)};
const select=document.getElementById('qb');
const metricSelect=document.getElementById('metric');
const totals={{}};
for(const d of DATA) {{
  if(!totals[d.qb_key]) totals[d.qb_key]={{name:d.quarterback,team:d.possessionTeam,n:0,games:0}};
  totals[d.qb_key].n+=d.dropbacks; totals[d.qb_key].games+=1;
}}
Object.entries(totals).sort((a,b)=>b[1].n-a[1].n).forEach(([key,v])=>{{
  const option=document.createElement('option'); option.value=key;
  option.textContent=`${{v.name}} (${{v.team}}) — ${{v.n}} dropbacks`;
  select.appendChild(option);
}});
select.value=DEFAULT_QB;

function correlation(rows,xKey,y) {{
  if(rows.length<2) return NaN;
  const xs=rows.map(d=>d[xKey]), ys=rows.map(d=>d[y]);
  const mx=xs.reduce((a,b)=>a+b,0)/xs.length, my=ys.reduce((a,b)=>a+b,0)/ys.length;
  const num=xs.reduce((s,x,i)=>s+(x-mx)*(ys[i]-my),0);
  const dx=Math.sqrt(xs.reduce((s,x)=>s+(x-mx)**2,0));
  const dy=Math.sqrt(ys.reduce((s,v)=>s+(v-my)**2,0));
  return dx&&dy ? num/(dx*dy) : NaN;
}}
function extent(values) {{
  let lo=Math.min(...values), hi=Math.max(...values);
  if(lo===hi) {{ lo-=1; hi+=1; }}
  const pad=(hi-lo)*0.14; return [lo-pad,hi+pad];
}}
function plot(target,rows,xKey,xLabel,yKey,yLabel) {{
  if(!rows.length) {{ document.getElementById(target).innerHTML='<div class="empty">No games</div>'; return; }}
  const W=520,H=390,m={{l:62,r:20,t:20,b:55}};
  const [xmin,xmax]=extent(rows.map(d=>d[xKey]));
  const [ymin,ymax]=extent(rows.map(d=>d[yKey]));
  const sx=x=>m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r);
  const sy=y=>H-m.b-(y-ymin)/(ymax-ymin)*(H-m.t-m.b);
  const ticks=(lo,hi,n=5)=>Array.from({{length:n}},(_,i)=>lo+(hi-lo)*i/(n-1));
  let svg=`<svg viewBox="0 0 ${{W}} ${{H}}" role="img">`;
  for(const y of ticks(ymin,ymax)) svg+=`<line class="grid" x1="${{m.l}}" x2="${{W-m.r}}" y1="${{sy(y)}}" y2="${{sy(y)}}"/><text class="tick" x="${{m.l-8}}" y="${{sy(y)+4}}" text-anchor="end">${{y.toFixed(1)}}</text>`;
  for(const x of ticks(xmin,xmax)) svg+=`<text class="tick" x="${{sx(x)}}" y="${{H-m.b+19}}" text-anchor="middle">${{x.toFixed(1)}}</text>`;
  svg+=`<line class="axis" x1="${{m.l}}" x2="${{m.l}}" y1="${{m.t}}" y2="${{H-m.b}}"/><line class="axis" x1="${{m.l}}" x2="${{W-m.r}}" y1="${{H-m.b}}" y2="${{H-m.b}}"/>`;
  if(rows.length>=2) {{
    const mx=rows.reduce((s,d)=>s+d[xKey],0)/rows.length;
    const my=rows.reduce((s,d)=>s+d[yKey],0)/rows.length;
    const den=rows.reduce((s,d)=>s+(d[xKey]-mx)**2,0);
    if(den) {{ const slope=rows.reduce((s,d)=>s+(d[xKey]-mx)*(d[yKey]-my),0)/den; const intercept=my-slope*mx;
      svg+=`<line class="trend" x1="${{sx(xmin)}}" y1="${{sy(intercept+slope*xmin)}}" x2="${{sx(xmax)}}" y2="${{sy(intercept+slope*xmax)}}"/>`; }}
  }}
  for(const d of rows) {{
    const tip=`Week ${{d.week}} vs ${{d.defensiveTeam}}\n${{d.dropbacks}} dropbacks\nWindow rate: ${{d.window_rate.toFixed(1)}}\nActionable rate: ${{d.chance_rate.toFixed(1)}}\n${{yLabel}}: ${{d[yKey].toFixed(1)}}`;
    svg+=`<circle class="point" cx="${{sx(d[xKey])}}" cy="${{sy(d[yKey])}}" r="5"><title>${{tip}}</title></circle>`;
    svg+=`<text class="label" x="${{sx(d[xKey])+7}}" y="${{sy(d[yKey])-7}}">W${{d.week}}</text>`;
  }}
  svg+=`<text class="label" x="${{(m.l+W-m.r)/2}}" y="${{H-10}}" text-anchor="middle">${{xLabel}} per 100 dropbacks</text>`;
  svg+=`<text class="label" transform="translate(15 ${{(m.t+H-m.b)/2}}) rotate(-90)" text-anchor="middle">${{yLabel}}</text></svg>`;
  document.getElementById(target).innerHTML=svg;
}}
function render() {{
  const rows=DATA.filter(d=>d.qb_key===select.value);
  const xKey=metricSelect.value;
  const xLabel=xKey==='window_rate'?'Chances Created':'Actionable Chances';
  const dropbacks=rows.reduce((s,d)=>s+d.dropbacks,0);
  const chances=rows.reduce((s,d)=>s+d.team_chances,0);
  const windows=rows.reduce((s,d)=>s+d.receiver_windows,0);
  const first=rows.reduce((s,d)=>s+d.first_down_rate*d.dropbacks,0)/dropbacks;
  const yards=rows.reduce((s,d)=>s+d.yards_per_dropback*d.dropbacks,0)/dropbacks;
  document.getElementById('cards').innerHTML=`
    <div class="card"><small>Games</small><strong>${{rows.length}}</strong></div>
    <div class="card"><small>Dropbacks</small><strong>${{dropbacks}}</strong></div>
    <div class="card"><small>Window rate</small><strong>${{(100*windows/dropbacks).toFixed(1)}}</strong></div>
    <div class="card"><small>Actionable rate</small><strong>${{(100*chances/dropbacks).toFixed(1)}}</strong></div>
    <div class="card"><small>Actionability</small><strong>${{windows?(100*chances/windows).toFixed(1):'—'}}%</strong></div>
    <div class="card"><small>First-down rate</small><strong>${{first.toFixed(1)}}</strong></div>`;
  const r1=correlation(rows,xKey,'first_down_rate'), r2=correlation(rows,xKey,'yards_per_dropback');
  document.getElementById('first-title').textContent=`${{xLabel}} vs first-down rate`;
  document.getElementById('yards-title').textContent=`${{xLabel}} vs yards per dropback`;
  document.getElementById('first-meta').textContent=`Game correlation: ${{Number.isFinite(r1)?r1.toFixed(2):'not enough data'}}`;
  document.getElementById('yards-meta').textContent=`Game correlation: ${{Number.isFinite(r2)?r2.toFixed(2):'not enough data'}}`;
  plot('first-plot',rows,xKey,xLabel,'first_down_rate','First downs per 100 dropbacks');
  plot('yards-plot',rows,xKey,xLabel,'yards_per_dropback','Yards per dropback');
}}
select.addEventListener('change',render); metricSelect.addEventListener('change',render); render();
</script></body></html>"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    qb_games = build_qb_games()
    qb_games.to_csv(OUTPUT / "qb_game_summary.csv", index=False)
    defense = build_defense_summary()
    defense.to_csv(OUTPUT / "defense_summary.csv", index=False)

    totals = qb_games.groupby("qb_key")["dropbacks"].sum()
    default_qb = str(totals.idxmax())
    records = qb_games.replace({np.nan: None}).to_dict(orient="records")
    (OUTPUT / "qb_dashboard.html").write_text(
        dashboard_html(records, default_qb), encoding="utf-8"
    )
    print(f"Wrote {OUTPUT / 'qb_game_summary.csv'}")
    print(f"Wrote {OUTPUT / 'defense_summary.csv'}")
    print(f"Wrote {OUTPUT / 'qb_dashboard.html'}")
    print(f"QB-games: {len(qb_games):,}; QBs: {qb_games['qbId' if 'qbId' in qb_games else 'qb_key'].nunique():,}")


if __name__ == "__main__":
    main()
