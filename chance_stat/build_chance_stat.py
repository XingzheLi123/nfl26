#!/usr/bin/env python3
"""Build receiver chance-creation scores from tracking data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def preferred_event_frame(play: pd.DataFrame, primary: str, fallback: str) -> int | None:
    """Return first frame for preferred event label, then automated fallback."""
    for label in (primary, fallback):
        frames = play.loc[play["event"].eq(label), "frameId"]
        if not frames.empty:
            return int(frames.min())
    return None


def score_play(
    tracking: pd.DataFrame,
    route_ids: set[int],
    coverage_ids: set[int],
    yards_to_go: float,
    separation_threshold: float,
    projected_separation_threshold: float,
    projection_seconds: float,
    required_frames: int,
    fallback_after_snap: int,
) -> list[dict]:
    """Return one score row per route runner for one play."""
    snap = preferred_event_frame(tracking, "ball_snap", "autoevent_ballsnap")
    if snap is None:
        return []

    release = preferred_event_frame(tracking, "pass_forward", "autoevent_passforward")
    sack = preferred_event_frame(tracking, "qb_sack", "qb_strip_sack")
    last_frame = int(tracking["frameId"].max())
    decision = release if release is not None else sack
    if decision is None:
        decision = min(snap + fallback_after_snap, last_frame)
    decision = min(decision, last_frame)

    football = tracking.loc[
        tracking["team"].eq("football") & tracking["frameId"].eq(snap), "x"
    ]
    if football.empty:
        return []
    line_x = float(football.iloc[0])
    direction = tracking["playDirection"].dropna().iloc[0]
    direction_sign = 1.0 if direction == "right" else -1.0

    available_frames = sorted(
        tracking.loc[tracking["frameId"].between(snap, decision), "frameId"].unique()
    )
    decision_frames = available_frames[-required_frames:]
    window = tracking.loc[
        tracking["frameId"].isin(decision_frames)
        & tracking["nflId"].notna()
        & tracking["nflId"].isin(route_ids | coverage_ids),
        ["frameId", "nflId", "x", "y", "s", "dir"],
    ].copy()
    if window.empty:
        return []
    window["nflId"] = window["nflId"].astype(int)

    observations: dict[int, list[tuple[int, bool, float, float, float]]] = {
        receiver_id: [] for receiver_id in route_ids
    }
    for frame_id, frame in window.groupby("frameId", sort=True):
        receivers = frame.loc[frame["nflId"].isin(route_ids)]
        defenders = frame.loc[frame["nflId"].isin(coverage_ids)]
        if receivers.empty or defenders.empty:
            continue

        receiver_xy = receivers[["x", "y"]].to_numpy(float)
        defender_xy = defenders[["x", "y"]].to_numpy(float)
        distances = np.sqrt(
            ((receiver_xy[:, None, :] - defender_xy[None, :, :]) ** 2).sum(axis=2)
        )
        nearest = distances.min(axis=1)

        receiver_speed = receivers["s"].fillna(0).to_numpy(float)
        receiver_dir = np.deg2rad(receivers["dir"].fillna(0).to_numpy(float))
        defender_speed = defenders["s"].fillna(0).to_numpy(float)
        defender_dir = np.deg2rad(defenders["dir"].fillna(0).to_numpy(float))
        receiver_future = receiver_xy + projection_seconds * np.column_stack(
            (receiver_speed * np.sin(receiver_dir), receiver_speed * np.cos(receiver_dir))
        )
        defender_future = defender_xy + projection_seconds * np.column_stack(
            (defender_speed * np.sin(defender_dir), defender_speed * np.cos(defender_dir))
        )
        projected_distances = np.sqrt(
            ((receiver_future[:, None, :] - defender_future[None, :, :]) ** 2).sum(axis=2)
        )
        projected_nearest = projected_distances.min(axis=1)
        depths = direction_sign * (receivers["x"].to_numpy(float) - line_x)

        for receiver_id, separation, projected_separation, depth in zip(
            receivers["nflId"].astype(int), nearest, projected_nearest, depths
        ):
            qualifies = (
                separation >= separation_threshold
                and projected_separation >= projected_separation_threshold
                and depth >= yards_to_go
            )
            observations[int(receiver_id)].append(
                (
                    int(frame_id),
                    bool(qualifies),
                    float(separation),
                    float(projected_separation),
                    float(depth),
                )
            )

    rows = []
    qualifying_ids = []
    receiver_details = {}
    for receiver_id in sorted(route_ids):
        obs = observations.get(receiver_id, [])
        flags = [item[1] for item in obs]
        needed = min(required_frames, len(flags)) if flags else required_frames
        receiver_window = bool(flags) and len(flags) >= needed and all(flags[-needed:])
        if receiver_window:
            qualifying_ids.append(receiver_id)

        eligible = [item for item in obs if item[4] >= yards_to_go]
        best = max(eligible, key=lambda item: item[2]) if eligible else None
        receiver_details[receiver_id] = {
            "receiver_window": int(receiver_window),
            "raw_chance": int(receiver_window),
            "best_frame": best[0] if best else np.nan,
            "best_separation": best[2] if best else np.nan,
            "projected_separation_at_best": best[3] if best else np.nan,
            "route_depth_at_best": best[4] if best else np.nan,
        }

    chance_count = len(qualifying_ids)
    for receiver_id in sorted(route_ids):
        details = receiver_details[receiver_id]
        rows.append(
            {
                "nflId": receiver_id,
                "snap_frame": snap,
                "decision_frame": decision,
                "decision_event": (
                    "pass_release" if release is not None else "sack" if sack is not None else "fixed_time"
                ),
                **details,
                "chance_receivers_on_play": chance_count,
                "window_credit": (
                    1.0 / chance_count if details["raw_chance"] and chance_count else 0.0
                ),
                "chance_credit": (
                    1.0 / chance_count if details["raw_chance"] and chance_count else 0.0
                ),
                "team_play_window": int(chance_count > 0),
            }
        )
    return rows


def build(args: argparse.Namespace) -> None:
    games = pd.read_csv(DATA / "games.csv")
    players = pd.read_csv(DATA / "players.csv")
    plays = pd.read_csv(DATA / "plays.csv")
    scouting = pd.read_csv(DATA / "pffScoutingData.csv")

    play_context = plays.set_index(["gameId", "playId"])
    role_groups = scouting.groupby(["gameId", "playId", "pff_role"])["nflId"].agg(set)
    pressure_columns = ["pff_hurry", "pff_hit", "pff_sack"]
    pressure_by_play = (
        scouting.groupby(["gameId", "playId"])[pressure_columns]
        .max()
        .fillna(0)
        .max(axis=1)
        .astype(int)
    )

    route_rows: list[dict] = []
    tracking_files = sorted((DATA / "tracking").glob("tracking_*.csv"))
    for file_number, path in enumerate(tracking_files, start=1):
        tracking = pd.read_csv(path)
        game_id = int(tracking["gameId"].iloc[0])
        for play_id, play_tracking in tracking.groupby("playId", sort=False):
            key = (game_id, int(play_id))
            if key not in play_context.index:
                continue
            try:
                route_ids = {
                    int(value) for value in role_groups.loc[key + ("Pass Route",)]
                }
                coverage_ids = {
                    int(value) for value in role_groups.loc[key + ("Coverage",)]
                }
            except KeyError:
                continue

            context = play_context.loc[key]
            scored = score_play(
                play_tracking,
                route_ids,
                coverage_ids,
                float(context["yardsToGo"]),
                args.separation,
                args.projected_separation,
                args.projection_seconds,
                args.required_frames,
                args.fallback_after_snap,
            )
            pressured = int(pressure_by_play.get(key, 0))
            for row in scored:
                actionable = int(row["team_play_window"] and not pressured)
                route_rows.append(
                    {
                        "gameId": game_id,
                        "playId": int(play_id),
                        "possessionTeam": context["possessionTeam"],
                        "defensiveTeam": context["defensiveTeam"],
                        "down": context["down"],
                        "yardsToGo": context["yardsToGo"],
                        "playResult": context["playResult"],
                        "passResult": context["passResult"],
                        "pressured": pressured,
                        "actionable_chance": actionable,
                        "team_play_chance": actionable,
                        **row,
                    }
                )
        print(f"[{file_number:3d}/{len(tracking_files)}] {path.name}", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    routes = pd.DataFrame(route_rows)
    routes = routes.merge(
        players[["nflId", "displayName", "officialPosition"]], on="nflId", how="left"
    ).merge(games[["gameId", "week"]], on="gameId", how="left")
    routes.to_csv(output / "route_chances.csv", index=False)

    receiver_scores = (
        routes.groupby(["nflId", "displayName", "officialPosition"], dropna=False)
        .agg(
            routes=("playId", "size"),
            games=("gameId", "nunique"),
            receiver_windows=("receiver_window", "sum"),
            credited_windows=("window_credit", "sum"),
            mean_best_separation=("best_separation", "mean"),
        )
        .reset_index()
    )
    receiver_scores["raw_chances"] = receiver_scores["receiver_windows"]
    receiver_scores["credited_chances"] = receiver_scores["credited_windows"]
    receiver_scores["raw_open_rate_per_100_routes"] = (
        100 * receiver_scores["receiver_windows"] / receiver_scores["routes"]
    )
    receiver_scores["chance_created_per_100_routes"] = (
        100 * receiver_scores["credited_windows"] / receiver_scores["routes"]
    )
    receiver_scores.sort_values(
        ["chance_created_per_100_routes", "routes"], ascending=[False, False]
    ).to_csv(output / "receiver_scores.csv", index=False)

    play_scores = (
        routes.groupby(["gameId", "playId", "possessionTeam", "defensiveTeam"])
        .agg(
            team_play_window=("team_play_window", "max"),
            actionable_chance=("actionable_chance", "max"),
            pressured=("pressured", "max"),
            chance_receivers=("receiver_window", "sum"),
        )
        .reset_index()
    )
    outcomes = plays.copy()
    outcomes["first_down_success"] = (
        outcomes["prePenaltyPlayResult"] >= outcomes["yardsToGo"]
    ).astype(int)
    outcomes["touchdown"] = outcomes["playDescription"].str.contains(
        "TOUCHDOWN", case=False, na=False
    ).astype(int)
    play_scores = play_scores.merge(
        outcomes[
            [
                "gameId",
                "playId",
                "first_down_success",
                "touchdown",
                "prePenaltyPlayResult",
            ]
        ],
        on=["gameId", "playId"],
        how="left",
    )
    team_games = (
        play_scores.groupby(["gameId", "possessionTeam", "defensiveTeam"])
        .agg(
            dropbacks=("playId", "size"),
            team_windows=("team_play_window", "sum"),
            actionable_chances=("actionable_chance", "sum"),
            pressured_dropbacks=("pressured", "sum"),
            receiver_opportunities=("chance_receivers", "sum"),
            successful_dropbacks=("first_down_success", "sum"),
            touchdowns=("touchdown", "sum"),
            total_passing_yards=("prePenaltyPlayResult", "sum"),
        )
        .reset_index()
    )
    team_games["team_chances"] = team_games["actionable_chances"]
    team_games["window_rate_per_100_dropbacks"] = (
        100 * team_games["team_windows"] / team_games["dropbacks"]
    )
    team_games["actionable_chance_rate_per_100_dropbacks"] = (
        100 * team_games["actionable_chances"] / team_games["dropbacks"]
    )
    team_games["chance_rate_per_100_dropbacks"] = (
        team_games["actionable_chance_rate_per_100_dropbacks"]
    )
    team_games["actionability_rate"] = (
        100 * team_games["actionable_chances"] / team_games["team_windows"].replace(0, np.nan)
    )
    team_games["first_down_rate_per_100_dropbacks"] = (
        100 * team_games["successful_dropbacks"] / team_games["dropbacks"]
    )
    team_games = team_games.merge(games[["gameId", "week"]], on="gameId", how="left")
    team_games.to_csv(output / "team_game_summary.csv", index=False)

    correlations = []
    for team, group in team_games.groupby("possessionTeam"):
        correlations.append(
            {
                "team": team,
                "games": len(group),
                "actionable_chance_vs_first_down_correlation": group[
                    "actionable_chance_rate_per_100_dropbacks"
                ].corr(group["first_down_rate_per_100_dropbacks"]),
                "actionable_chance_vs_touchdown_correlation": group[
                    "actionable_chance_rate_per_100_dropbacks"
                ].corr(group["touchdowns"]),
                "actionable_chance_vs_yards_correlation": group[
                    "actionable_chance_rate_per_100_dropbacks"
                ].corr(group["total_passing_yards"]),
            }
        )
    pd.DataFrame(correlations).to_csv(output / "team_correlations.csv", index=False)

    print(f"Wrote outputs to {output}")
    print(f"Scored {len(routes):,} routes across {routes[['gameId', 'playId']].drop_duplicates().shape[0]:,} plays")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--separation", type=float, default=3.5)
    parser.add_argument("--projected-separation", type=float, default=2.0)
    parser.add_argument("--projection-seconds", type=float, default=0.5)
    parser.add_argument("--required-frames", type=int, default=3)
    parser.add_argument("--fallback-after-snap", type=int, default=25)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "output"
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
