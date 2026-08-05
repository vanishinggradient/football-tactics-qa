"""Collect match data from StatsBomb open data and convert to text narratives."""

import hashlib
from statsbombpy import sb


# Free, open competitions with good tactical variety
COMPETITIONS = [
    (43, 106, "FIFA World Cup 2022"),
    (55, 43, "UEFA Euro 2020"),
    (11, 90, "La Liga 2020/21"),
    (2, 44, "Premier League 2003/04"),
    (49, 3, "NWSL 2018"),
]


def _match_narrative(match_row, events_df, lineups):
    """Turn structured match + event data into a readable text narrative."""
    home = match_row["home_team"]
    away = match_row["away_team"]
    date = match_row["match_date"]
    score_home = match_row["home_score"]
    score_away = match_row["away_score"]
    competition = match_row.get("competition", "")
    stadium = match_row.get("stadium", "Unknown")
    referee = match_row.get("referee", "Unknown")

    # Extract key events
    goals = events_df[events_df["type"] == "Shot"]
    goals = goals[goals["shot_outcome"] == "Goal"] if "shot_outcome" in goals.columns else goals.head(0)

    passes = events_df[events_df["type"] == "Pass"]
    pass_count = len(passes)
    pass_complete = len(passes[passes.get("pass_outcome", "").isna()]) if "pass_outcome" in passes.columns else 0

    cards = events_df[events_df["type"].isin(["Bad Behaviour", "Foul Won"])]

    # Build narrative
    parts = []
    parts.append(
        f"Match Report: {home} vs {away} ({competition})"
    )
    parts.append(
        f"Played on {date} at {stadium}. Referee: {referee}."
    )
    parts.append(f"Final score: {home} {score_home} - {score_away} {away}.")

    # Goal details
    if len(goals) > 0:
        goal_lines = []
        for _, g in goals.iterrows():
            player = g.get("player", "Unknown")
            minute = g.get("minute", "?")
            team = g.get("team", "")
            goal_lines.append(f"  - {player} ({team}) at minute {minute}")
        parts.append("Goals:\n" + "\n".join(goal_lines))

    # Passing stats
    if pass_count > 0:
        home_passes = len(passes[passes["team"] == home])
        away_passes = len(passes[passes["team"] == away])
        parts.append(
            f"Passing: {home} attempted {home_passes} passes, "
            f"{away} attempted {away_passes} passes."
        )

    # Lineup info
    for team_name, team_lineup in lineups.items():
        if len(team_lineup) > 0:
            players = team_lineup["player_name"].tolist()[:11]
            parts.append(f"{team_name} lineup: {', '.join(players)}.")

    return "\n\n".join(parts)


def collect_statsbomb_data():
    """Download StatsBomb open data and create text narratives for each match."""
    documents = []

    for comp_id, season_id, comp_name in COMPETITIONS:
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception as e:
            print(f"Skipping {comp_name}: {e}")
            continue

        print(f"Processing {comp_name}: {len(matches)} matches")

        for _, match_row in matches.iterrows():
            match_id = match_row["match_id"]
            try:
                events_df = sb.events(match_id=match_id, flatten_attrs=True)
                lineups = sb.lineups(match_id=match_id)
            except Exception as e:
                print(f"  Skipping match {match_id}: {e}")
                continue

            narrative = _match_narrative(match_row, events_df, lineups)
            doc_id = hashlib.md5(f"statsbomb-{match_id}".encode()).hexdigest()[:12]

            home = match_row["home_team"]
            away = match_row["away_team"]

            documents.append({
                "doc_id": doc_id,
                "source": "statsbomb",
                "title": f"{home} vs {away} - {comp_name}",
                "content": narrative,
                "metadata": {
                    "match_id": int(match_id),
                    "competition": comp_name,
                    "date": str(match_row["match_date"]),
                    "home_team": home,
                    "away_team": away,
                },
            })

        print(f"  Collected {len(documents)} match narratives so far")

    return documents


if __name__ == "__main__":
    docs = collect_statsbomb_data()
    print(f"Total StatsBomb documents: {len(docs)}")
    if docs:
        print(f"Sample:\n{docs[0]['content'][:500]}")
