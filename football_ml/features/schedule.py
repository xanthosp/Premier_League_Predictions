"""Fixture congestion from every competition (league, domestic cups, Europe), as of each match.

Per team and match date d:
- rest_days: days since the previous played match in any competition;
- matches_14d: played matches in [d - 14, d);
- prev_europe: the previous match was European;
- days_to_next / next_europe: the next scheduled match after d. Fixture dates are published weeks ahead,
  so they are known before kickoff (a later cup draw can add a match we did not know about yet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCHEDULE_FEATURES = ["rest_days", "matches_14d", "prev_europe", "days_to_next", "next_europe"]


def schedule_features(targets: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """targets: key columns + team, date. schedule: team, date, kind, played. Returns targets + SCHEDULE_FEATURES."""
    targets = targets.assign(_d=pd.to_datetime(targets.date), _row=np.arange(len(targets)))
    schedule = schedule.assign(_d=pd.to_datetime(schedule.date))
    played = schedule[schedule.played.astype(bool)].sort_values("_d")
    ordered = targets.sort_values("_d")
    prev = pd.merge_asof(ordered, played[["team", "_d", "kind"]].rename(columns={"_d": "_prev", "kind": "_pkind"}),
                         left_on="_d", right_on="_prev", by="team", direction="backward", allow_exact_matches=False)
    nxt = pd.merge_asof(ordered[["_row", "team", "_d"]],
                        schedule.sort_values("_d")[["team", "_d", "kind"]].rename(columns={"_d": "_next", "kind": "_nkind"}),
                        left_on="_d", right_on="_next", by="team", direction="forward", allow_exact_matches=False)
    out = prev.merge(nxt[["_row", "_next", "_nkind"]], on="_row").sort_values("_row")
    out["rest_days"] = (out._d - out._prev).dt.days
    out["prev_europe"] = out._pkind.eq("europe").where(out._prev.notna())
    out["days_to_next"] = (out._next - out._d).dt.days
    out["next_europe"] = out._nkind.eq("europe").where(out._next.notna())
    counts = np.zeros(len(out))
    played_days = {team: group._d.to_numpy() for team, group in played.groupby("team")}
    for i, (team, day) in enumerate(zip(out.team, out._d.to_numpy())):
        days = played_days.get(team)
        if days is not None:
            counts[i] = np.searchsorted(days, day, "left") - np.searchsorted(days, day - np.timedelta64(14, "D"), "left")
    out["matches_14d"] = counts
    return out.drop(columns=[c for c in out.columns if c.startswith("_")]).reset_index(drop=True)
