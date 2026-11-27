"""Decision models and a fixed menu of betting strategies, compared by CLV (never chosen by ROI).

Building blocks (all fitted on earlier seasons only):
- stacking: log E[goals] = c0 + c1 log(open market rate) + c2 log(model rate), a Poisson regression per side.
  Unlike the convex mix of policy.py, c1 + c2 is free, so the model's compression of strong-vs-weak gaps
  (it saw Liverpool only 0.9 goals better than Burnley, the market 1.9) cannot leak into fake longshot value.
- uncertainty: block bootstrap of the stacking (resampling whole weeks) -> P10/P90 of each bet's expected profit.
- movement: ridge regression of (close - open) supremacy and total goals on pre-match information, i.e. where the
  price is expected to go by kickoff. The expected CLV of a bet is its expected profit under that predicted close.

Every probability comes from the market's exact scoreline matrix tilted by Poisson(new rates) / Poisson(market),
so with no model information the fair market is reproduced and a margin is never mistaken for value.

STRATEGIES is the menu we choose from: inputs set by us, a priori, each run for both the average price ("Avg") and
the best price among bookmakers ("Max", only reachable with several accounts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.scorelines import score_matrix
from . import policy
from .markets import SELECTIONS, expected_profit, market_matrix

B_BOOTSTRAP = 100
MOVE_ALPHA = 50.0
MOVE_FEATURES = ["open_sup", "open_tot", "gap_sup", "gap_tot", "elo_diff", "npxg_for", "npxg_against", "shots_for",
                 "shots_against", "rest_days", "matches_14d", "europe", "missing_xg", "load_8d"]

# name: (decision source, minimum expected profit, require P10 > 0, maximum price, require predicted CLV > 0,
#        scale the minimum with the price risk). Scaling is a soft constraint instead of excluding longshots:
#        the required edge grows with sqrt(price - 1), i.e. with the spread of the bet's result (3% at evens, ~9.5% at 11.0).
STRATEGIES = {
    "blend_ev3": ("blend", 0.03, False, None, False, False),        # Phase 4 baseline
    "blend_ev3_scaled": ("blend", 0.03, False, None, False, True),
    "stack_ev3": ("stack", 0.03, False, None, False, False),
    "stack_p10": ("stack", 0.03, True, None, False, False),
    "stack_p10_odds4": ("stack", 0.03, True, 4.0, False, False),
    "move_clv1": ("move", 0.01, False, None, False, False),          # bet where the close is expected to beat our price
    "stack_p10_move": ("stack", 0.03, True, None, True, False),
    "stack_p10_move_scaled": ("stack", 0.03, True, None, True, True),
}


# ---------- stacking ----------

def irls_poisson(x: np.ndarray, y: np.ndarray, iterations: int = 30) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(max(y.mean(), 0.1))
    for _ in range(iterations):
        mu = np.exp(np.clip(x @ beta, -10, 5))
        z = x @ beta + (y - mu) / mu
        info = x.T @ (x * mu[:, None])
        new = np.linalg.solve(info, x.T @ (mu * z))
        if np.abs(new - beta).max() < 1e-8:
            beta = new
            break
        beta = new
    mu = np.exp(x @ beta)
    return beta, np.linalg.inv(x.T @ (x * mu[:, None]))


def _stack_design(frame: pd.DataFrame, side: str) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), np.log(frame[f"market_l{side}"]), np.log(frame[f"model_l{side}"])])


def fit_stack(frame: pd.DataFrame) -> dict:
    return {side: irls_poisson(_stack_design(frame, side), frame[goals].to_numpy(float))[0]
            for side, goals in (("h", "fthg"), ("a", "ftag"))}


def stack_rates(frame: pd.DataFrame, coef: dict) -> tuple[np.ndarray, np.ndarray]:
    return tuple(np.exp(_stack_design(frame, side) @ coef[side]) for side in ("h", "a"))


def bootstrap_stack(frame: pd.DataFrame, b: int = B_BOOTSTRAP, seed: int = 0) -> list[dict]:
    """Resample whole weeks (matches of a week share shocks), refit the stacking each time."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(frame.date)
    weeks = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.strftime("%Y%m%d").to_numpy()
    unique = np.unique(weeks)
    groups = {w: np.flatnonzero(weeks == w) for w in unique}
    fits = []
    for _ in range(b):
        idx = np.concatenate([groups[w] for w in rng.choice(unique, len(unique))])
        fits.append(fit_stack(frame.iloc[idx]))
    return fits


# ---------- movement ----------

def move_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["open_sup"], out["open_tot"] = frame.market_lh - frame.market_la, frame.market_lh + frame.market_la
    out["gap_sup"] = (frame.model_lh - frame.model_la) - out.open_sup
    out["gap_tot"] = (frame.model_lh + frame.model_la) - out.open_tot
    for c in MOVE_FEATURES[4:]:
        out[c] = frame[c]
    return out


def move_targets(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"d_sup": (frame.close_lh - frame.close_la) - (frame.market_lh - frame.market_la),
                         "d_tot": (frame.close_lh + frame.close_la) - (frame.market_lh + frame.market_la)})


class MoveModel:
    """Ridge regression on standardized features (missing values -> training mean)."""

    def fit(self, frame: pd.DataFrame) -> "MoveModel":
        x, y = move_features(frame), move_targets(frame)
        ok = y.notna().all(axis=1) & frame.get("move_ok", pd.Series(True, index=frame.index)).astype(bool)
        x, y = x[ok], y[ok]
        if len(x) < 100:  # no reliable history of price moves: predict no move
            self.mean, self.std = x.mean(), x.std()
            self.coef, self.y_mean = np.zeros((x.shape[1], 2)), np.zeros(2)
            return self
        self.mean, self.std = x.mean(), x.std().replace(0, 1)
        z = ((x - self.mean) / self.std).fillna(0).to_numpy()
        self.y_mean = y.mean().to_numpy()
        a = z.T @ z + MOVE_ALPHA * np.eye(z.shape[1])
        self.coef = np.linalg.solve(a, z.T @ (y.to_numpy() - self.y_mean))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        z = ((move_features(frame) - self.mean) / self.std).fillna(0).to_numpy()
        return z @ self.coef + self.y_mean  # (n, 2): predicted change of supremacy and total

    def to_dict(self) -> dict:
        return {"features": MOVE_FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "coef": self.coef.tolist(), "y_mean": list(self.y_mean)}

    @classmethod
    def from_dict(cls, d: dict) -> "MoveModel":
        if d["features"] != MOVE_FEATURES:
            raise ValueError("policy.json was fitted with other movement features: rerun `paper --fit-policy`")
        model = cls()
        model.mean, model.std = pd.Series(d["mean"], index=MOVE_FEATURES), pd.Series(d["std"], index=MOVE_FEATURES)
        model.coef, model.y_mean = np.array(d["coef"]), np.array(d["y_mean"])
        return model


def move_matrix(frame: pd.DataFrame, move: MoveModel, base: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scoreline matrix of the predicted closing market: open supremacy/total shifted by the predicted move."""
    d = move.predict(frame)
    sup, tot = (frame.market_lh - frame.market_la) + d[:, 0], (frame.market_lh + frame.market_la) + d[:, 1]
    lh, la = np.clip((tot + sup) / 2, 0.05, None), np.clip((tot - sup) / 2, 0.05, None)
    return tilted(frame, lh, la, base), d


def price_card(frame: pd.DataFrame, matrix: np.ndarray, min_ev: float) -> pd.DataFrame:
    """For every selection: the fair price under `matrix` and the minimum price that gives an expected profit of
    `min_ev` (expected profit is linear in the price, also for Asian lines with pushes and half results)."""
    out = []
    for market, selections in SELECTIONS.items():
        line = frame.ah_line.to_numpy(float) if market == "ah" and "ah_line" in frame else np.full(len(frame), np.nan)
        if market == "ah" and np.isnan(line).all():
            continue
        for s in selections:
            ep2, ep3 = (expected_profit(matrix, market, s, line, np.full(len(frame), p)) for p in (2.0, 3.0))
            slope, intercept = ep3 - ep2, ep2 - 2 * (ep3 - ep2)
            with np.errstate(divide="ignore", invalid="ignore"):
                fair, minimum = -intercept / slope, (min_ev - intercept) / slope
            price = frame[f"{market}_{s}"].to_numpy(float) if f"{market}_{s}" in frame else np.full(len(frame), np.nan)
            out.append(pd.DataFrame({"match_id": frame.match_id.to_numpy(), "market": market, "selection": s, "line": line,
                                     "fair_price": fair, "min_price": minimum, "our_price": price}))
    return pd.concat(out, ignore_index=True).dropna(subset=["fair_price"])


# ---------- matrices and candidate bets ----------

def tilted(frame: pd.DataFrame, lh, la, base: np.ndarray | None = None) -> np.ndarray:
    market = (frame.market_lh.to_numpy(float), frame.market_la.to_numpy(float))
    base = market_matrix(frame) if base is None else base
    m = base * score_matrix(np.asarray(lh, float), np.asarray(la, float)) / score_matrix(*market)
    return m / m.sum(axis=(1, 2), keepdims=True)


def candidates(frame: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    """The best selection of every market of every match under `matrix`, at the frame's prices."""
    out = []
    for market, selections in SELECTIONS.items():
        if f"{market}_{selections[0]}" not in frame:
            continue
        line = frame.ah_line.to_numpy(float) if market == "ah" else np.full(len(frame), np.nan)
        prices = np.column_stack([frame[f"{market}_{s}"].to_numpy(float) for s in selections])
        evs = np.column_stack([expected_profit(matrix, market, s, line, prices[:, k]) for k, s in enumerate(selections)])
        evs = np.where(np.isfinite(evs), evs, -np.inf)
        best = evs.argmax(axis=1)
        rows = np.arange(len(frame))
        ok = np.isfinite(evs[rows, best])
        out.append(pd.DataFrame({"row": rows[ok], "match_id": frame.match_id.to_numpy()[ok], "market": market,
                                 "selection": np.array(selections)[best[ok]], "line": line[ok],
                                 "price": prices[rows, best][ok], "ev": evs[rows, best][ok]}))
    return pd.concat(out, ignore_index=True)


def all_selections(frame: pd.DataFrame) -> pd.DataFrame:
    """Every priced selection of every match (the 'back everything' baseline)."""
    out = []
    for market, selections in SELECTIONS.items():
        line = frame.ah_line.to_numpy(float) if market == "ah" else np.full(len(frame), np.nan)
        for s in selections:
            price = frame[f"{market}_{s}"].to_numpy(float)
            ok = np.isfinite(price)
            out.append(pd.DataFrame({"row": np.flatnonzero(ok), "season": frame.season.to_numpy()[ok], "market": market,
                                     "selection": s, "line": line[ok], "price": price[ok]}))
    return pd.concat(out, ignore_index=True)


def ev_under(matrix: np.ndarray, bets: pd.DataFrame) -> np.ndarray:
    """Expected profit of each candidate row under the matrix row of its match."""
    sub = matrix[bets.row.to_numpy()]
    return policy.evaluate_bets(sub, bets.reset_index(drop=True))


def run_strategies(frame: pd.DataFrame, close_matrix: np.ndarray, price_frames: dict[str, pd.DataFrame],
                   menu: dict | None = None) -> pd.DataFrame:
    """Season-by-season walk-forward: stacking, bootstrap and movement fitted on earlier seasons only.
    Returns every bet of every strategy with flat/Kelly profit and realised CLV."""
    seasons = sorted(frame.season.unique())
    bets = []
    for season in seasons[1:]:
        past, now = frame.season < season, (frame.season == season).to_numpy()
        coef = fit_stack(frame[past])
        boots = bootstrap_stack(frame[past])
        w = policy.fit_weight((frame[past].model_lh, frame[past].model_la), (frame[past].market_lh, frame[past].market_la),
                              frame[past].fthg, frame[past].ftag)
        move = MoveModel().fit(frame[past])
        cur = frame[now].reset_index(drop=True)
        base = market_matrix(cur)
        matrices = {"stack": tilted(cur, *stack_rates(cur, coef), base),
                    "blend": policy.decision_matrix(cur, w)}
        matrices["move"] = move_matrix(cur, move, base)[0]
        boot_m = [tilted(cur, *stack_rates(cur, c), base) for c in boots]
        cm = close_matrix[now]
        for price_name, prices in price_frames.items():
            pcur = prices[now].reset_index(drop=True)
            for name, (source, min_ev, need_p10, max_price, need_move, scaled) in (menu or STRATEGIES).items():
                cand = candidates(pcur, matrices[source])
                cand = cand[cand.ev >= min_ev * (np.sqrt(np.maximum(cand.price - 1, 1e-9)) if scaled else 1)]
                if max_price is not None:
                    cand = cand[cand.price <= max_price]
                if cand.empty:
                    continue
                cand = cand.reset_index(drop=True)
                draws = np.column_stack([ev_under(m, cand) for m in boot_m])
                cand["p10"], cand["p90"] = np.percentile(draws, 10, axis=1), np.percentile(draws, 90, axis=1)
                cand["pred_clv"] = ev_under(matrices["move"], cand)
                if need_p10:
                    cand = cand[cand.p10 > 0]
                if need_move:
                    cand = cand[cand.pred_clv > 0]
                if cand.empty:
                    continue
                cand = cand.reset_index(drop=True)
                hg, ag = cur.fthg.to_numpy()[cand.row], cur.ftag.to_numpy()[cand.row]
                cand["stake"] = policy.STAKE
                edge = np.clip(np.where(need_p10, cand.p10, cand.ev), 0, None)
                cand["kelly_stake"] = np.minimum(policy.KELLY_FRACTION * edge / (cand.price - 1) * policy.BANKROLL,
                                                 policy.KELLY_CAP).round(2)
                settled = policy.settle(cand, hg, ag, cm[cand.row.to_numpy()])
                bets.append(cand.join(settled).assign(strategy=name, prices=price_name, season=season))
    return pd.concat(bets, ignore_index=True)


def summarize(bets: pd.DataFrame) -> pd.DataFrame:
    g = bets.groupby(["strategy", "prices"])
    out = pd.DataFrame({"bets": g.size(), "clv": g.clv.mean(), "clv_se": g.clv.std() / np.sqrt(g.size()),
                        "clv_positive": g.clv.apply(lambda c: (c > 0).mean()), "roi": g.profit.sum() / g.stake.sum(),
                        "kelly_profit": g.kelly_profit.sum(), "mean_ev": g.ev.mean(),
                        "skill": (bets.clv - bets.clv_base).groupby([bets.strategy, bets.prices]).mean()})
    per_season = bets.pivot_table(index=["strategy", "prices"], columns="season", values="clv", aggfunc="mean")
    out = out.join(per_season.add_prefix("clv_"))
    return out.round(4).sort_values("clv", ascending=False).reset_index()
