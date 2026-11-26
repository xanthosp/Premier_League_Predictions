"""Dixon-Coles team ratings with time decay, refit every week.

log E[home goals] = mu + home + attack[h] + defence[a];  log E[away goals] = mu + attack[a] + defence[h]
(defence > 0 means the team concedes more). Matches are weighted by exp(-xi * days ago) and only the last
WINDOW_DAYS are used. A small ridge penalty shrinks teams with few matches towards average. The low-score
correction rho is fitted afterwards on goals with the ratings fixed (it barely moves the ratings).

`target`: "goals" fits real scores; "npxg" fits each side's non-penalty xG instead (a less noisy measure of how
good the chances were); "mix" uses the average of the two. Teams without matches in the window (promoted) get
the mean rating of the three weakest rated teams.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .scorelines import outcome_probabilities, score_matrix


class DixonColes:
    def __init__(self, target: str = "goals", xi: float = 0.0019, window_days: int = 3 * 365, ridge: float = 1.0):
        self.target, self.xi, self.window_days, self.ridge = target, xi, window_days, ridge
        self.name = "dixon_coles" if target == "goals" else f"dixon_coles_{target}"
        self.params = {"target": target, "xi": xi, "window_days": window_days, "ridge": ridge}
        self._start, self.league = None, None

    def _targets(self, train: pd.DataFrame):
        goals = train[["fthg", "ftag"]].to_numpy(float)
        if self.target == "goals":
            return goals
        xg = train[["result_home_npxg", "result_away_npxg"]].to_numpy(float)
        xg = np.where(np.isnan(xg), goals, xg)  # the few matches without xG fall back to goals
        return xg if self.target == "npxg" else (xg + goals) / 2

    def fit(self, train: pd.DataFrame) -> None:
        if self.league is not None:
            train = train[train.league == self.league]
        dates = pd.to_datetime(train.date)
        recent = train[dates > dates.max() - pd.Timedelta(days=self.window_days)]
        age = (pd.to_datetime(recent.date).max() - pd.to_datetime(recent.date)).dt.days.to_numpy()
        w = np.exp(-self.xi * age)
        teams = sorted(set(recent.home_team) | set(recent.away_team))
        index = {t: i for i, t in enumerate(teams)}
        h, a = recent.home_team.map(index).to_numpy(), recent.away_team.map(index).to_numpy()
        y = self._targets(recent)
        n = len(teams)

        def unpack(theta):
            return theta[0], theta[1], theta[2:2 + n], theta[2 + n:]

        def loss(theta):
            mu, home, att, dfn = unpack(theta)
            lh, la = np.exp(mu + home + att[h] + dfn[a]), np.exp(mu + att[a] + dfn[h])
            nll = (w * (lh - y[:, 0] * np.log(lh) + la - y[:, 1] * np.log(la))).sum()
            nll += self.ridge * ((att ** 2).sum() + (dfn ** 2).sum())
            rh, ra = w * (lh - y[:, 0]), w * (la - y[:, 1])  # d nll / d log(lambda)
            grad = np.concatenate([[rh.sum() + ra.sum(), rh.sum()],
                                   np.bincount(h, rh, n) + np.bincount(a, ra, n) + 2 * self.ridge * att,
                                   np.bincount(a, rh, n) + np.bincount(h, ra, n) + 2 * self.ridge * dfn])
            return nll, grad

        start = np.zeros(2 + 2 * n)
        start[0] = np.log(max(y.mean(), 0.1))
        if self._start is not None:  # warm start from last week's ratings
            old_teams, old = self._start
            start[:2] = old[:2]
            for t, i in index.items():
                if t in old_teams:
                    j = old_teams[t]
                    start[2 + i], start[2 + n + i] = old[2 + j], old[2 + len(old_teams) + j]
        theta = minimize(loss, start, jac=True, method="L-BFGS-B").x
        self._start = (index, theta)
        self.mu, self.home, att, dfn = unpack(theta)
        self.attack, self.defence = dict(zip(teams, att)), dict(zip(teams, dfn))
        weakest = np.argsort(att - dfn)[:3]
        self.default = (att[weakest].mean(), dfn[weakest].mean())

        goals = recent[["fthg", "ftag"]].to_numpy(int)
        lh, la = self._rates(recent)
        cells = np.clip(goals, 0, 10)

        def rho_loss(rho):
            m = score_matrix(lh, la, rho)
            return -(w * np.log(np.clip(m[np.arange(len(m)), cells[:, 0], cells[:, 1]], 1e-12, 1))).sum()

        self.rho = minimize_scalar(rho_loss, bounds=(-0.2, 0.2), method="bounded").x

    def _rates(self, frame: pd.DataFrame):
        att_h = frame.home_team.map(lambda t: self.attack.get(t, self.default[0])).to_numpy()
        dfn_h = frame.home_team.map(lambda t: self.defence.get(t, self.default[1])).to_numpy()
        att_a = frame.away_team.map(lambda t: self.attack.get(t, self.default[0])).to_numpy()
        dfn_a = frame.away_team.map(lambda t: self.defence.get(t, self.default[1])).to_numpy()
        return np.exp(self.mu + self.home + att_h + dfn_a), np.exp(self.mu + att_a + dfn_h)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        lh, la = self._rates(test)
        return outcome_probabilities(lh, la, self.rho)
