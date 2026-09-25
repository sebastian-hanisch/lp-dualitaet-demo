"""Auswertung: Analyse, Wertfunktion, Vorhersage gegen Neulösung, Zukauf, Bereichsbreiten, Basiskipp unter Kostenrauschen, Entartung."""

import math
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import dua_algorithm as A
import dua_constants as C
import dua_scenario as S
import dua_sensitivity as D


@dataclass(frozen=True)
class Settings:
    kind: str = "centre"
    m: int = C.DEFAULT_M                      # Ressourcen (Transport: Lager)
    n: int = C.DEFAULT_N                      # Dienste (Transport: Kunden)
    seed: int = C.DEFAULT_SEED
    resource: int = 0
    price: float = C.DEFAULT_PRICE
    var: int = 0

    @property
    def res_i(self):
        return min(self.resource, instance_of(self).m - 1)

    @property
    def var_j(self):
        return min(self.var, instance_of(self).n - 1)


@lru_cache(maxsize=256)
def instance_of(settings):
    return S.generate(settings.kind, settings.m, settings.n, C.DENSITY, settings.seed)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    sens: object                  # None: unzulässig oder unbeschränkt
    status: str


@lru_cache(maxsize=128)
def analyse(settings):
    inst = instance_of(settings)
    sol = A.solve(inst)
    return Analysis(settings, inst, D.analyse_lp(inst) if sol.status == "optimal" else None, sol.status)


@lru_cache(maxsize=128)
def faces(settings):
    """Kleinster und größter Schattenpreis je Ressource über alle optimalen Dualen."""
    return D.dual_face_ranges(instance_of(settings))


@lru_cache(maxsize=128)
def pieces(settings, hi_frac=1.5):
    """Wertfunktion der gewählten Ressource als Knickpunkte im Fenster [b - 0.6 s, b + hi_frac s] mit s = max(1, |b|)."""
    inst = instance_of(settings)
    i = settings.res_i
    b = inst.b[i]
    s = max(1.0, abs(b))
    return D.value_pieces(inst, i, b - 0.6 * s, b + hi_frac * s)


def prediction_row(inst, sens, i, delta):
    """Vorhersage y_i·delta gegen den tatsächlichen Gewinn durch Neulösung (nan: unzulässig)."""
    z = D.optimum(D.with_rhs(inst, i, delta))
    actual = z - sens.obj if not math.isnan(z) else float("nan")
    return {"delta": delta, "predicted": sens.y[i] * delta, "actual": actual, "in_range": sens.rhs_range[i][0] - 1e-9 <= delta <= sens.rhs_range[i][1] + 1e-9}


@lru_cache(maxsize=128)
def prediction_table(settings):
    """Erweiterung um 0.5 / 1 / 2 / 4 mal die obere Bereichsgrenze (ist sie unendlich: mal |b_i|); jeweils Vorhersage und Neulösung."""
    a = analyse(settings)
    if a.sens is None:
        return []
    i = settings.res_i
    hi = a.sens.rhs_range[i][1]
    unit = hi if (not math.isinf(hi) and hi > 1e-9) else max(1.0, abs(a.inst.b[i]))
    return [dict(factor=f, **prediction_row(a.inst, a.sens, i, f * unit)) for f in C.PREDICTION_FACTORS]


@lru_cache(maxsize=128)
def purchase(settings):
    """Zukauf von Kapazität der gewählten Ressource zum Preis `price`: exaktes LP, Schattenpreis-Regel in Schritten und die naive Vorhersage (y_i - Preis) mal gekaufte Menge. None bei nicht-<=-Zeile."""
    a = analyse(settings)
    if a.sens is None:
        return None
    i = settings.res_i
    if a.inst.senses[i] != S.LE:
        return None
    exact = D.purchase_exact(a.inst, i, settings.price)
    rule = D.purchase_steps(a.inst, i, settings.price)
    if exact is None:
        return {"unbounded": True, "steps": rule["steps"], "y": a.sens.y[i]}
    value, extra = exact
    return {"unbounded": False, "value": value, "extra": extra, "gain": value - a.sens.obj, "naive_gain": (a.sens.y[i] - settings.price) * extra if a.sens.y[i] > settings.price else 0.0, "steps": rule["steps"],
            "y": a.sens.y[i], "rule_value": rule["value"], "price": settings.price}


def _seed_instances(settings, seeds=C.SWEEP_SEEDS):
    if settings.kind in S.FIXTURE_KINDS:
        return [instance_of(settings)]
    return [S.generate(settings.kind, settings.m, settings.n, C.DENSITY, s) for s in seeds]


@lru_cache(maxsize=128)
def range_widths(settings):
    """Median über die festen Instanzen (Seeds 100000-100004; Fixtures: die Instanz): Erhöhung und Senkung der rechten Seite bindender Ressourcen bis zur Grenze (Anteil von b_i, nur endliche Grenzen), Breite des
    Kostenbereichs der Basis-Dienste (Anteil von c_j)."""
    up, down, width = [], [], []
    for inst in _seed_instances(settings):
        sens = D.analyse_lp(inst)
        if sens is None:
            continue
        for i in range(inst.m):
            if sens.binding[i] and inst.b[i] > 0:
                lo, hi = sens.rhs_range[i]
                if not math.isinf(hi):
                    up.append(hi / inst.b[i])
                if not math.isinf(lo):
                    down.append(-lo / inst.b[i])
        for j in range(inst.n):
            if sens.x[j] > 1e-9 and inst.c[j] > 0:
                lo, hi = sens.cost_range[j]
                if not (math.isinf(lo) or math.isinf(hi)):
                    width.append((hi - lo) / inst.c[j])
    med = lambda v: float(np.median(v)) if v else float("nan")
    return {"up": med(up), "down": med(down), "cost_width": med(width), "n_up": len(up), "n_down": len(down), "n_cost": len(width)}


@lru_cache(maxsize=128)
def flip_rate(settings, sigma):
    """Anteil der 50 Läufe (Seeds 200000-200049), bei denen multiplikatives Kostenrauschen der Stärke sigma die optimale Basis ändert (Zufallsarten: je Seed eine Instanz; Fixtures: 50 Rauschziehungen)."""
    flips = tot = 0
    for s in C.NOISE_SEEDS:
        inst = instance_of(settings) if settings.kind in S.FIXTURE_KINDS else S.generate(settings.kind, settings.m, settings.n, C.DENSITY, s)
        base = A.solve(inst)
        if base.status != "optimal":
            continue
        rng = random.Random(f"dua-noise-{s}")
        noisy = S.Instance(inst.A, inst.b, tuple(v * (1 + sigma * rng.gauss(0.0, 1.0)) for v in inst.c), inst.senses, inst.names, inst.row_names, inst.kind)
        new = A.solve(noisy)
        tot += 1
        flips += new.status != "optimal" or new.basic_set != base.basic_set
    return flips / tot if tot else float("nan")


@lru_cache(maxsize=64)
def degeneracy_share(kind, m, n):
    """Anteil der 50 Instanzen (Seeds 200000-200049) mit entarteter Endbasis und mit nicht eindeutigen Dualen; Fixtures: die eine Instanz."""
    deg = nonuni = tot = 0
    for s in C.NOISE_SEEDS if kind not in S.FIXTURE_KINDS else (0,):
        inst = S.generate(kind, m, n, C.DENSITY, s)
        sens = D.analyse_lp(inst)
        if sens is None:
            continue
        tot += 1
        deg += sens.degenerate
        nonuni += not D.dual_unique(inst)
    return {"n": tot, "degenerate": deg / tot if tot else float("nan"), "non_unique": nonuni / tot if tot else float("nan")}


def check_limit(settings, side):
    """Neulösung knapp innerhalb (99.9 % der Grenze) und knapp außerhalb (100.1 %) der oberen (side = 1) bzw. unteren (side = -1) Bereichsgrenze der gewählten Ressource: Basis unverändert? neuer Wert?"""
    a = analyse(settings)
    if a.sens is None:
        return None
    i = settings.res_i
    limit = a.sens.rhs_range[i][1] if side == 1 else a.sens.rhs_range[i][0]
    if math.isinf(limit) or abs(limit) < 1e-9:
        return {"limit": limit, "rows": []}
    rows = []
    for label, factor in (("knapp innerhalb (99.9 %)", 0.999), ("knapp außerhalb (100.1 %)", 1.001)):
        new = D.analyse_lp(D.with_rhs(a.inst, i, factor * limit))
        rows.append({"label": label, "delta": factor * limit, "status": "unzulässig" if new is None else "optimal", "same_basis": None if new is None else new.sol.basic_set == a.sens.sol.basic_set,
                     "y": None if new is None else new.y[i], "obj": None if new is None else new.obj})
    return {"limit": limit, "rows": rows}


def check_cost_limit(settings, side):
    """Neulösung knapp innerhalb/außerhalb der oberen (side = 1) bzw. unteren (side = -1) Kostengrenze des gewählten Dienstes: Lösung unverändert?"""
    a = analyse(settings)
    if a.sens is None:
        return None
    j = settings.var_j
    limit = a.sens.cost_range[j][1] if side == 1 else a.sens.cost_range[j][0]
    if math.isinf(limit):
        return {"limit": limit, "rows": []}
    rows = []
    gap = 1e-3 * max(1.0, abs(limit))
    for label, c in (("knapp innerhalb", limit - side * gap), ("knapp außerhalb", limit + side * gap)):
        new = A.solve(D.with_cost(a.inst, j, c))
        rows.append({"label": label, "c": c, "same_solution": new.status == "optimal" and bool(np.allclose(new.x, a.sens.x, atol=1e-6)), "x_j": new.x[j] if new.status == "optimal" else float("nan"),
                     "obj": new.obj})
    return {"limit": limit, "rows": rows}
