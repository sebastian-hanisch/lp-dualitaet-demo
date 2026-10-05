"""Orakel für das Ranging: die Bereichsgrenzen werden NICHT aus dem Endtableau, sondern über je ein eigenes LP (HiGHS) bestimmt.

Rechte Seite: größtes delta, für das noch eine zulässige Lösung mit c·x >= z* + y_i·delta existiert (y_i bleibt dann ein optimaler Schattenpreis).
Kosten: größtes delta, für das die Lösung x* noch optimal bleibt (Dual zulässig mit Komplementarität zu x*). Beide Intervalle enthalten den Basisbereich der Demo und sind
ohne Dual-Entartung gleich. Zusätzlich: Optimalwert, Schlupf und reduzierte Kosten gegen HiGHS auf Zufallsinstanzen mit kleinen ganzen Zahlen (stark entartet, viele Gleichstände)."""

import math
import random

import numpy as np
import pytest

import dua_algorithm as A
import dua_scenario as S
import dua_sensitivity as D

linprog = pytest.importorskip("scipy.optimize").linprog


def _highs(inst):
    Am, b, c = inst.arrays()
    ub_a, ub_b, eq_a, eq_b = [], [], [], []
    for i, s in enumerate(inst.senses):
        if s == S.LE:
            ub_a.append(Am[i]), ub_b.append(b[i])
        elif s == S.GE:
            ub_a.append(-Am[i]), ub_b.append(-b[i])
        else:
            eq_a.append(Am[i]), eq_b.append(b[i])
    return linprog(-c, A_ub=np.array(ub_a) if ub_a else None, b_ub=ub_b or None, A_eq=np.array(eq_a) if eq_a else None, b_eq=eq_b or None, bounds=(0, None), method="highs")


def _int_instance(rng):
    m, n = rng.randint(1, 4), rng.randint(1, 4)
    rows = [[rng.randint(-2, 3) for _ in range(n)] for _ in range(m)]
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(rng.randint(-3, 8)) for _ in range(m)), tuple(float(rng.randint(-2, 5)) for _ in range(n)),
                      tuple(rng.choice([S.LE, S.LE, S.GE, S.EQ]) for _ in range(m)), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(m)), "custom")


def _rhs_extreme(inst, sens, i, direction):
    """Äußerstes delta (direction = +1 / -1), für das c·x >= z* + y_i delta bei b_i + delta zulässig bleibt."""
    Am, b, c = inst.arrays()
    n = inst.n
    ub, rub, eq, req = [], [], [], []
    for k, s in enumerate(inst.senses):
        row = np.concatenate([Am[k], [-(1.0 if k == i else 0.0)]])
        if s == S.LE:
            ub.append(row), rub.append(b[k])
        elif s == S.GE:
            ub.append(-row), rub.append(-b[k])
        else:
            eq.append(row), req.append(b[k])
    ub.append(np.concatenate([-c, [sens.y[i]]])), rub.append(-sens.obj)
    obj = np.zeros(n + 1)
    obj[-1] = -direction
    r = linprog(obj, A_ub=np.array(ub), b_ub=rub, A_eq=np.array(eq) if eq else None, b_eq=req or None, bounds=[(0, None)] * n + [(None, None)], method="highs")
    return direction * math.inf if r.status == 3 else r.x[-1]


def _cost_extreme(inst, sens, j, direction):
    """Äußerstes delta, für das x* bei c_j + delta optimal bleibt: Dual y zulässig, y_i = 0 bei Schlupf, A^T y = c an den Stellen x_k > 0."""
    Am, _b, c = inst.arrays()
    m, n = Am.shape
    ub, rub, eq, req = [], [], [], []
    for k in range(n):
        row = np.concatenate([Am[:, k], [-(1.0 if k == j else 0.0)]])
        if sens.x[k] > 1e-9:
            eq.append(row), req.append(c[k])
        else:
            ub.append(-row), rub.append(-c[k])
    bounds = [(0, 0) if not sens.binding[i] else ((0, None) if s == S.LE else ((None, 0) if s == S.GE else (None, None))) for i, s in enumerate(inst.senses)] + [(None, None)]
    obj = np.zeros(m + 1)
    obj[-1] = -direction
    r = linprog(obj, A_ub=np.array(ub) if ub else None, b_ub=rub or None, A_eq=np.array(eq) if eq else None, b_eq=req or None, bounds=bounds, method="highs")
    return direction * math.inf if r.status == 3 else r.x[-1]


def _dual_degenerate(inst, sens):
    basis = set(sens.sol.basis)
    return any(j not in basis and abs(sens.reduced[j]) < 1e-7 for j in range(inst.n)) or any(sens.binding[i] and inst.senses[i] != S.EQ and abs(sens.y[i]) < 1e-7 for i in range(inst.m))


def _close(a, b):
    return a == b if (math.isinf(a) or math.isinf(b)) else abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def _instances():
    rng = random.Random(7)
    count = 0
    while count < 22:
        inst = _int_instance(rng)
        if A.solve(inst).status == "optimal":
            count += 1
            yield inst
    for seed in range(4):
        yield S.generate("random", 6, 8, 0.5, seed)
        yield S.generate("mixed", 6, 8, 0.5, seed)
    yield S.textbook_instance()
    yield S.centre_instance()


def test_ranges_are_inside_the_lp_oracle_and_equal_without_dual_degeneracy():
    equal = 0
    for inst in _instances():
        sens = D.analyse_lp(inst)
        h = _highs(inst)
        assert h.status == 0 and sens.obj == pytest.approx(-h.fun, rel=1e-7, abs=1e-7)
        Am, b, c = inst.arrays()
        x = np.array(sens.x)
        assert list(sens.reduced) == pytest.approx(list(Am.T @ np.array(sens.y) - c), abs=1e-7)
        dd = _dual_degenerate(inst, sens)
        for i in range(inst.m):
            lo, hi = sens.rhs_range[i]
            o_lo, o_hi = _rhs_extreme(inst, sens, i, -1), _rhs_extreme(inst, sens, i, 1)
            assert lo >= o_lo - 1e-6 * max(1.0, abs(o_lo)) and hi <= o_hi + 1e-6 * max(1.0, abs(o_hi)), (inst, i)
            if not sens.degenerate and not dd:
                assert _close(lo, o_lo) and _close(hi, o_hi), (inst, i, (lo, hi), (o_lo, o_hi))
                equal += 1
        for j in range(inst.n):
            lo, hi = sens.cost_range[j]
            o_lo, o_hi = inst.c[j] + _cost_extreme(inst, sens, j, -1), inst.c[j] + _cost_extreme(inst, sens, j, 1)
            assert lo >= o_lo - 1e-6 * max(1.0, abs(o_lo)) and hi <= o_hi + 1e-6 * max(1.0, abs(o_hi)), (inst, j)
            if not sens.degenerate and not dd:
                assert _close(lo, o_lo) and _close(hi, o_hi), (inst, j, (lo, hi), (o_lo, o_hi))
                equal += 1
        assert x.min() >= -1e-9
    assert equal > 100


def test_statuses_and_values_on_small_integer_instances_agree_with_highs():
    rng = random.Random(1)
    seen = {"optimal": 0, "infeasible": 0, "unbounded": 0}
    for _ in range(250):
        inst = _int_instance(rng)
        sol = A.solve(inst)
        h = _highs(inst)
        if h.status == 0:
            ref = "optimal"
        else:
            Am, b, _c = inst.arrays()
            zero = S.Instance(inst.A, inst.b, tuple(0.0 for _ in inst.c), inst.senses, inst.names, inst.row_names, "custom")
            ref = "infeasible" if _highs(zero).status == 2 else "unbounded"
        assert sol.status == ref, inst
        seen[ref] += 1
        if ref == "optimal":
            assert sol.obj == pytest.approx(-h.fun, rel=1e-7, abs=1e-7)
            assert A.primal_violation(inst, sol.x) < 1e-7 and A.dual_violation(inst, sol.y) < 1e-6 and float(np.dot(sol.y, inst.b)) == pytest.approx(sol.obj, abs=1e-6)
    assert all(v > 20 for v in seen.values())
