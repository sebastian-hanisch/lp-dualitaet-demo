"""Korrektheitskette: Zertifikate (starke Dualität, Zulässigkeit, komplementärer Schlupf), Dual-LP, Schattenpreis = Ableitung, Ranging-Grenzen gegen Neulösung, Entartung, Zukauf, Sonderfälle."""

import math

import numpy as np
import pytest

import dua_algorithm as A
import dua_scenario as S
import dua_sensitivity as D
from tests.test_scenario import _highs, reference_status


def _custom(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


def _families(seeds=60):
    """Instanzen aller Arten: Zufall, Mischung (>= und =), Transport, Zentrum, Lehrbuch, entartete Ecke."""
    for seed in range(seeds):
        yield S.generate("random", 6, 8, 0.5, seed)
        yield S.generate("mixed", 6, 8, 0.5, seed)
        yield S.generate("mixed", 8, 5, 0.5, seed)
        yield S.generate("transport", 2, 4, 0.5, seed)
        yield S.generate("transport", 3, 3, 0.5, seed)
    yield S.textbook_instance()
    yield S.centre_instance()
    yield S.degenerate_instance()


def _marginals_to_duals(inst, h):
    """HiGHS-Marginals als Schattenpreise in der Orientierung dz/db_i (Maximierung): <=: -m; >=: +m (Zeile wurde gespiegelt); =: -m."""
    y, ub, eq = [], 0, 0
    for s in inst.senses:
        if s == S.LE:
            y.append(-h.ineqlin.marginals[ub])
            ub += 1
        elif s == S.GE:
            y.append(h.ineqlin.marginals[ub])
            ub += 1
        else:
            y.append(-h.eqlin.marginals[eq])
            eq += 1
    return y


def test_certificates_on_every_family_instance():
    count, degenerate = 0, 0
    for inst in _families():
        sens, h = D.analyse_lp(inst), _highs(inst)
        assert sens is not None and h.status == 0, (inst.kind, inst.m, inst.n)
        assert sens.obj == pytest.approx(-h.fun, rel=1e-7, abs=1e-7)
        assert A.primal_violation(inst, sens.x) < 1e-7 and A.dual_violation(inst, sens.y) < 1e-6
        assert sens.dual_obj == pytest.approx(sens.obj, rel=1e-7, abs=1e-7)                           # starke Dualität c·x = b·y
        assert max(abs(y * s) for y, s in zip(sens.y, sens.slack)) < 1e-7                              # komplementärer Schlupf (Ressourcen)
        assert max(abs(x * r) for x, r in zip(sens.x, sens.reduced)) < 1e-7 and min(sens.reduced) > -1e-7  # komplementärer Schlupf (Dienste)
        count += 1
        degenerate += sens.degenerate
        if not sens.degenerate:
            assert list(sens.y) == pytest.approx(_marginals_to_duals(inst, h), abs=1e-7)              # eindeutige Duale == HiGHS
    assert count >= 300 and degenerate > 20


def test_dual_lp_gives_the_same_value_with_our_solver_and_with_highs():
    for inst in list(_families(20)):
        dual, decode = D.dual_instance(inst)
        primal = D.analyse_lp(inst)
        ours = D.solve_dual(inst)
        assert ours is not None and ours[0] == pytest.approx(primal.obj, rel=1e-7, abs=1e-7) and A.dual_violation(inst, ours[1]) < 1e-6
        h = _highs(dual)
        assert h.status == 0 and -h.fun == pytest.approx(-primal.obj, rel=1e-7, abs=1e-7)              # dual: max -b·y


def test_textbook_by_hand():
    s = D.analyse_lp(S.textbook_instance())
    assert s.y == pytest.approx((0.0, 1.5, 1.0)) and s.slack == pytest.approx((2.0, 0.0, 0.0)) and s.binding == (False, True, True) and s.reduced == pytest.approx((0.0, 0.0))
    assert s.x == pytest.approx((2.0, 6.0)) and s.obj == pytest.approx(36.0) and not s.degenerate
    assert s.rhs_range[0][0] == pytest.approx(-2.0) and math.isinf(s.rhs_range[0][1]) and s.rhs_range[1] == pytest.approx((-6.0, 6.0)) and s.rhs_range[2] == pytest.approx((-6.0, 6.0))
    assert s.cost_range[0] == pytest.approx((0.0, 7.5)) and s.cost_range[1][0] == pytest.approx(2.0) and math.isinf(s.cost_range[1][1])


def test_centre_by_construction():
    s = D.analyse_lp(S.centre_instance())
    assert s.y == pytest.approx((5.0, 0.5, 4.0, 0.0)) and s.slack == pytest.approx((0, 0, 0, 160.0)) and s.x == pytest.approx((20, 10, 10, 0, 0))
    assert s.reduced == pytest.approx((0, 0, 0, 3.5, 3.5)) and s.rhs_range[0] == pytest.approx((-10.0, 10.0)) and s.rhs_range[1] == pytest.approx((-28.0, 17.5))
    assert s.rhs_range[3][0] == pytest.approx(-160.0) and math.isinf(s.rhs_range[3][1]) and s.cost_range[3] == pytest.approx((-math.inf, 23.5)) and s.cost_range[4][1] == pytest.approx(15.5)


# --- Schattenpreis = Ableitung ------------------------------------------------------------------------------------------------------------------------


def _highs_opt(inst):
    h = _highs(inst)
    return -h.fun if h.status == 0 else float("nan")


def test_shadow_price_is_the_derivative_inside_the_range_and_not_outside():
    checked, outside = 0, 0
    for inst in _families(25):
        s = D.analyse_lp(inst)
        if s.degenerate:
            continue
        for i in range(inst.m):
            lo, hi = s.rhs_range[i]
            for side, limit in ((-1, lo), (1, hi)):
                if math.isinf(limit) or abs(limit) < 1e-9:
                    continue
                d = 0.5 * limit
                for opt in (D.optimum, _highs_opt):
                    assert (opt(D.with_rhs(inst, i, d)) - s.obj) / d == pytest.approx(s.y[i], abs=1e-6)
                far = 1.5 * limit
                z = D.optimum(D.with_rhs(inst, i, far))
                if not math.isnan(z):                                                                   # jenseits der Grenze ist die Steigung eine andere (konkav: kleiner beim Erhöhen)
                    assert abs((z - s.obj) / far - s.y[i]) > 1e-6
                    outside += 1
                checked += 1
    assert checked > 150 and outside > 50


def test_range_limits_against_the_resolve_just_inside_and_outside():
    inside, changed = 0, 0
    for inst in _families(25):
        s = D.analyse_lp(inst)
        if s.degenerate:
            continue
        for i in range(inst.m):
            for limit in s.rhs_range[i]:
                if math.isinf(limit) or abs(limit) < 1e-9:
                    continue
                near = D.analyse_lp(D.with_rhs(inst, i, limit * (1 - 1e-6)))
                assert near is not None and near.sol.basic_set == s.sol.basic_set                      # innerhalb: dieselbe Basis
                assert near.y == pytest.approx(s.y, abs=1e-6)
                beyond = D.analyse_lp(D.with_rhs(inst, i, limit * (1 + 1e-3)))
                if beyond is not None:
                    assert beyond.sol.basic_set != s.sol.basic_set                                     # außerhalb: andere Basis
                    changed += 1
                inside += 1
    assert inside > 100 and changed > 40


def test_value_function_is_concave_in_b_and_convex_in_c():
    for inst in (S.centre_instance(), S.textbook_instance(), S.generate("random", 5, 6, 0.6, 3), S.generate("mixed", 6, 6, 0.6, 5)):
        s = D.analyse_lp(inst)
        for i in range(inst.m):
            if inst.senses[i] != S.LE:
                continue
            base = max(1.0, abs(inst.b[i]))
            deltas = np.linspace(-0.6 * base, 1.5 * base, 43)
            z = np.array(D.value_curve(inst, i, deltas))
            ok = ~np.isnan(z)
            z, dd = z[ok], deltas[ok]
            assert len(z) > 10
            slopes = np.diff(z) / np.diff(dd)
            assert (np.diff(slopes) <= 1e-7).all() and (np.diff(z) >= -1e-9).all()                     # konkav, nicht fallend (<=-Zeile)
        for j in range(inst.n):
            grid = np.linspace(0.3 * inst.c[j], 2.5 * inst.c[j] + 1, 41)
            zc = np.array([D.optimum(D.with_cost(inst, j, v)) for v in grid])
            assert (np.diff(np.diff(zc)) >= -1e-7).all() and s.obj == pytest.approx(D.optimum(inst))


def test_cost_ranges_keep_the_solution_inside_and_change_it_outside():
    inside, outside = 0, 0
    for inst in _families(20):
        s = D.analyse_lp(inst)
        if s.degenerate or not all(min(v) > -1e-6 for v in [s.reduced]):
            continue
        for j in range(inst.n):
            lo, hi = s.cost_range[j]
            for limit, sgn in ((lo, -1), (hi, 1)):
                if math.isinf(limit):
                    continue
                gap = 1e-4 * max(1.0, abs(limit))
                near = A.solve(D.with_cost(inst, j, limit - sgn * gap))
                assert near.status == "optimal" and near.x == pytest.approx(s.x, abs=1e-6)
                far = A.solve(D.with_cost(inst, j, limit + sgn * gap))
                if far.status == "optimal":
                    assert not np.allclose(far.x, s.x, atol=1e-6)
                    outside += 1
                inside += 1
    assert inside > 100 and outside > 60


def test_nonbasic_variable_enters_exactly_above_its_cost_limit():
    inst = S.centre_instance()
    s = D.analyse_lp(inst)
    for j in (3, 4):
        limit = s.cost_range[j][1]
        assert A.solve(D.with_cost(inst, j, limit - 1e-3)).x[j] == pytest.approx(0.0, abs=1e-9)
        assert A.solve(D.with_cost(inst, j, limit + 1e-3)).x[j] > 1e-6


# --- Vorzeichen, Sonderfälle ---------------------------------------------------------------------------------------------------------------------------


def test_signs_of_the_duals_for_less_equal_greater_equal_and_equality_rows():
    seen = set()
    for inst in _families(30):
        s = D.analyse_lp(inst)
        for y, sense in zip(s.y, inst.senses):
            if sense == S.LE:
                assert y >= -1e-9
            elif sense == S.GE:
                assert y <= 1e-9
            seen.add((sense, y > 1e-9, y < -1e-9))
    assert {(S.GE, False, True), (S.EQ, True, False), (S.EQ, False, True), (S.LE, True, False)} <= seen


def test_mixed_rows_range_and_derivative_match_the_resolve():
    checked = 0
    for inst in _families(30):
        if S.GE not in inst.senses and S.EQ not in inst.senses:
            continue
        s = D.analyse_lp(inst)
        if s.degenerate:
            continue
        for i, sense in enumerate(inst.senses):
            if sense == S.LE:
                continue
            lo, hi = s.rhs_range[i]
            for limit in (lo, hi):
                if math.isinf(limit) or abs(limit) < 1e-9:
                    continue
                d = 0.5 * limit
                assert (D.optimum(D.with_rhs(inst, i, d)) - s.obj) / d == pytest.approx(s.y[i], abs=1e-6)
                checked += 1
    assert checked > 30


def test_redundant_equality_has_a_zero_range_and_unbounded_or_infeasible_have_no_analysis():
    inst = _custom([[1, 1, 0], [1, 1, 0], [0, 1, 1]], [4, 4, 6], [1, 2, 1], [S.EQ, S.EQ, S.LE])
    s = D.analyse_lp(inst)
    assert s is not None and s.obj == pytest.approx(-_highs(inst).fun) and (s.rhs_range[0] == (0.0, 0.0) or s.rhs_range[1] == (0.0, 0.0))
    assert D.analyse_lp(S.infeasible_instance()) is None and D.analyse_lp(S.unbounded_instance()) is None
    assert A.solve(S.infeasible_instance()).status == "infeasible" and A.solve(S.unbounded_instance()).status == "unbounded"
    assert reference_status(S.infeasible_instance()) == "infeasible"


def test_single_row_and_non_binding_row_and_determinism():
    one = _custom([[2, 3]], [12], [4, 5], [S.LE])
    s = D.analyse_lp(one)
    assert s.y == pytest.approx((2.0,)) and s.x == pytest.approx((6.0, 0.0)) and s.rhs_range[0][0] == pytest.approx(-12.0) and math.isinf(s.rhs_range[0][1])
    c = D.analyse_lp(S.centre_instance())
    assert c.y[3] == 0.0 and c.binding == (True, True, True, False)
    assert D.analyse_lp(S.centre_instance()).y == c.y and D.analyse_lp(S.centre_instance()).rhs_range == c.rhs_range


# --- Entartung ---------------------------------------------------------------------------------------------------------------------------------------


def test_degenerate_vertex_has_different_left_and_right_shadow_prices():
    inst = S.degenerate_instance()
    s = D.analyse_lp(inst)
    assert s.degenerate and not D.dual_unique(inst) and s.y == pytest.approx((0.0, 1.5, 1.0, 0.0))
    slopes = [D.one_sided_slopes(inst, i) for i in range(4)]
    assert slopes[1] == pytest.approx((1.5, 1.0), abs=1e-6) and slopes[2] == pytest.approx((1.0, 0.0), abs=1e-6) and slopes[3] == pytest.approx((3.0, 0.0), abs=1e-6) and slopes[0] == pytest.approx((0.0, 0.0), abs=1e-6)
    for i in (1, 2, 3):
        left, right = slopes[i]
        assert right - 1e-6 <= s.y[i] <= left + 1e-6                                                     # der gewählte Duale liegt zwischen rechts- und linksseitiger Steigung
    for i in (1, 2, 3):
        h = 1e-4
        assert (_highs_opt(D.with_rhs(inst, i, h)) - s.obj) / h == pytest.approx(slopes[i][1], abs=1e-3)   # rechtsseitig gegen HiGHS
        assert (s.obj - _highs_opt(D.with_rhs(inst, i, -h))) / h == pytest.approx(slopes[i][0], abs=1e-3)  # linksseitig gegen HiGHS
    assert s.rhs_range[1][1] == pytest.approx(0.0) and s.rhs_range[2][1] == pytest.approx(0.0)


def test_non_degenerate_bases_always_have_unique_duals_and_transport_never_does():
    unique_checked, transports, degenerate_transports, nonunique_transports = 0, 0, 0, 0
    for inst in _families(30):
        s = D.analyse_lp(inst)
        if not s.degenerate:
            assert D.dual_unique(inst)
            unique_checked += 1
        if inst.kind == "transport":
            transports += 1
            degenerate_transports += s.degenerate
            nonunique_transports += not D.dual_unique(inst)
    assert unique_checked > 60 and degenerate_transports == transports == nonunique_transports


def _highs_face(inst, i, sign):
    """Kleinster (sign = 1) bzw. größter (sign = -1) Schattenpreis y_i über die optimale Dual-Seitenfläche, mit HiGHS."""
    from scipy.optimize import linprog
    A_, b, c = inst.arrays()
    z = D.optimum(inst)
    bounds = [(0, None) if s == S.LE else ((None, 0) if s == S.GE else (None, None)) for s in inst.senses]
    e = np.zeros(inst.m)
    e[i] = sign
    r = linprog(e, A_ub=-A_.T, b_ub=-c, A_eq=b.reshape(1, -1), b_eq=[z], bounds=bounds, method="highs")
    return sign * r.fun if r.status == 0 else (-sign * math.inf if r.status == 3 else float("nan"))


def test_dual_face_ranges_match_highs_and_the_one_sided_slopes():
    checked = 0
    insts = [S.degenerate_instance(), S.textbook_instance(), S.centre_instance()] + [S.generate("transport", 2, 3, 0.5, s) for s in range(6)] + [S.generate("mixed", 6, 6, 0.5, s) for s in range(12)]
    for inst in insts:
        face = D.dual_face_ranges(inst)
        for i in range(inst.m):
            lo, hi = face[i]
            assert lo == pytest.approx(_highs_face(inst, i, 1), abs=1e-6) and hi == pytest.approx(_highs_face(inst, i, -1), abs=1e-6)
            checked += 1
    assert checked > 100
    d = S.degenerate_instance()
    face = D.dual_face_ranges(d)
    for i in (1, 2, 3):
        left, right = D.one_sided_slopes(d, i)
        assert face[i] == pytest.approx((right, left), abs=1e-6)                                         # Spanne = rechts- bis linksseitige Steigung
    assert face[1] == pytest.approx((1.0, 1.5)) and face[2] == pytest.approx((0.0, 1.0)) and face[3] == pytest.approx((0.0, 3.0)) and face[0] == pytest.approx((0.0, 0.0))
    assert [v for r in D.dual_face_ranges(S.textbook_instance()) for v in r] == pytest.approx([0.0, 0.0, 1.5, 1.5, 1.0, 1.0])


def test_degenerate_instances_with_unique_duals_exist_and_non_unique_only_if_degenerate():
    nonunique = 0
    for inst in _families(30):
        s = D.analyse_lp(inst)
        if not D.dual_unique(inst):
            assert s.degenerate
            nonunique += 1
    assert nonunique > 5


# --- Zukauf ------------------------------------------------------------------------------------------------------------------------------------------


def test_purchase_rule_in_steps_equals_the_exact_lp():
    inst = S.centre_instance()
    z0 = D.optimum(inst)
    for i, prices in ((0, (1.0, 2.0, 3.0, 4.5)), (1, (0.1, 0.3)), (2, (1.0, 3.0))):
        for price in prices:
            exact = D.purchase_exact(inst, i, price)
            rule = D.purchase_steps(inst, i, price)
            assert exact is not None and not rule["unbounded"]
            assert rule["value"] == pytest.approx(exact[0], rel=1e-5, abs=1e-3) and rule["extra"] == pytest.approx(exact[1], rel=1e-4, abs=1e-3) and exact[0] >= z0 - 1e-9
    assert len(D.purchase_steps(inst, 0, 2.0)["steps"]) == 3 and [round(y, 3) for _d, y in D.purchase_steps(inst, 0, 2.0)["steps"]] == [5.0, 4.125, 2.833]


def test_no_purchase_at_or_above_the_shadow_price_and_never_for_a_free_resource():
    inst = S.centre_instance()
    for i, price in ((0, 5.0), (0, 7.0), (3, 0.5), (1, 0.5)):
        exact, rule = D.purchase_exact(inst, i, price), D.purchase_steps(inst, i, price)
        assert exact[1] == pytest.approx(0.0, abs=1e-9) and exact[0] == pytest.approx(D.optimum(inst)) and rule["steps"] == [] and rule["extra"] == 0.0


def test_unbounded_purchase_is_detected():
    inst = _custom([[1]], [10], [1], [S.LE])
    assert D.purchase_exact(inst, 0, 0.5) is None and D.purchase_steps(inst, 0, 0.5)["unbounded"]
    with pytest.raises(ValueError):
        D.with_purchase(_custom([[1]], [10], [1], [S.GE]), 0, 1.0)


def test_purchase_marginal_value_falls_with_every_step():
    steps = D.purchase_steps(S.centre_instance(), 0, 0.5)["steps"]
    ys = [y for _d, y in steps]
    assert len(steps) >= 3 and all(a > b for a, b in zip(ys, ys[1:])) and ys[-1] > 0.5
