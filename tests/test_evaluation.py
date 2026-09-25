"""Auswertung: Analyse, Wertfunktion, Vorhersage gegen Neulösung, Zukauf, Bereichsbreiten, Basiskipp, Entartungsanteile, Grenzprüfungen."""

import math

import pytest

import dua_constants as C
import dua_evaluation as ev
import dua_scenario as S
import dua_sensitivity as D
from dua_evaluation import Settings


def test_settings_clamp_the_resource_and_variable_to_the_instance():
    s = Settings("textbook", resource=9, var=9)
    assert (s.res_i, s.var_j) == (2, 1) and Settings("centre", resource=3, var=4).res_i == 3


def test_analyse_statuses_and_caching():
    a = ev.analyse(Settings("centre"))
    assert a.status == "optimal" and a.sens.obj == pytest.approx(720.0) and ev.analyse(Settings("centre")) is a
    for kind, status in (("infeasible", "infeasible"), ("unbounded", "unbounded")):
        b = ev.analyse(Settings(kind))
        assert b.status == status and b.sens is None
    assert ev.pieces(Settings("infeasible")) == []
    assert ev.prediction_table(Settings("infeasible")) == [] and ev.purchase(Settings("unbounded")) is None


def test_pieces_are_the_exact_concave_value_function():
    pts = ev.pieces(Settings("textbook", resource=1))
    zs = dict((round(b, 3), round(z, 3)) for b, z in pts)
    assert zs[6.0] == 27.0 and zs[12.0] == 36.0 and zs[18.0] == 45.0
    slopes = [(z2 - z1) / (b2 - b1) for (b1, z1), (b2, z2) in zip(pts, pts[1:])]
    assert all(a >= b - 1e-7 for a, b in zip(slopes, slopes[1:])) and slopes[0] == pytest.approx(2.5) and slopes[1] == pytest.approx(1.5)
    inst = S.textbook_instance()
    for b, z in pts:
        assert z == pytest.approx(D.optimum(D.with_rhs(inst, 1, b - 12.0)), abs=1e-4)


def test_prediction_table_is_exact_inside_the_range_and_overestimates_outside():
    rows = ev.prediction_table(Settings("centre", resource=1))
    assert [r["factor"] for r in rows] == list(C.PREDICTION_FACTORS)
    assert [r["in_range"] for r in rows] == [True, True, False, False]
    assert rows[0]["predicted"] == pytest.approx(rows[0]["actual"]) and rows[1]["predicted"] == pytest.approx(rows[1]["actual"])
    assert rows[2]["predicted"] > rows[2]["actual"] + 1 and rows[3]["actual"] == pytest.approx(8.75)
    free = ev.prediction_table(Settings("centre", resource=3))                       # Fahrzeugkapazität: Schattenpreis 0, Bereich nach oben unendlich
    assert all(r["predicted"] == 0.0 and r["actual"] == pytest.approx(0.0, abs=1e-9) for r in free)


def test_purchase_summary_and_special_cases():
    p = ev.purchase(Settings("centre", resource=0, price=2.0))
    assert p["extra"] == pytest.approx(135.0) and p["value"] == pytest.approx(880.0) and p["gain"] == pytest.approx(160.0) and p["naive_gain"] == pytest.approx(405.0) and len(p["steps"]) == 3
    assert p["rule_value"] == pytest.approx(p["value"], rel=1e-5)
    first = ev.purchase(Settings("centre", resource=0, price=4.5))
    assert first["steps"] and first["extra"] == pytest.approx(10.0)                    # Preis 4.5 unter y = 5: nur der erste Bereich
    no_buy = ev.purchase(Settings("centre", resource=1, price=1.0))
    assert no_buy["steps"] == [] and no_buy["extra"] == pytest.approx(0.0, abs=1e-9)
    seed = next(s for s in range(60) if S.GE in S.generate("mixed", 8, 8, C.DENSITY, s).senses)
    ge = S.generate("mixed", 8, 8, C.DENSITY, seed).senses.index(S.GE)
    assert ev.purchase(Settings("mixed", 8, 8, seed, ge)) is None


def test_range_widths_medians():
    w = ev.range_widths(Settings("centre"))
    assert w["up"] == pytest.approx(0.0875) and w["down"] == pytest.approx(0.125) and w["n_up"] == 3 and w["n_cost"] == 3 and w["cost_width"] == pytest.approx(0.0953, abs=1e-3)
    t = ev.range_widths(Settings("textbook"))
    assert t["up"] == pytest.approx(0.4167, abs=1e-3) and t["cost_width"] == pytest.approx(2.5)
    r = ev.range_widths(Settings("random", 10, 10))
    assert r["n_up"] > 5 and 0 < r["up"] < 1


def test_flip_rate_is_deterministic_monotone_and_zero_for_tiny_noise_in_the_textbook():
    s = Settings("random", 10, 10)
    rates = [ev.flip_rate(s, x) for x in C.NOISE_SIGMAS]
    assert rates == [ev.flip_rate(s, x) for x in C.NOISE_SIGMAS] and rates[0] <= rates[1] <= rates[2] and rates[2] > 0.2
    assert [ev.flip_rate(Settings("textbook"), x) for x in C.NOISE_SIGMAS] == [0.0, 0.0, 0.0]
    assert ev.flip_rate(Settings("centre"), 0.05) > 0.3


def test_degeneracy_share_by_kind():
    assert ev.degeneracy_share("transport", 3, 5) == {"n": 50, "degenerate": 1.0, "non_unique": 1.0}
    z = ev.degeneracy_share("random", 10, 10)
    assert z["degenerate"] == 0.0 and z["non_unique"] == 0.0
    assert ev.degeneracy_share("degenerate", 2, 2) == {"n": 1, "degenerate": 1.0, "non_unique": 1.0}


def test_limit_checks_by_hand_and_for_infinite_sides():
    s = Settings("centre", resource=2)
    up = ev.check_limit(s, 1)
    assert up["limit"] == pytest.approx(3.8889, abs=1e-4) and [r["same_basis"] for r in up["rows"]] == [True, False] and up["rows"][1]["y"] == pytest.approx(0.0, abs=1e-9)
    down = ev.check_limit(s, -1)
    assert down["rows"][1]["y"] == pytest.approx(7.25) and down["rows"][0]["y"] == pytest.approx(4.0)
    assert ev.check_limit(Settings("centre", resource=3), 1)["rows"] == []                     # Fahrzeugkapazität: keine obere Grenze
    c = ev.check_cost_limit(Settings("centre", var=3), 1)
    assert c["limit"] == pytest.approx(23.5) and [r["same_solution"] for r in c["rows"]] == [True, False] and c["rows"][1]["x_j"] == pytest.approx(7.7778, abs=1e-3)
    assert ev.check_cost_limit(Settings("centre", var=3), -1)["rows"] == []
    assert ev.check_limit(Settings("infeasible"), 1) is None and ev.check_cost_limit(Settings("unbounded"), 1) is None


def test_faces_are_cached_and_match_the_sensitivity_module():
    s = Settings("degenerate")
    assert ev.faces(s) is ev.faces(s) and ev.faces(s) == D.dual_face_ranges(ev.instance_of(s)) and not math.isnan(ev.faces(s)[1][0])
