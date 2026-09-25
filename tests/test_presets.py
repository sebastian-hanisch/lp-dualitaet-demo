"""Presets: gültige Werte und jede Zahl im Hilfetext gegen die echten Auswertungsfunktionen."""

import math

import pytest

import dua_constants as C
import dua_evaluation as ev
import dua_scenario as S
import dua_sensitivity as D
from dua_presets import PRESET_KEYS, SETTING_SPECS


def _settings(name):
    p = C.PRESETS[name]
    m, n = (p["k"], p["l"]) if p["kind"] == "transport" else (p["m"], p["n"])
    return ev.Settings(p["kind"], m, n, p["seed"], p["res"], p["price"], p["var"])


def _has(name, *values):
    for v in values:
        assert v in C.PRESET_HELP[name], (name, v)


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 10
    for name, p in C.PRESETS.items():
        assert set(p) <= set(PRESET_KEYS) and {"kind", "step"} <= set(p), name
        for key, state_key in PRESET_KEYS.items():
            if key in p and state_key in SETTING_SPECS:
                spec = SETTING_SPECS[state_key]
                assert spec.caster(p[key]) == p[key], (name, key)
                if spec.lo is not None:
                    assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip()
        if "delta" in p:
            assert p["step"] == 2 and p["delta"] in C.DELTA_OPTIONS and p["delta"] != 0


def test_help_textbook_and_centre_and_mixed():
    s = ev.analyse(_settings("Lehrbuch: Duale von Hand")).sens
    assert s.y == pytest.approx((0, 1.5, 1)) and s.obj == pytest.approx(36) and s.dual_obj == pytest.approx(12 * 1.5 + 18 * 1) and s.slack[0] == pytest.approx(2.0)
    _has("Lehrbuch: Duale von Hand", "Optimum 36", "(2, 6)", "y = (0, 1.5, 1)", "Schlupf 2", "12·1.5 + 18·1")
    c = ev.analyse(_settings("Zentrum: drei Preise, eine freie Ressource")).sens
    assert c.y == pytest.approx((5, 0.5, 4, 0)) and c.obj == pytest.approx(720) and c.slack[3] == pytest.approx(160) and c.reduced[3:] == pytest.approx((3.5, 3.5))
    _has("Zentrum: drei Preise, eine freie Ressource", "Optimum 720", "(20, 10, 10, 0, 0)", "y = (5, 0.5, 4, 0)", "Schlupf 160", "3.5")
    a = ev.analyse(_settings("Mischung: Vorzeichen der Dualen"))
    m = a.sens
    assert sum(m.binding) == 4 and [round(y, 2) for y in m.y[:4]] == [2.73, -1.29, 2.36, 0.37] and a.inst.senses[1] == S.GE and m.obj == pytest.approx(320.67, abs=0.005)
    _has("Mischung: Vorzeichen der Dualen", "8 Ressourcen und 8 Diensten", "2.73, 2.36 und 0.37", "−1.29", "320.67", "4 bindende")


def test_help_prediction_purchase_and_ranging():
    s = _settings("Schattenpreis gilt nur im Bereich")
    a = ev.analyse(s)
    lo, hi = a.sens.rhs_range[1]
    assert (round(a.inst.b[1] + lo, 2), round(a.inst.b[1] + hi, 2)) == (172.0, 217.5) and hi == pytest.approx(17.5)
    row = ev.prediction_row(a.inst, a.sens, 1, 0.75 * a.inst.b[1])
    assert (round(row["predicted"], 2), round(row["actual"], 2)) == (75.0, 8.75) and not row["in_range"] and C.PRESETS["Schattenpreis gilt nur im Bereich"]["delta"] == 75
    _has("Schattenpreis gilt nur im Bereich", "y = 0.5", "172 und 217.5", "+75 %", "Δ = 150", "75 voraus", "8.75", "17.5")
    p = ev.purchase(_settings("Zukauf: Kommissionierstunden"))
    assert [round(y, 3) for _d, y in p["steps"]] == [5.0, 4.125, 2.833] and (round(p["extra"], 6), round(p["gain"], 6), round(p["value"], 6), round(p["naive_gain"], 6)) == (135.0, 160.0, 880.0, 405.0)
    _has("Zukauf: Kommissionierstunden", "Preis 2", "3 Schritten 135 Einheiten", "5, 4.125, 2.833", "160", "880", "405")
    r = ev.analyse(_settings("Ranging der Rampenzeit")).sens.rhs_range[2]
    assert (round(r[0], 2), round(r[1], 2)) == (-2.69, 3.89)
    up, down = ev.check_limit(_settings("Ranging der Rampenzeit"), 1), ev.check_limit(_settings("Ranging der Rampenzeit"), -1)
    assert round(up["rows"][1]["y"], 2) == 0.0 and round(down["rows"][1]["y"], 2) == 7.25
    _has("Ranging der Rampenzeit", "y = 4", "52.31 bis 58.89", "−2.69 bis +3.89", "auf 0", "7.25")
    c = ev.check_cost_limit(_settings("Sperrgut lohnt ab 23.5"), 1)
    assert c["limit"] == pytest.approx(23.5) and round(c["rows"][1]["c"], 2) == 23.52 and round(c["rows"][1]["x_j"], 2) == 7.78 and round(c["rows"][1]["obj"], 2) == 720.18
    _has("Sperrgut lohnt ab 23.5", "c = 20", "reduzierte Kosten 3.5", "c = 23.5", "23.52", "7.78", "720.18")


def test_help_noise_degeneracy_and_transport():
    s = _settings("Kostenrauschen kippt die Basis")
    assert [ev.flip_rate(s, x) for x in C.NOISE_SIGMAS] == [0.04, 0.26, 0.4]
    assert [ev.flip_rate(ev.Settings("centre"), x) for x in C.NOISE_SIGMAS] == [0.0, 0.5, 0.82]
    _has("Kostenrauschen kippt die Basis", "10 × 10", "1 %, 5 % und 10 %", "4 %, 26 % und 40 %", "0 %, 50 %, 82 %")
    d = _settings("Entartete Ecke: links ungleich rechts")
    face = ev.faces(d)
    assert ev.analyse(d).sens.y == pytest.approx((0, 1.5, 1, 0)) and face[1] == pytest.approx((1.0, 1.5)) and face[2] == pytest.approx((0.0, 1.0)) and face[3] == pytest.approx((0.0, 3.0))
    left, right = D.one_sided_slopes(ev.instance_of(d), 1)
    assert (round(left, 2), round(right, 2)) == (1.5, 1.0)
    _has("Entartete Ecke: links ungleich rechts", "(2, 6)", "y = (0, 1.5, 1, 0)", "y₂ ∈ [1, 1.5]", "y₃ ∈ [0, 1]", "y₄ ∈ [0, 3]", "1.5 und 1.0")
    t = _settings("Transport: Duale nicht eindeutig")
    assert ev.analyse(t).sens.degenerate and any(math.isinf(v) for pair in ev.faces(t) for v in pair)
    shares = [ev.degeneracy_share("transport", 3, 5), ev.degeneracy_share("transport", 4, 8), ev.degeneracy_share("random", 10, 10), ev.degeneracy_share("mixed", 10, 10)]
    assert [x["non_unique"] for x in shares] == [1.0, 1.0, 0.0, 0.04]
    _has("Transport: Duale nicht eindeutig", "3 Lagern und 5 Kunden", "unbeschränkt", "je 100 %", "0 %", "4 %")
