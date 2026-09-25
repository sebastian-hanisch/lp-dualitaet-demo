"""Jede Zahl aus README und App-Texten gegen die echten Auswertungsfunktionen (dieselben, die die App aufruft)."""

from pathlib import Path

import pytest

import dua_constants as C
import dua_evaluation as ev
import dua_scenario as S
import dua_sensitivity as D
from dua_evaluation import Settings
from tests.test_algorithm import _families

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
APP_SRC = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")


def _has(*values):
    for v in values:
        assert v in README, v


def test_certificate_instance_count():
    assert len(list(_families())) == 303
    _has("Auf 303 Instanzen")


def test_textbook_ranges_by_hand():
    s = ev.analyse(Settings("textbook")).sens
    assert (s.inst.b[1] + s.rhs_range[1][0], s.inst.b[1] + s.rhs_range[1][1]) == pytest.approx((6.0, 18.0)) and (s.inst.b[2] + s.rhs_range[2][0], s.inst.b[2] + s.rhs_range[2][1]) == pytest.approx((12.0, 24.0))
    assert s.inst.b[0] + s.rhs_range[0][0] == pytest.approx(2.0) and s.cost_range[0] == pytest.approx((0.0, 7.5)) and s.cost_range[1][0] == pytest.approx(2.0)
    _has("Kommissionierstunden b ∈ [6, 18], Lagerfläche b ∈ [12, 24], Rampenzeit b ≥ 2", "c₁ ∈ [0, 7.5], c₂ ∈ [2, ∞)", "12·1.5 + 18·1 = 36")


def test_centre_ranges():
    s = ev.analyse(Settings("centre")).sens
    b = s.inst.b
    assert [(round(b[i] + s.rhs_range[i][0], 2), round(b[i] + s.rhs_range[i][1], 2)) for i in range(3)] == [(70.0, 90.0), (172.0, 217.5), (52.31, 58.89)]
    assert [round(v, 2) for v in s.cost_range[0]] == [14.8, 16.28] and [round(v, 2) for v in s.cost_range[1]] == [13.25, 19.67] and s.cost_range[2] == pytest.approx((24.9231, 26.875), abs=1e-3)
    _has("Kommissionierstunden [70, 90], Lagerfläche [172, 217.5], Rampenzeit [52.31, 58.89]", "Express [14.8, 16.28], Palettenversand [13.25, 19.67], Kühlware [24.92, 26.88]")


def test_prediction_tables():
    rows = ev.prediction_table(Settings("centre", resource=1))
    assert [r["predicted"] for r in rows] == pytest.approx([4.375, 8.75, 17.5, 35.0], abs=1e-6) and [r["actual"] for r in rows] == pytest.approx([4.375, 8.75, 8.75, 8.75], abs=1e-6)
    kom = ev.prediction_table(Settings("centre", resource=0))
    assert [r["predicted"] for r in kom] == pytest.approx([25.0, 50.0, 100.0, 200.0], abs=1e-6) and [r["actual"] for r in kom] == pytest.approx([25.0, 50.0, 91.2, 160.8], abs=0.05)
    _has("Vorhersage 4.4 / 8.8 / 17.5 / 35, Neulösung 4.4 / 8.8 / **8.75 / 8.75**", "25 / 50 / 100 / 200 gegen 25 / 50 / 91.2 / 160.8", "bei dem Vierfachen der Grenze verspricht y·Δ 35, die Neulösung findet 8.75", "exakt bis 17.5 Einheiten")


def test_purchase_numbers():
    two = ev.purchase(Settings("centre", resource=0, price=2.0))
    assert (round(two["extra"], 3), round(two["gain"], 3), round(two["value"], 3), round(two["naive_gain"], 3), len(two["steps"])) == (135.0, 160.0, 880.0, 405.0, 3)
    three = ev.purchase(Settings("centre", resource=0, price=3.0))
    assert (round(three["extra"], 3), round(three["gain"], 2), round(three["naive_gain"], 2), len(three["steps"])) == (30.0, 42.5, 60.0, 2)
    _has("in **3 Schritten 135 Einheiten** (Schattenpreis 5 → 4.125 → 2.833)", "**160**, Optimalwert 880", "**405**", "Zum Preis 3 kauft die Regel 30 Einheiten (2 Schritte) und gewinnt 42.5 statt der naiv erwarteten 60",
         "405, tatsächlich sind es 160", "(880)")
    assert "405 statt 160" in APP_SRC


def test_range_widths_by_kind():
    def w(kind, m=0, n=0):
        r = ev.range_widths(Settings(kind, m, n))
        return tuple(100 * r[k] for k in ("up", "down", "cost_width"))
    for got, want in ((w("centre"), (8.8, 12.5, 9.5)), (w("textbook"), (41.7, 41.7, 250.0)), (w("random", 6, 8), (50.3, 42.0, 68.3)), (w("random", 10, 10), (16.0, 37.3, 76.2)), (w("mixed", 8, 8), (47.1, 48.1, 162.7))):
        assert got == pytest.approx(want, abs=0.06)
    assert ev.range_widths(Settings("transport", 3, 5))["up"] == 0.0
    _has("Zentrum **8.8 % / 12.5 %** und 9.5 %", "Lehrbuch 41.7 % / 41.7 % und 250 %", "Zufall 6 × 8 50 % / 42 % und 68 %", "Zufall 10 × 10 16 % / 37 % und 76 %", "Mischung 8 × 8 47 % / 48 % und 163 %", "Erhöhung 0 % (entartet)")
    _has("im Median +8.8 % und −12.5 % des Bestands")
    assert "im Median +8.8 % und −12.5 % des Bestands" in APP_SRC


def test_flip_rates_by_kind():
    def rates(kind, m=0, n=0):
        return [ev.flip_rate(Settings(kind, m, n), x) for x in C.NOISE_SIGMAS]
    assert rates("centre") == [0.0, 0.5, 0.82] and rates("random", 10, 10) == [0.04, 0.26, 0.4] and rates("random", 6, 8) == [0.04, 0.14, 0.22] and rates("mixed", 8, 8) == [0.06, 0.16, 0.22]
    assert rates("transport", 3, 5) == [0.14, 0.52, 0.76] and rates("textbook") == [0.0, 0.0, 0.0]
    _has("Zentrum **0 % / 50 % / 82 %**", "Zufall 10 × 10 **4 % / 26 % / 40 %**", "Zufall 6 × 8 4 % / 14 % / 22 %", "Mischung 8 × 8 6 % / 16 % / 22 %", "Transport 3 × 5 14 % / 52 % / 76 %", "Lehrbuch 0 % / 0 % / 0 %",
         "im Zentrum in 50 % und bei Zufall 10 × 10 in 26 % der Läufe")
    assert "in 50 % und bei Zufall 10 × 10 in 26 % der Läufe" in APP_SRC


def test_degeneracy_shares_and_the_dual_face():
    sh = {(k, m, n): ev.degeneracy_share(k, m, n) for k, m, n in (("random", 6, 8), ("random", 10, 10), ("mixed", 8, 8), ("mixed", 10, 10), ("transport", 3, 5), ("transport", 4, 8))}
    assert [(v["degenerate"], v["non_unique"]) for v in sh.values()] == [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.04, 0.04), (1.0, 1.0), (1.0, 1.0)]
    inst = S.degenerate_instance()
    left, right = D.one_sided_slopes(inst, 1)
    assert (round(left, 2), round(right, 2)) == (1.5, 1.0)
    _has("Zufall 6 × 8 und 10 × 10 **0 % / 0 %**", "Mischung 8 × 8 0 %", "Mischung 10 × 10 **4 % / 4 %**", "Transport 3 × 5 und 4 × 8 **100 % / 100 %**", "**1.5 und 1.0**", "y₂ ∈ [1, 1.5], y₃ ∈ [0, 1], y₄ ∈ [0, 3]")
    _has("Zufallsinstanzen sind nie entartet (0 %), Mischinstanzen fast nie (0 % bis 4 %), Transportprobleme **immer**")
    assert "0 % bis 4 %" in APP_SRC


def test_rampenzeit_limits_and_sperrgut():
    up, down = ev.check_limit(Settings("centre", resource=2), 1), ev.check_limit(Settings("centre", resource=2), -1)
    assert round(up["rows"][1]["y"], 2) == 0.0 and round(down["rows"][1]["y"], 2) == 7.25
    _has("oben fällt y auf 0, unten steigt es auf 7.25")


def test_literature_lines_are_in_the_readme_and_the_app():
    _has("Gale, D., Kuhn, H. W., & Tucker, A. W. (1951)", "Dantzig, G. B. (1963)", "Gal, T. (1979)")
    for s in ("Gale, D., Kuhn, H. W., & Tucker, A. W. (1951)", "Dantzig, G. B. (1963)", "Gal, T. (1979)"):
        assert s in APP_SRC
