"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz, Ressourcen- und Dienst-Auswahl, δ-Randwerte, Zukauf, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dua_constants as C
import dua_scenario as S

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="dua_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label.startswith(label))


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()


def test_default_run_shows_the_centre_with_strong_duality():
    at = _run()
    _ok(at)
    assert {"Optimalwert", "Bindend", "Größter Preis", "Entartet"} <= {m.label for m in at.metric}
    assert _metric(at, "Optimalwert").value == "720.00" and _metric(at, "Bindend").value == "3 von 4" and _metric(at, "Größter Preis").value == "5.00" and _metric(at, "Entartet").value == "nein"
    assert any("Starke Dualität" in s.value and "720.00" in s.value for s in at.success) and len(at.dataframe) == 2 and at.get("plotly_chart")


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    _click(at, f"preset_{name}")
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["dua_step"], ss["resource_select"], ss["var_select"], ss["price_select"]) == (p["kind"], p["step"], p["res"], p["var"], p["price"])
    if "delta" in p:
        assert ss["delta_pct"] == p["delta"]


@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", list(S.KINDS))
def test_every_step_runs_for_every_kind(step, kind):
    at = _run(kind_select=kind, m_slider=5, n_slider=6, k_slider=2, l_slider=3, dua_step=step)
    _ok(at)
    assert at.session_state["dua_step"] == step
    if kind in ("infeasible", "unbounded"):
        assert any("keine Duale" in w.value for w in at.warning)


@pytest.mark.parametrize("kind", ["textbook", "centre", "degenerate", "mixed", "transport"])
def test_every_resource_and_variable_can_be_selected_on_the_analysis_steps(kind):
    base = _run(kind_select=kind, m_slider=5, n_slider=5, k_slider=2, l_slider=3)
    n_res = len(next(s for s in base.selectbox if s.key == "resource_select").options)
    n_var = len(next(s for s in base.selectbox if s.key == "var_select").options)
    for step in (2, 3):
        for i in {0, n_res - 1}:
            for j in {0, n_var - 1}:
                _ok(_run(step=step, kind_select=kind, m_slider=5, n_slider=5, k_slider=2, l_slider=3, resource_select=i, var_select=j))


def test_resource_selection_is_clamped_when_the_instance_gets_smaller():
    at = _run(kind_select="random", m_slider=10, n_slider=10, resource_select=9, var_select=9)
    _ok(at)
    at.session_state["kind_select"] = "textbook"
    at.run()
    _ok(at)
    assert at.session_state["resource_select"] == 2 and at.session_state["var_select"] == 1


def test_step_two_delta_slider_shows_exact_inside_and_changed_outside():
    at = _run(step=2, resource_select=1)
    _ok(at)
    at.select_slider(key="delta_pct").set_value(5).run()
    _ok(at)
    assert any("innerhalb" in m.value and "exakt" in m.value for m in at.markdown)
    at.select_slider(key="delta_pct").set_value(75).run()
    _ok(at)
    assert any("außerhalb" in m.value and "75.00" in m.value and "8.75" in m.value for m in at.markdown) and at.dataframe
    for v in (-50, 0, 100):
        at.select_slider(key="delta_pct").set_value(v).run()
        _ok(at)


def test_purchase_block_for_less_equal_and_other_resources():
    at = _run(step=2, resource_select=0)
    _ok(at)
    assert any("Kaufe 135.0 Einheiten" in m.value and "405.00" in m.value for m in at.markdown)
    at.select_slider(key="price_widget").set_value(4.5).run()
    _ok(at)
    assert at.session_state["price_select"] == 4.5 and any("Kaufe 10.0 Einheiten" in m.value for m in at.markdown)
    at.select_slider(key="price_widget").set_value(0.25).run()
    _ok(at)
    free = _run(step=2, resource_select=3)
    _ok(free)
    assert any("Nichts zu kaufen" in i.value for i in free.info)
    seed = next(s for s in range(60) if S.GE in S.generate("mixed", 8, 8, C.DENSITY, s).senses)
    ge = S.generate("mixed", 8, 8, C.DENSITY, seed).senses.index(S.GE)
    other = _run(step=2, kind_select="mixed", m_slider=8, n_slider=8, seed_input=seed, resource_select=ge)
    _ok(other)
    assert any("nur für ≤-Ressourcen" in i.value for i in other.info)


def test_step_three_limits_and_cost_ranges_and_noise_on_demand():
    at = _run(step=3, resource_select=2, var_select=3)
    _ok(at)
    assert len(at.get("plotly_chart")) == 2 and any("knapp innerhalb" in str(d.value.to_dict()) for d in at.dataframe)
    at.radio(key="limit_side").set_value(-1).run()
    _ok(at)
    at.radio(key="cost_side").set_value(-1).run()
    _ok(at)
    assert any("keine endliche Kostengrenze" in i.value for i in at.info)
    _click(at, "flip_start")
    _ok(at)
    assert len(at.get("plotly_chart")) == 3 and any("Anteil der 50 Läufe" in c.value for c in at.caption)
    free = _run(step=3, resource_select=3)
    assert any("keine endliche Grenze" in i.value for i in free.info)


def test_step_four_face_and_shares_on_demand():
    at = _run(step=4, kind_select="degenerate")
    _ok(at)
    assert any("nicht eindeutig" in w.value for w in at.warning) and at.get("plotly_chart")
    _click(at, "deg_start")
    _ok(at)
    table = at.dataframe[-1].value
    assert list(table["entartete Endbasis"])[2:] == ["100%", "100%"] and list(table["Duale nicht eindeutig"])[0] == "0%"
    ok = _run(step=4)
    _ok(ok)
    assert any("eindeutig" in s.value for s in ok.success)


@pytest.mark.parametrize("kw", [dict(kind_select="random", m_slider=C.M_MAX, n_slider=C.N_MAX), dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MIN), dict(kind_select="mixed", m_slider=C.M_MAX, n_slider=C.N_MAX),
                                dict(kind_select="transport", k_slider=C.K_MAX, l_slider=C.L_MAX), dict(kind_select="transport", k_slider=C.K_MIN, l_slider=C.L_MIN),
                                dict(kind_select="random", m_slider=6, n_slider=2)])
def test_extreme_settings_run_on_every_step(kw):
    for step in (1, 2, 3, 4):
        _ok(_run(step=step, **kw))


def test_two_service_instances_draw_the_geometry():
    for kind in ("textbook", "degenerate"):
        at = _run(kind_select=kind)
        _ok(at)
        assert at.get("plotly_chart")
    rnd = _run(kind_select="random", m_slider=5, n_slider=2)
    _ok(rnd)
    assert any("Pfeile" in c.value or "Kräftegleichgewicht" in m.value for m in rnd.markdown for c in rnd.caption) or rnd.get("plotly_chart")


def test_dice_button_changes_the_seed():
    at = _run(kind_select="random")
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old and at.session_state["seed_widget"] == at.session_state["seed_input"]


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(m="999", n="1", k="1", l="99", step="9", kind="nope", res="99", var="-3", price="7", seed="-4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["m_slider"], ss["n_slider"], ss["k_slider"], ss["l_slider"], ss["dua_step"], ss["kind_select"], ss["resource_select"], ss["var_select"], ss["price_select"], ss["seed_input"]) == (
        C.M_MAX, C.N_MIN, C.K_MIN, C.L_MAX, 1, "centre", 3, 0, C.DEFAULT_PRICE, 0)


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(kind="mixed", m="8", n="7", seed="7", res="2", var="3", price="0.5", step="3").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["m_slider"], ss["n_slider"], ss["seed_input"], ss["resource_select"], ss["var_select"], ss["price_select"], ss["dua_step"]) == ("mixed", 8, 7, 7, 2, 3, 0.5, 3)


def test_sidebar_shows_the_controls_that_belong_to_the_instance():
    fixed = _run()
    assert not any(w.key in ("m_widget", "k_widget") for w in fixed.slider) and not any(n.key == "seed_widget" for n in fixed.number_input)
    assert any(s.key == "resource_select" for s in fixed.selectbox) and any(s.key == "var_select" for s in fixed.selectbox)
    rnd = _run(kind_select="random")
    assert any(w.key == "m_widget" for w in rnd.slider) and any(w.key == "n_widget" for w in rnd.slider) and any(n.key == "seed_widget" for n in rnd.number_input)
    tr = _run(kind_select="transport")
    assert any(w.key == "k_widget" for w in tr.slider) and any(w.key == "l_widget" for w in tr.slider) and not any(w.key == "m_widget" for w in tr.slider)


def test_changing_kind_and_step_on_later_steps_does_not_crash():
    for step in (2, 3, 4):
        at = _run(step=step)
        _ok(at)
        for kw in (dict(kind_select="mixed", m_slider=8, n_slider=8), dict(kind_select="transport"), dict(kind_select="infeasible"), dict(kind_select="unbounded"), dict(kind_select="degenerate"),
                   dict(kind_select="textbook"), dict(kind_select="random", m_slider=4, n_slider=3), dict(kind_select="centre", resource_select=3)):
            for k, v in kw.items():
                at.session_state[k] = v
            at.run()
            _ok(at)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Gale, D., Kuhn, H. W., & Tucker, A. W. (1951)" in m.value and "Dantzig, G. B. (1963)" in m.value for m in at.markdown)
