"""Dualität und Sensitivität – was ein Schattenpreis sagt und wie lange er gilt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Lineare-Programmierung-Reihe der "Konzepte"-Reihe: Die Duale y der Endbasis sind die Schattenpreise der Ressourcen. Die Demo zeigt die Zertifikate (starke Dualität, komplementärer Schlupf),
die Steigung der Wertfunktion und wie weit sie trägt (Ranging), den Zukauf von Kapazität und die Entartung, bei der die Duale nicht eindeutig sind.

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import dua_constants as C
import dua_evaluation as ev
import dua_scenario as S
import dua_sensitivity as D
from dua_evaluation import Settings, analyse
from dua_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from dua_visualization import (
    build_duals,
    build_face,
    build_flip,
    build_geometry,
    build_purchase,
    build_ranges,
    build_value_curve,
)

st.set_page_config(page_title="Dualität und Sensitivität – Sebastian Hanisch", layout="wide")


def thousands(x):
    return f"{int(round(x)):,}".replace(",", " ")


def num(x, digits=2):
    if isinstance(x, float) and math.isinf(x):
        return "∞" if x > 0 else "−∞"
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    x = 0.0 if abs(x) < 5e-13 else x
    return f"{x:.{digits}f}"


STATUS_TEXT = {"optimal": "Optimum", "infeasible": "unzulässig", "unbounded": "unbeschränkt", "limit": "Pivot-Grenze erreicht"}

st.title("⚖️ Dualität und Sensitivität – was ein Schattenpreis sagt und wie lange er gilt")
st.markdown(
    """
**Fünftes Stück der Lineare-Programmierung-Reihe.** Bisher waren die Duale y nur ein Nebenprodukt der Endbasis. Hier sind sie das Thema: **y_i ist der Wert einer zusätzlichen Einheit der Ressource i** - was ein Distributionszentrum für eine
Stunde mehr Kommissionierung oder einen Quadratmeter mehr Lager zahlen dürfte. Vier Fragen, alle gemessen: **(1) Die Duale** - stimmen starke Dualität und komplementärer Schlupf, und was heißt y geometrisch? **(2) Schattenpreis** - y ist
die Steigung des Optimalwerts über die rechte Seite, aber wie weit trägt sie, und was passiert beim Zukauf? **(3) Ranging** - in welchem Bereich bleibt die Endbasis optimal, und wie oft kippt sie schon bei wenig Rauschen?
**(4) Entartung** - wann sind die Duale nicht eindeutig?
"""
)
st.caption("Kind von [Tableau-Simplex](https://github.com/sebastian-hanisch/tableau-simplex-demo). Folgestück (Dualer Simplex und Neuoptimierung) ist [noch nicht gebaut].")

with st.expander("So funktioniert die Dualität", expanded=True):
    st.markdown(
        """
1. **Primal und dual:** das LP max c·x unter A x ≤ b, x ≥ 0 hat das Dual min b·y unter Aᵀy ≥ c, y ≥ 0. **Starke Dualität:** im Optimum sind beide Werte gleich, c·x = b·y. Die Duale stehen im Endtableau (y = c_B B⁻¹) - der Löser rechnet sie mit.
2. **Komplementärer Schlupf:** eine Ressource mit Schlupf hat den Schattenpreis 0 (mehr davon nützt nichts), ein Dienst mit reduzierten Kosten r_j > 0 wird nicht produziert. Nur bindende Ressourcen haben einen Preis.
3. **Geometrie (zwei Dienste):** der Zielvektor c ist die Summe der Normalen der bindenden Ressourcen, gewichtet mit ihren Schattenpreisen: c = Σ yᵢ·aᵢ - ein Kräftegleichgewicht im Optimum.
4. **Schattenpreis = Steigung:** z*(b_i) ist konkav und stückweise linear; y_i ist die Steigung des Stücks, auf dem die aktuelle Basis optimal bleibt. Der **Bereich** (Ranging) sagt, wie weit b_i sich ändern darf, bevor die Basis wechselt und y_i springt.
5. **Entartung:** ist die Endbasis entartet (eine Basisvariable 0), gibt es mehrere optimale Duale; links- und rechtsseitige Steigung unterscheiden sich, und der Löser wählt einen von ihnen.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:7], preset_names[7:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Instanz", options=list(S.KINDS), format_func=lambda v: S.KIND_LABELS[v], key="kind_select",
                        help="Lehrbuchbeispiel, Zentrum, entartete Ecke und die Sonderfälle sind fest; Zufall, Mischung (≥ und =) und Transport (entartet) sind regelbar.")
    random_kind = kind not in S.FIXTURE_KINDS
    if random_kind and kind == "transport":
        m = st.slider("Lager", *bounds("k_slider"), value=int(ss["k_slider"]), key="k_widget", on_change=store_from_widget, args=("k_slider",), help="Lager (Angebot); Angebot gleich Nachfrage macht die Instanz entartet.")
        n = st.slider("Kunden", *bounds("l_slider"), value=int(ss["l_slider"]), key="l_widget", on_change=store_from_widget, args=("l_slider",), help="Kunden (Nachfrage).")
    elif random_kind:
        m = st.slider("Ressourcen m", *bounds("m_slider"), value=int(ss["m_slider"]), key="m_widget", on_change=store_from_widget, args=("m_slider",), help="Zahl der Bedingungen.")
        n = st.slider("Dienste n", *bounds("n_slider"), value=int(ss["n_slider"]), key="n_widget", on_change=store_from_widget, args=("n_slider",), help="Zahl der Variablen; bei n = 2 gibt es die Zeichnung.")
    else:
        m, n = C.DEFAULT_M, C.DEFAULT_N
    if random_kind:
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        seed = C.DEFAULT_SEED
    tmp = S.generate(kind, int(m), int(n), C.DENSITY, int(seed))
    ss["resource_select"] = min(int(ss["resource_select"]), tmp.m - 1)
    ss["var_select"] = min(int(ss["var_select"]), tmp.n - 1)
    st.selectbox("Betrachtete Ressource", options=list(range(tmp.m)), format_func=lambda i: tmp.row_names[i], key="resource_select", help="Wirkt auf Schritt 2 (Wertfunktion, Zukauf) und Schritt 3 (Grenze prüfen).")
    st.selectbox("Betrachteter Dienst", options=list(range(tmp.n)), format_func=lambda j: tmp.names[j], key="var_select", help="Wirkt auf Schritt 3 (Kostenbereich prüfen).")

sync_query_params({"kind_select": kind, "m_slider": int(ss["m_slider"]), "n_slider": int(ss["n_slider"]), "k_slider": int(ss["k_slider"]), "l_slider": int(ss["l_slider"]), "seed_input": int(ss["seed_input"]),
                   "resource_select": int(ss["resource_select"]), "var_select": int(ss["var_select"]), "price_select": float(ss["price_select"]), "dua_step": int(ss["dua_step"])})

settings = Settings(kind, int(m), int(n), int(seed), int(ss["resource_select"]), float(ss["price_select"]), int(ss["var_select"]))
with st.spinner("Rechne..."):
    a = analyse(settings)
inst, sens = a.inst, a.sens
ri, vj = settings.res_i, settings.var_j

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Schattenpreise und ihre Grenzen")
step = st.select_slider("Schritt", options=list(C.STEPS), key="dua_step", format_func=lambda s: C.STEPS[s])

if sens is None:
    st.warning(f"Die Instanz ist **{STATUS_TEXT[a.status]}**: es gibt kein Optimum und damit keine Duale. Zu einer unzulässigen Instanz gehört ein Farkas-Zertifikat, zu einer unbeschränkten ein unzulässiges Dual - hier keine Analyse.")
elif step == 1:
    gap = abs(sens.obj - sens.dual_obj)
    st.success(f"✅ **Starke Dualität:** c·x = {num(sens.obj)} = b·y = {num(sens.dual_obj)} (Lücke {gap:.1e}). Komplementärer Schlupf: größtes |y·Schlupf| = "
               f"{max(abs(y * s) for y, s in zip(sens.y, sens.slack)):.1e}, größtes |x·r| = {max(abs(x * r) for x, r in zip(sens.x, sens.reduced)):.1e}.")
    used = [inst.b[i] - sens.slack[i] if inst.senses[i] == S.LE else inst.b[i] + sens.slack[i] for i in range(inst.m)]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ressourcen:** Bestand, Nutzung, Schlupf und Schattenpreis")
        st.dataframe(pd.DataFrame([{"Ressource": inst.row_names[i], "Art": inst.senses[i], "Bestand b": num(inst.b[i]), "genutzt": num(used[i]), "Schlupf": num(sens.slack[i]), "Schattenpreis y": num(sens.y[i]),
                                    "y · Schlupf": num(sens.y[i] * sens.slack[i])} for i in range(inst.m)]), hide_index=True, width="stretch")
    with c2:
        st.markdown("**Dienste:** Menge, Deckungsbeitrag und reduzierte Kosten")
        st.dataframe(pd.DataFrame([{"Dienst": inst.names[j], "Menge x": num(sens.x[j]), "Deckungsbeitrag c": num(inst.c[j]), "reduzierte Kosten r": num(sens.reduced[j]), "x · r": num(sens.x[j] * sens.reduced[j])}
                                   for j in range(inst.n)]), hide_index=True, width="stretch")
    if inst.n == 2:
        st.plotly_chart(build_geometry(inst, sens), width="stretch", key="s1_geo")
        st.caption("Orange = bindende Ressourcen, rot = Optimum. Die Pfeile am Optimum: die mit den Schattenpreisen gewichteten Normalen der bindenden Ressourcen (violett), aneinandergelegt ergeben sie den Zielvektor c (blau). "
                   + ("Drei Ressourcen binden an der entarteten Ecke: die Zerlegung ist nicht eindeutig, die Zeichnung zeigt eine davon." if sens.degenerate else ""))
    else:
        st.plotly_chart(build_duals(inst, sens), width="stretch", key="s1_duals")
        st.caption("Orange = bindende Ressourcen mit ihrem Schattenpreis, grau = Ressourcen mit Schlupf (Preis 0). Bei ≥-Ressourcen ist der Schattenpreis ≤ 0 (mehr Mindestmenge kostet), bei =-Ressourcen hat er ein freies Vorzeichen.")
elif step == 2:
    pts = ev.pieces(settings)
    ss.setdefault("delta_pct", 0)
    sel = st.select_slider("Änderung der rechten Seite (in % des Bestands)", options=list(C.DELTA_OPTIONS), key="delta_pct", format_func=lambda v: f"{v:+d} %",
                           help="Verschiebt b_i der betrachteten Ressource; das Diagramm zeigt die Vorhersage y·Δ und den Wert nach Neulösung.")
    b0 = inst.b[ri]
    delta = sel / 100.0 * (abs(b0) if abs(b0) > 1e-12 else 1.0)
    pred = ev.prediction_row(inst, sens, ri, delta) if abs(delta) > 1e-12 else None
    lo, hi = sens.rhs_range[ri]
    st.markdown(f"**{inst.row_names[ri]}:** Schattenpreis **y = {num(sens.y[ri])}** gilt für b zwischen {num(b0 + lo)} und {num(b0 + hi)} (Bereich {num(lo)} bis {num(hi)}); Schlupf {num(sens.slack[ri])}.")
    if pred is not None:
        gap_txt = f"Vorhersage y·Δ = {num(pred['predicted'])}, Neulösung {num(pred['actual'])}" if not math.isnan(pred["actual"]) else f"Vorhersage y·Δ = {num(pred['predicted'])}, Neulösung: unzulässig"
        st.markdown(f"Δ = {num(delta)} ({sel:+d} %): {gap_txt} - {'**innerhalb** des Bereichs: exakt.' if pred['in_range'] else '**außerhalb** des Bereichs: die Steigung hat sich geändert.'}")
    st.plotly_chart(build_value_curve(inst, ri, pts, sens, delta if pred is not None else None, pred["actual"] if pred is not None else None), width="stretch", key="s2_curve")
    st.caption("Grün: Bereich der Endbasis. Die Kurve (Neulösung, Knickpunkte exakt) ist konkav: jede zusätzliche Einheit bringt höchstens so viel wie die vorige; die Tangente ist die Vorhersage der Schattenpreis-Regel.")
    rows = ev.prediction_table(settings)
    if rows:
        st.markdown("**Vorhersage gegen Neulösung** (Erweiterung als Vielfaches der oberen Bereichsgrenze):")
        st.dataframe(pd.DataFrame([{"Vielfaches": f"{r['factor']:g}", "Δ": num(r["delta"]), "Vorhersage y·Δ": num(r["predicted"]), "Neulösung": num(r["actual"]), "im Bereich": "ja" if r["in_range"] else "nein",
                                    "Überschätzung": num(r["predicted"] - r["actual"]) if not math.isnan(r["actual"]) else "-"} for r in rows]), hide_index=True, width="stretch")
    st.markdown("---")
    st.markdown("**Kapazität zukaufen:** eine Einheit der Ressource kostet den Preis p. Die Schattenpreis-Regel: kaufe bis zur Bereichsgrenze, solange y > p, dann neu bepreisen.")
    if inst.senses[ri] != S.LE:
        st.info("Zukauf gibt es nur für ≤-Ressourcen (Kapazitäten); die betrachtete Ressource ist eine Mindest- oder Gleichungs-Bedingung.")
    else:
        price = st.select_slider("Zukaufspreis je Einheit", options=list(C.PRICE_OPTIONS), value=float(ss["price_select"]), key="price_widget", on_change=store_from_widget, args=("price_select",),
                                 format_func=lambda v: f"{v:g}")
        pu = ev.purchase(Settings(kind, int(m), int(n), int(seed), int(ss["resource_select"]), float(price), int(ss["var_select"])))
        if pu is None or pu["unbounded"]:
            st.warning("Der Zukauf ist unbeschränkt: der Schattenpreis liegt über dem Preis, und der Bereich hat kein Ende.")
        elif not pu["steps"]:
            st.info(f"Nichts zu kaufen: der Schattenpreis {num(pu['y'])} liegt nicht über dem Preis {num(price)}. Der Optimalwert bleibt {num(sens.obj)}.")
        else:
            st.markdown(f"**Kaufe {num(pu['extra'], 1)} Einheiten** in {len(pu['steps'])} Schritten (Schattenpreis fällt von {num(pu['steps'][0][1])} auf {num(pu['steps'][-1][1])}): Gewinn nach Zukaufskosten **{num(pu['gain'])}**, "
                        f"Optimalwert {num(pu['value'])} (das exakte LP mit Zukaufsspalte liefert dasselbe). Die naive Rechnung (y − p) · Menge verspricht {num(pu['naive_gain'])}.")
            st.plotly_chart(build_purchase(pu["steps"], price, None), width="stretch", key="s2_purchase")
            st.caption("Grenznutzen je zugekaufter Menge: jede Stufe ist ein Bereich der Basis, danach fällt der Schattenpreis. Gekauft wird, solange die Treppe über der roten Preislinie liegt.")
elif step == 3:
    st.markdown("**Rechte Seiten:** in welchem Bereich bleibt die Endbasis optimal (und damit der Schattenpreis gültig)?")
    st.plotly_chart(build_ranges(inst, sens, "rhs"), width="stretch", key="s3_rhs")
    st.dataframe(pd.DataFrame([{"Ressource": inst.row_names[i], "b": num(inst.b[i]), "y": num(sens.y[i]), "von": num(inst.b[i] + sens.rhs_range[i][0]), "bis": num(inst.b[i] + sens.rhs_range[i][1]),
                                "Senkung bis": f"{-sens.rhs_range[i][0] / inst.b[i]:.0%}" if not math.isinf(sens.rhs_range[i][0]) and inst.b[i] > 0 else "-",
                                "Erhöhung bis": f"{sens.rhs_range[i][1] / inst.b[i]:.0%}" if not math.isinf(sens.rhs_range[i][1]) and inst.b[i] > 0 else "-"} for i in range(inst.m)]), hide_index=True, width="stretch")
    side = st.radio("Grenze prüfen", options=[1, -1], format_func=lambda s: "obere Grenze" if s == 1 else "untere Grenze", horizontal=True, key="limit_side")
    chk = ev.check_limit(settings, side)
    if chk is not None and chk["rows"]:
        st.markdown(f"**{inst.row_names[ri]}, Grenze bei Δ = {num(chk['limit'])}:** Neulösung knapp innerhalb und knapp außerhalb.")
        st.dataframe(pd.DataFrame([{"Lage": r["label"], "Δ": num(r["delta"]), "Ergebnis": r["status"], "Basis unverändert": "-" if r["same_basis"] is None else ("ja" if r["same_basis"] else "nein"), "neuer Schattenpreis": num(r["y"]),
                                    "Optimalwert": num(r["obj"])} for r in chk["rows"]]), hide_index=True, width="stretch")
    elif chk is not None:
        st.info(f"Auf dieser Seite hat die Ressource {inst.row_names[ri]} keine endliche Grenze (unbegrenzt oder Bereich der Breite 0).")
    st.markdown("**Deckungsbeiträge:** in welchem Bereich bleibt die Lösung (nicht nur der Wert) dieselbe?")
    st.plotly_chart(build_ranges(inst, sens, "cost"), width="stretch", key="s3_cost")
    st.dataframe(pd.DataFrame([{"Dienst": inst.names[j], "Menge x": num(sens.x[j]), "c": num(inst.c[j]), "von": num(sens.cost_range[j][0]), "bis": num(sens.cost_range[j][1]), "reduzierte Kosten": num(sens.reduced[j])}
                               for j in range(inst.n)]), hide_index=True, width="stretch")
    cside = st.radio("Kostengrenze prüfen", options=[1, -1], format_func=lambda s: "obere Grenze" if s == 1 else "untere Grenze", horizontal=True, key="cost_side")
    cchk = ev.check_cost_limit(settings, cside)
    if cchk is not None and cchk["rows"]:
        st.markdown(f"**{inst.names[vj]}, Grenze bei c = {num(cchk['limit'])}:** die Lösung x bleibt knapp innerhalb, knapp außerhalb ändert sie sich.")
        st.dataframe(pd.DataFrame([{"Lage": r["label"], "Deckungsbeitrag c": num(r["c"]), "Lösung unverändert": "ja" if r["same_solution"] else "nein", "Menge x": num(r["x_j"]), "Optimalwert": num(r["obj"])} for r in cchk["rows"]]),
                     hide_index=True, width="stretch")
    elif cchk is not None:
        st.info(f"Auf dieser Seite hat der Dienst {inst.names[vj]} keine endliche Kostengrenze.")
    st.markdown("**Wie oft kippt die Basis unter Kostenrauschen?** (50 Läufe, multiplikatives Rauschen der Deckungsbeiträge; 🔬 auf Abruf)")
    if st.button("Basiskipp berechnen", key="flip_start"):
        ss["flip_done"] = settings
    if ss.get("flip_done") == settings:
        with st.spinner("Rechne..."):
            rates = [ev.flip_rate(settings, s) for s in C.NOISE_SIGMAS]
        st.plotly_chart(build_flip(rates, [f"{s:.0%}" for s in C.NOISE_SIGMAS]), width="stretch", key="s3_flip")
        st.caption("Anteil der 50 Läufe, in denen die optimale Basis wechselt: die Bereiche sind nicht breit genug, um Rauschen in den Deckungsbeiträgen zu überstehen.")
else:
    face = ev.faces(settings)
    unique = all(hi - lo <= 1e-6 * max(1.0, abs(hi), abs(lo)) for lo, hi in face if not (math.isnan(lo) or math.isnan(hi) or math.isinf(lo) or math.isinf(hi))) and not any(math.isinf(v) for pair in face for v in pair)
    if unique:
        st.success("✅ Die Duale sind **eindeutig**: die optimale Dual-Seitenfläche ist ein Punkt, links- und rechtsseitige Schattenpreise stimmen überein.")
    else:
        st.warning("⚠️ Die Duale sind **nicht eindeutig**: es gibt mehrere optimale Schattenpreise. Der Löser wählt einen von ihnen; für die Entscheidung zählt das Intervall.")
    st.plotly_chart(build_face(inst, face, list(sens.y)), width="stretch", key="s4_face")
    rows = []
    for i in range(inst.m):
        left, right = D.one_sided_slopes(inst, i) if inst.m <= 8 else (float("nan"), float("nan"))
        rows.append({"Ressource": inst.row_names[i], "gewählter Schattenpreis": num(sens.y[i]), "kleinster optimaler": num(face[i][0]), "größter optimaler": num(face[i][1]), "rechtsseitige Steigung": num(right),
                     "linksseitige Steigung": num(left)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Balken = alle optimalen Schattenpreise (die optimale Seitenfläche des Dual-LP, gerechnet mit 2 m LPs), Punkt = vom Löser gewählt. Rechts- und linksseitige Steigung (Neulösung mit b_i ± ε; nur bis 8 Ressourcen) begrenzen das Intervall, wo beide Seiten zulässig sind. "
               "Beim Transportproblem ist die Seitenfläche unbeschränkt: alle Potenziale lassen sich gemeinsam verschieben, weil Angebot gleich Nachfrage ist.")
    st.markdown("**Wie häufig?** Anteil der 50 Instanzen mit entarteter Endbasis und mit nicht eindeutigen Dualen (🔬 auf Abruf).")
    if st.button("Entartungsanteile berechnen", key="deg_start"):
        ss["deg_done"] = True
    if ss.get("deg_done"):
        with st.spinner("Rechne..."):
            shares = [(label, ev.degeneracy_share(k, mm, nn)) for label, k, mm, nn in (("Zufall 10 × 10", "random", 10, 10), ("Mischung 10 × 10", "mixed", 10, 10), ("Transport 3 × 5", "transport", 3, 5),
                                                                                       ("Transport 4 × 8", "transport", 4, 8))]
        st.dataframe(pd.DataFrame([{"Instanzen": label, "Läufe": s["n"], "entartete Endbasis": f"{s['degenerate']:.0%}", "Duale nicht eindeutig": f"{s['non_unique']:.0%}"} for label, s in shares]), hide_index=True, width="stretch")

st.markdown("---")

# --- Kennzahlen --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ⚙️ Die gewählte Instanz")
m1, m2, m3, m4 = st.columns(4)
if sens is not None:
    m1.metric("Optimalwert", thousands(sens.obj) if abs(sens.obj) >= 1000 else num(sens.obj), delta=f"{inst.m} Ressourcen, {inst.n} Dienste", delta_color="off")
    m2.metric("Bindend", f"{sum(sens.binding)} von {inst.m}", delta="Schattenpreis > 0" if any(y > 1e-9 for y in sens.y) else "kein Preis", delta_color="off")
    top = max(range(inst.m), key=lambda i: abs(sens.y[i]))
    m3.metric("Größter Preis", num(sens.y[top]), delta=inst.row_names[top][:22], delta_color="off")
    m4.metric("Entartet", "ja" if sens.degenerate else "nein", delta="Duale eindeutig" if D.dual_unique(inst) else "Duale mehrdeutig", delta_color="off")
else:
    m1.metric("Optimalwert", "-", delta=STATUS_TEXT[a.status], delta_color="off")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Schattenpreis gilt für jede Menge.** | Nur im Bereich der Endbasis. Im Zentrum trägt der Preis der Lagerfläche (0.5) für 17.5 Einheiten mehr, danach bringt jede weitere nichts: bei 4 mal der Grenze sagt y·Δ 35 voraus, die Neulösung findet 8.8. | Wertfunktion, Ranging |
| **Zukaufen, solange y > Preis, ist einfach.** | Die Regel ist exakt, wenn man nach jedem Bereich neu bepreist; die naive Rechnung (y − Preis) mal Menge überschätzt: 405 statt 160 bei Kommissionierstunden zum Preis 2. | Dualer Simplex (nächstes Stück) |
| **Die Duale sind eindeutig.** | Bei entarteter Endbasis nicht: es gibt ein Intervall optimaler Schattenpreise. Zufalls- und Mischinstanzen sind fast nie entartet (0 % bis 4 %), Transportprobleme immer, mit unbeschränkter Dual-Seitenfläche. | Entartung, Störung |
| **Ein Bereich ist breit genug.** | Bereiche sind einzeln und eng: im Zentrum im Median +8.8 % und −12.5 % des Bestands; schon 5 % Rauschen auf den Deckungsbeiträgen ändert im Zentrum in 50 % und bei Zufall 10 × 10 in 26 % der Läufe die Basis. | Robuste Optimierung |
| **Änderungen wirken einzeln.** | Das Ranging gilt für einen Parameter allein; gleichzeitige Änderungen (100-%-Regel) sind hier nicht gebaut. | Parametrische Programmierung |
| **Duale gibt es immer.** | Bei unzulässigen und unbeschränkten Instanzen gibt es kein optimales Dual (nur ein Zertifikat der Unzulässigkeit bzw. der Unbeschränktheit): die Demo zeigt dann keine Analyse. | Farkas-Lemma |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Primal und Dual.** $\max c^\top x$ unter $Ax \le b,\ x \ge 0$ und $\min b^\top y$ unter $A^\top y \ge c,\ y \ge 0$; für $\ge$-Zeilen ist $y \le 0$, für $=$-Zeilen ist $y$ frei. **Starke Dualität** $c^\top x^* = b^\top y^*$; **komplementärer Schlupf** $y_i (b_i - a_i^\top x) = 0$ und $x_j (a_j^\top y - c_j) = 0$.

**Endtableau.** $y^\top = c_B^\top B^{-1}$ (unter den Einheitsspalten der Anfangsbasis), reduzierte Kosten $r_j = y^\top a_j - c_j \ge 0$. **Rechte Seite:** $x_B(\delta) = x_B + \delta\, B^{-1} e_i \ge 0$ liefert den Bereich $[\delta_{\min}, \delta_{\max}]$, in dem $y_i$ die Steigung von $z^*(b_i)$ ist.
**Kosten:** für eine Nichtbasisspalte $c_j \le y^\top a_j$; für eine Basisspalte in Zeile $p$ muss $r_k + \delta\, \alpha_{pk} \ge 0$ für alle Nichtbasisspalten $k$ gelten ($\alpha_{pk}$ = Tableau-Eintrag).

**Wertfunktion.** $z^*(b)$ ist konkav und stückweise linear in $b$, konvex in $c$; die Steigung ist $y$. Bei Entartung unterscheiden sich links- und rechtsseitige Steigung, und die optimalen Duale bilden eine Seitenfläche $\{y : A^\top y \ge c,\ b^\top y = z^*\}$.
**Zukauf:** LP mit einer Zusatzspalte $u \ge 0$ (Kapazität $b_i + u$, Kosten $p\,u$).

**Literatur.** Gale, D., Kuhn, H. W., & Tucker, A. W. (1951). *Linear programming and the theory of games.* In T. C. Koopmans (Hrsg.), Activity Analysis of Production and Allocation, 317-329. Wiley. Dantzig, G. B. (1963). *Linear Programming and Extensions.* Princeton University Press.
Gal, T. (1979). *Postoptimal Analyses, Parametric Programming, and Related Topics.* McGraw-Hill (nur genannt).

Implementiert in `dua_algorithm.py` (Tableau-Simplex mit Endtableau), `dua_sensitivity.py` (Duale, Ranging, Wertfunktion, Dual-Seitenfläche, Zukauf), `dua_evaluation.py`, `dua_scenario.py`.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
