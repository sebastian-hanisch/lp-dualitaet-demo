"""Plotly-Abbildungen: Geometrie der Dualität (2 Dienste), Schattenpreise und Schlupf, Wertfunktion mit Tangente und Bereich, Zukaufstreppe, Bereichsbalken, Dual-Seitenfläche.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen; bei gleichem Maßstab (scaleanchor) gibt es keine expliziten Achsenbereiche."""

import math

import numpy as np
import plotly.graph_objects as go

import dua_scenario as S

TEAL, ORANGE, RED, BLUE, GREY, PURPLE = "#2F6B65", "#e8a13a", "#d62728", "#1f4e9c", "#8a8f98", "#7b3fbf"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _vertices(inst):
    """Ecken der zulässigen Menge im Zwei-Dienste-Bild: Schnittpunkte aller Bedingungsgeraden und Achsen, die alle Bedingungen erfüllen."""
    A_, b, _c = inst.arrays()
    lines = [(A_[i], b[i]) for i in range(inst.m)] + [(np.array([1.0, 0.0]), 0.0), (np.array([0.0, 1.0]), 0.0)]
    pts = []
    for p in range(len(lines)):
        for q in range(p + 1, len(lines)):
            M = np.array([lines[p][0], lines[q][0]])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            x = np.linalg.solve(M, np.array([lines[p][1], lines[q][1]]))
            if x.min() < -1e-9:
                continue
            ok = True
            for i in range(inst.m):
                act = float(A_[i] @ x)
                s = inst.senses[i]
                if (s == S.LE and act > b[i] + 1e-9) or (s == S.GE and act < b[i] - 1e-9) or (s == S.EQ and abs(act - b[i]) > 1e-9):
                    ok = False
            if ok and not any(np.allclose(x, q_, atol=1e-9) for q_ in pts):
                pts.append(x)
    return pts


def build_geometry(inst, sens):
    """Zwei Dienste: zulässige Menge, Bedingungsgeraden (bindende hervorgehoben), Optimum; der Zielvektor c ist die Summe der mit den Schattenpreisen gewichteten Normalen der bindenden Bedingungen (Pfeile)."""
    pts = _vertices(inst)
    ctr = np.mean(pts, axis=0)
    order = sorted(range(len(pts)), key=lambda k: float(np.arctan2(pts[k][1] - ctr[1], pts[k][0] - ctr[0])))
    poly = [pts[k] for k in order]
    xmax = max(p[0] for p in pts) * 1.2 + 0.5
    ymax = max(p[1] for p in pts) * 1.2 + 0.5
    A_, b, c = inst.arrays()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p[0] for p in poly] + [poly[0][0]], y=[p[1] for p in poly] + [poly[0][1]], mode="lines", fill="toself", fillcolor="rgba(47,107,101,0.13)", line=dict(color=GREY, width=1), name="zulässig",
                             hoverinfo="skip"))
    for i in range(inst.m):
        a1, a2 = A_[i]
        if abs(a2) > 1e-12:
            xs = np.array([0.0, xmax])
            ys = (b[i] - a1 * xs) / a2
        else:
            xs = np.array([b[i] / a1] * 2)
            ys = np.array([0.0, ymax])
        bind = sens.binding[i]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=ORANGE if bind else GREY, width=3 if bind else 1.2, dash="solid" if bind else "dot"), name=f"{inst.row_names[i]}" + (f" (y = {sens.y[i]:.2f})" if bind else ""),
                                 hoverinfo="skip"))
    x = np.array(sens.x)
    fig.add_trace(go.Scatter(x=[x[0]], y=[x[1]], mode="markers", marker=dict(size=13, color=RED, line=dict(width=2, color="white")), name="Optimum", hoverinfo="skip"))
    vecs = [(c, "c", BLUE)] + [(sens.y[i] * A_[i], f"{sens.y[i]:.2f}·a{i + 1}", PURPLE) for i in range(inst.m) if sens.binding[i] and abs(sens.y[i]) > 1e-9]
    longest = max(float(np.hypot(*v)) for v, _l, _c in vecs) or 1.0
    scale = 0.5 * min(xmax, ymax) / longest
    tip = x.copy()
    for v, label, color in vecs[1:]:
        end = tip + scale * v
        fig.add_annotation(x=end[0], y=end[1], ax=tip[0], ay=tip[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2.5, arrowcolor=color)
        fig.add_annotation(x=(tip[0] + end[0]) / 2, y=(tip[1] + end[1]) / 2, text=label, showarrow=False, font=dict(color=color, size=12), bgcolor="rgba(255,255,255,0.75)", yshift=10)
        tip = end
    end = x + scale * c
    fig.add_annotation(x=end[0], y=end[1], ax=x[0], ay=x[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=3.5, arrowcolor=BLUE)
    fig.add_annotation(x=end[0], y=end[1], text="c", showarrow=False, font=dict(color=BLUE, size=14), bgcolor="rgba(255,255,255,0.75)", xshift=12, yshift=8)
    fig.update_xaxes(title_text=inst.names[0], scaleanchor="y", rangemode="tozero")
    fig.update_yaxes(title_text=inst.names[1], rangemode="tozero")
    return _base(fig, 470, legend_y=-0.35)


def build_duals(inst, sens):
    """Schattenpreise je Ressource (Balken) mit Schlupf als zweite Abbildung: bindende Ressourcen haben einen Preis, Schlupf-Ressourcen keinen."""
    names = list(inst.row_names)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=list(sens.y), marker_color=[ORANGE if bd else GREY for bd in sens.binding], text=[f"{v:.2f}" for v in sens.y], textposition="outside", name="Schattenpreis y"))
    fig.update_yaxes(title_text="Schattenpreis (Optimalwert je Einheit)", zeroline=True)
    fig.update_xaxes(tickangle=-25)
    return _base(fig, 300)


def build_value_curve(inst, i, pts, sens, delta=None, actual=None):
    """Wertfunktion z*(b_i) (Knickpunkte durch Neulösung), Tangente der Steigung y_i im Bereich, Bereichsband; optional Vorhersage (y_i·delta) und tatsächlicher Wert an b_i + delta."""
    b0 = inst.b[i]
    lo, hi = sens.rhs_range[i]
    fig = go.Figure()
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    x_lo, x_hi = min(xs), max(xs)
    band_lo = max(x_lo, b0 + lo) if not math.isinf(lo) else x_lo
    band_hi = min(x_hi, b0 + hi) if not math.isinf(hi) else x_hi
    fig.add_vrect(x0=band_lo, x1=band_hi, fillcolor="rgba(47,107,101,0.12)", line_width=0)
    fig.add_trace(go.Scatter(x=xs, y=zs, mode="lines+markers", line=dict(color=TEAL, width=3), marker=dict(size=6), name="Optimalwert z*(b_i) (Neulösung)"))
    tx = [x_lo, x_hi]
    fig.add_trace(go.Scatter(x=tx, y=[sens.obj + sens.y[i] * (t - b0) for t in tx], mode="lines", line=dict(color=ORANGE, width=2, dash="dash"), name=f"Tangente, Steigung y = {sens.y[i]:.2f}"))
    fig.add_trace(go.Scatter(x=[b0], y=[sens.obj], mode="markers", marker=dict(size=12, color=BLUE, line=dict(width=2, color="white")), name="aktuell"))
    if delta is not None and abs(delta) > 1e-12:
        fig.add_trace(go.Scatter(x=[b0 + delta], y=[sens.obj + sens.y[i] * delta], mode="markers", marker=dict(size=12, color=ORANGE, symbol="diamond"), name="Vorhersage y·Δ"))
        if actual is not None and not math.isnan(actual):
            fig.add_trace(go.Scatter(x=[b0 + delta], y=[sens.obj + actual], mode="markers", marker=dict(size=12, color=RED, symbol="x"), name="Neulösung"))
    fig.update_xaxes(title_text=f"{inst.row_names[i]}: rechte Seite b")
    fig.update_yaxes(title_text="Optimalwert")
    return _base(fig, 380, legend_y=-0.4)


def build_purchase(steps, price, y_end=None):
    """Zukaufstreppe: Grenznutzen (Schattenpreis) je Bereich gegen die Menge; solange die Treppe über dem Preis liegt, lohnt der Zukauf."""
    ys, cum = [], 0.0
    fig = go.Figure()
    for delta, y in steps:
        fig.add_trace(go.Scatter(x=[cum, cum + delta], y=[y, y], mode="lines", line=dict(color=TEAL, width=4), showlegend=False, hoverinfo="skip"))
        cum += delta
        ys.append(y)
    if steps and y_end is not None:
        fig.add_trace(go.Scatter(x=[cum, cum * 1.15 + 1], y=[y_end, y_end], mode="lines", line=dict(color=GREY, width=4, dash="dot"), showlegend=False, hoverinfo="skip"))
    top = max(ys + [price]) * 1.3 + 0.1
    fig.add_hline(y=price, line=dict(color=RED, dash="dash"), annotation_text=f"Preis {price:g}", annotation_position="top left")
    fig.update_xaxes(title_text="zugekaufte Menge", rangemode="tozero")
    fig.update_yaxes(title_text="Schattenpreis (Grenznutzen)", range=[0, top])
    return _base(fig, 300)


def build_ranges(inst, sens, kind="rhs"):
    """Bereichsbalken: rechte Seiten (kind = rhs) bzw. Deckungsbeiträge (kind = cost); der Punkt ist der aktuelle Wert, unendliche Grenzen laufen bis zum Rand."""
    fig = go.Figure()
    if kind == "rhs":
        names = list(inst.row_names)
        cur = [inst.b[i] for i in range(inst.m)]
        lims = [(cur[i] + sens.rhs_range[i][0], cur[i] + sens.rhs_range[i][1]) for i in range(inst.m)]
    else:
        names = list(inst.names)
        cur = list(inst.c)
        lims = list(sens.cost_range)
    finite = [v for pair in lims for v in pair if not math.isinf(v)] + cur
    span = max(finite) - min(finite) or 1.0
    lo_edge, hi_edge = min(finite) - 0.15 * span, max(finite) + 0.15 * span
    for k, ((lo, hi), c0) in enumerate(zip(lims, cur)):
        a = lo_edge if math.isinf(lo) else lo
        b_ = hi_edge if math.isinf(hi) else hi
        fig.add_trace(go.Scatter(x=[a, b_], y=[names[k]] * 2, mode="lines", line=dict(color=TEAL, width=10), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[c0], y=[names[k]], mode="markers", marker=dict(size=11, color=ORANGE, line=dict(width=2, color="white")), showlegend=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="rechte Seite b (Balken = Bereich, Punkt = aktuell)" if kind == "rhs" else "Deckungsbeitrag c (Balken = Bereich, Punkt = aktuell)", range=[lo_edge, hi_edge])
    return _base(fig, 60 + 46 * len(names), legend_y=-0.2)


def build_face(inst, ranges, ys):
    """Optimale Dual-Seitenfläche: je Ressource das Intervall aller optimalen Schattenpreise (Balken), der vom Löser gewählte Wert als Punkt; unendliche Enden laufen bis zum Rand."""
    names = list(inst.row_names)
    finite = [v for pair in ranges for v in pair if not (math.isinf(v) or math.isnan(v))] + list(ys)
    span = (max(finite) - min(finite)) or 1.0
    lo_edge, hi_edge = min(finite) - 0.25 * span, max(finite) + 0.25 * span
    fig = go.Figure()
    for k, ((lo, hi), y) in enumerate(zip(ranges, ys)):
        a = lo_edge if math.isinf(lo) or math.isnan(lo) else lo
        b_ = hi_edge if math.isinf(hi) or math.isnan(hi) else hi
        fig.add_trace(go.Scatter(x=[a, b_], y=[names[k]] * 2, mode="lines", line=dict(color=PURPLE, width=10), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[y], y=[names[k]], mode="markers", marker=dict(size=11, color=ORANGE, line=dict(width=2, color="white")), showlegend=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Schattenpreis (Balken = alle optimalen Duale, Punkt = gewählt)", range=[lo_edge, hi_edge])
    return _base(fig, 60 + 46 * len(names), legend_y=-0.2)


def build_flip(rates, labels):
    """Anteil der Läufe, in denen Kostenrauschen die Basis ändert, je Rauschstärke."""
    fig = go.Figure(go.Bar(x=labels, y=rates, marker_color=TEAL, text=[f"{v:.0%}" for v in rates], textposition="outside"))
    fig.update_yaxes(title_text="Basis ändert sich", tickformat=".0%", range=[0, min(1.05, max(rates) * 1.25 + 0.05)])
    fig.update_xaxes(title_text="Rauschstärke der Deckungsbeiträge (multiplikativ)")
    return _base(fig, 300)
