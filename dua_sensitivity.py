"""Dualität und Sensitivität aus dem Endtableau: Duale y (Schattenpreise), reduzierte Kosten, Bereiche (Ranging) für rechte Seiten und Kosten, Wertfunktion, Entartung (links-/rechtsseitige Schattenpreise),
Dual-LP und der Zukauf von Kapazität."""

import math
from dataclasses import dataclass

import numpy as np

import dua_algorithm as A
import dua_scenario as S

INF = math.inf
TOL = 1e-9


@dataclass
class Sensitivity:
    inst: object
    sol: object
    y: tuple                         # Schattenpreise je Ressource (Ableitung des Optimums nach b_i im Bereich)
    slack: tuple                     # Schlupf (<=: b - Ax), Überschuss (>=: Ax - b), 0 bei =
    binding: tuple                   # Schlupf == 0
    reduced: tuple                   # reduzierte Kosten r_j = y a_j - c_j >= 0 der Dienste
    rhs_range: tuple                 # je Ressource (lo, hi): b_i + delta mit delta in [lo, hi] lässt die Endbasis optimal
    cost_range: tuple                # je Dienst (lo, hi): Deckungsbeitrag c_j' in [lo, hi] lässt die Endbasis optimal
    degenerate: bool                 # eine Basisvariable ist 0

    @property
    def x(self):
        return self.sol.x

    @property
    def obj(self):
        return self.sol.obj

    @property
    def dual_obj(self):
        return float(np.dot(self.y, self.inst.b))


def analyse_lp(inst):
    """Sensitivitätsanalyse der Endbasis; None, wenn die Instanz unzulässig oder unbeschränkt ist."""
    sol = A.solve(inst)
    if sol.status != "optimal":
        return None
    T, info = sol.T, sol.info
    m, n, ncols = info["m"], info["n"], info["ncols"]
    basis = list(sol.basis)
    art = set(info["art_col"].values())
    Binv = T[:m, list(sol.init_basis)]
    xB = np.maximum(T[:m, -1], 0.0)
    sign = info["sign"]
    rhs_range = []
    for i in range(m):
        g = sign[i] * Binv[:, i]
        lo, hi = -INF, INF
        for k in range(m):
            if basis[k] in art:
                if abs(g[k]) > TOL:
                    lo = hi = 0.0                      # eine redundante Gleichung: jede Änderung macht die Instanz unzulässig
                continue
            if g[k] > TOL:
                lo = max(lo, -xB[k] / g[k])
            elif g[k] < -TOL:
                hi = min(hi, xB[k] / -g[k])
        rhs_range.append((lo, hi))
    A_, b, c = inst.arrays()
    x = np.array(sol.x)
    act = A_ @ x
    slack = []
    for i in range(m):
        s = inst.senses[i]
        slack.append(float(b[i] - act[i]) if s == S.LE else (float(act[i] - b[i]) if s == S.GE else 0.0))
    slack = [0.0 if abs(v) < 1e-9 else v for v in slack]
    reduced = [float(T[m, j]) for j in range(n)]
    cost_range = []
    nonbasic = [k for k in range(ncols) if k not in basis and k not in art]
    for j in range(n):
        if j not in basis:
            cost_range.append((-INF, float(c[j]) + max(reduced[j], 0.0)))
            continue
        p = basis.index(j)
        lo, hi = -INF, INF
        for k in nonbasic:
            alpha, rk = T[p, k], max(T[m, k], 0.0)
            if alpha > TOL:
                lo = max(lo, -rk / alpha)
            elif alpha < -TOL:
                hi = min(hi, rk / -alpha)
        cost_range.append((float(c[j]) + lo, float(c[j]) + hi))
    degenerate = any(xB[k] <= 1e-9 * max(1.0, float(np.abs(b).max())) for k in range(m))
    return Sensitivity(inst, sol, tuple(sol.y), tuple(slack), tuple(v <= 1e-9 for v in slack), tuple(reduced), tuple(rhs_range), tuple(cost_range), degenerate)


def with_rhs(inst, i, delta):
    """Kopie der Instanz mit b_i + delta."""
    b = list(inst.b)
    b[i] += delta
    return S.Instance(inst.A, tuple(b), inst.c, inst.senses, inst.names, inst.row_names, inst.kind)


def with_cost(inst, j, value):
    """Kopie der Instanz mit dem Deckungsbeitrag c_j = value."""
    c = list(inst.c)
    c[j] = value
    return S.Instance(inst.A, inst.b, tuple(c), inst.senses, inst.names, inst.row_names, inst.kind)


def optimum(inst):
    """Optimalwert (nan, wenn unzulässig oder unbeschränkt)."""
    sol = A.solve(inst)
    return sol.obj if sol.status == "optimal" else float("nan")


def value_curve(inst, i, deltas):
    """Wertfunktion z*(b_i + delta) durch Neulösung je delta (nan: unzulässig); konkav und stückweise linear."""
    return [optimum(with_rhs(inst, i, d)) for d in deltas]


def one_sided_slopes(inst, i, eps=1e-6):
    """Links- und rechtsseitiger Differenzenquotient von z*(b_i) durch Neulösung mit Schritt eps mal max(1, |b_i|); nan, wenn die Seite unzulässig ist."""
    z0 = optimum(inst)
    h = eps * max(1.0, abs(inst.b[i]))
    right = (optimum(with_rhs(inst, i, h)) - z0) / h
    left = (z0 - optimum(with_rhs(inst, i, -h))) / h
    return left, right


def dual_face_ranges(inst):
    """Kleinster und größter Schattenpreis je Ressource über ALLE optimalen Dualen (die optimale Seitenfläche des Dual-LP: A^T y >= c, Vorzeichen, b·y = z*): 2 m LPs mit dem eigenen Löser. Bei eindeutigen
    Dualen ist die Spanne 0; bei Entartung ist sie das Intervall zwischen rechts- und linksseitiger Steigung (wo beide Seiten zulässig sind)."""
    sol = A.solve(inst)
    dual, decode = dual_instance(inst)
    m = inst.m
    # Spalten des Dual-LPs: obj = -b·y' je Spalte; die Zusatzzeile b·y = z* lautet sum(obj_col * y'_col) = -z*
    obj = np.array(dual.c)
    rows = [list(r) for r in dual.A] + [list(obj)]
    face = S.Instance(tuple(tuple(r) for r in rows), tuple(list(dual.b) + [-sol.obj]), dual.c, tuple(list(dual.senses) + [S.EQ]), dual.names, tuple(list(dual.row_names) + ["b·y = z*"]), "dual-face")
    # Zuordnung Spalte -> (Ressource, Vorzeichen) aus dem Aufbau von dual_instance
    plan = []
    for i in range(m):
        if inst.senses[i] == S.EQ:
            plan += [(i, 1.0), (i, -1.0)]
        elif inst.senses[i] == S.GE:
            plan.append((i, -1.0))
        else:
            plan.append((i, 1.0))
    out = []
    for i in range(m):
        e = np.array([sg if k == i else 0.0 for k, sg in plan])
        vals = []
        for direction in (1.0, -1.0):
            probe = S.Instance(face.A, face.b, tuple(direction * e), face.senses, face.names, face.row_names, "dual-face")
            r = A.solve(probe)
            vals.append(direction * r.obj if r.status == "optimal" else (direction * INF if r.status == "unbounded" else float("nan")))
        out.append((vals[1], vals[0]))                                                              # (min, max) von y_i; unbeschränkt: +-inf (z. B. Transport: alle Potenziale gemeinsam verschiebbar)
    return tuple(out)


def dual_unique(inst, tol=1e-6):
    """Sind die Schattenpreise eindeutig? Genau dann, wenn die optimale Dual-Seitenfläche ein Punkt ist (jede Spanne 0)."""
    for lo, hi in dual_face_ranges(inst):
        if math.isnan(lo) or math.isnan(hi):
            continue
        if math.isinf(lo) or math.isinf(hi) or hi - lo > tol * max(1.0, abs(hi), abs(lo)):
            return False
    return True


def dual_instance(inst):
    """Das Dual-LP als Instanz für denselben Löser: min b·y unter A^T y >= c mit y >= 0 (<=-Zeilen), y <= 0 (>=-Zeilen), y frei (=-Zeilen); als Maximierung von -b·y mit Ersetzungen y = y' bzw. -y' bzw. y+ - y-.
    Gibt (Instanz, decode) zurück; decode(x) liefert y je Ressource."""
    A_, b, c = inst.arrays()
    m, n = A_.shape
    cols, obj, plan = [], [], []
    for i in range(m):
        s = inst.senses[i]
        if s == S.LE:
            cols.append(A_[i]), obj.append(-b[i]), plan.append((i, 1.0))
        elif s == S.GE:
            cols.append(-A_[i]), obj.append(b[i]), plan.append((i, -1.0))
        else:
            cols.append(A_[i]), obj.append(-b[i]), plan.append((i, 1.0))
            cols.append(-A_[i]), obj.append(b[i]), plan.append((i, -1.0))
    M = np.array(cols).T                                               # n Zeilen (Dienste) x Spalten
    dual = S.Instance(tuple(tuple(float(v) for v in row) for row in M), tuple(float(v) for v in c), tuple(float(v) for v in obj), (S.GE,) * n, tuple(f"y{k}" for k in range(len(cols))),
                      tuple(f"d{j}" for j in range(n)), "dual")

    def decode(xd):
        y = [0.0] * m
        for (i, sg), v in zip(plan, xd):
            y[i] += sg * v
        return tuple(y)
    return dual, decode


def solve_dual(inst):
    """Dual-LP mit dem eigenen Löser: (Wert min b·y, y) oder None."""
    dual, decode = dual_instance(inst)
    sol = A.solve(dual)
    if sol.status != "optimal":
        return None
    return -sol.obj, decode(sol.x)


def with_purchase(inst, i, price):
    """Instanz mit zusätzlicher Spalte "Kapazität i zukaufen" (Verbrauch -1 in der <=-Zeile i, Deckungsbeitrag -price); nur für <=-Zeilen."""
    if inst.senses[i] != S.LE:
        raise ValueError("Zukauf nur für <=-Zeilen")
    A_ = [list(row) + [-1.0 if k == i else 0.0] for k, row in enumerate(inst.A)]
    return S.Instance(tuple(tuple(r) for r in A_), inst.b, tuple(list(inst.c) + [-float(price)]), inst.senses, tuple(list(inst.names) + ["Zukauf"]), inst.row_names, inst.kind)


def purchase_exact(inst, i, price):
    """Exakter Zukauf durch das LP mit Zusatzspalte: (Gesamtwert nach Zukaufskosten, gekaufte Menge) oder None (unbeschränkt)."""
    sol = A.solve(with_purchase(inst, i, price))
    if sol.status != "optimal":
        return None
    return sol.obj, sol.x[-1]


def purchase_steps(inst, i, price, max_steps=100):
    """Die Schattenpreis-Regel Schritt für Schritt: kaufe bis zur Bereichsgrenze, solange y_i > price; dann neu bepreisen. Gibt {"extra", "value", "steps": [(Menge, Schattenpreis)], "unbounded"} zurück;
    am Ende stimmt der Wert mit dem exakten LP überein (bis auf die Verschiebung um 1e-7 an den Grenzen)."""
    steps, extra, cur = [], 0.0, inst
    for _ in range(max_steps):
        sens = analyse_lp(cur)
        if sens is None:
            break
        y, hi = sens.y[i], sens.rhs_range[i][1]
        if y <= price + 1e-9:
            break
        if math.isinf(hi):
            return {"extra": INF, "value": INF, "steps": steps, "unbounded": True}
        delta = hi if hi > 1e-6 else 1e-6
        steps.append((delta, y))
        cur = with_rhs(cur, i, delta)
        extra += delta
    z = optimum(cur)
    return {"extra": extra, "value": z - price * extra, "steps": steps, "unbounded": False}


def value_pieces(inst, i, lo_b, hi_b, max_segments=40):
    """Wertfunktion z*(b_i) im Fenster [lo_b, hi_b] als Knickpunkte (b, z): von der aktuellen rechten Seite aus nach rechts und links entlang der Bereichsgrenzen der jeweiligen Endbasis (an den Grenzen wird um 1e-7
    verschoben, damit die nächste Basis gilt). Endet, wo die Instanz unzulässig wird oder das Fenster endet. Gibt aufsteigend sortierte Punkte zurück; zwischen ihnen ist z linear."""
    b0 = inst.b[i]
    sens0 = analyse_lp(inst)
    if sens0 is None:
        return []
    pts = {b0: sens0.obj}
    for direction in (1, -1):
        cur, b_cur, sens = inst, b0, sens0
        for _ in range(max_segments):
            if sens is None:
                break
            y = sens.y[i]
            limit = sens.rhs_range[i][1] if direction == 1 else -sens.rhs_range[i][0]
            z_cur = sens.obj
            edge = hi_b if direction == 1 else lo_b
            room = (edge - b_cur) * direction
            if room <= 1e-9:
                break
            if math.isinf(limit) or limit >= room:
                pts[edge] = z_cur + y * (edge - b_cur)
                break
            target = b_cur + direction * limit
            pts[target] = z_cur + y * (target - b_cur)
            step = direction * (limit + 1e-7 * max(1.0, abs(target)))
            cur = with_rhs(cur, i, step)
            b_cur = cur.b[i]
            sens = analyse_lp(cur)
    return sorted(pts.items())
