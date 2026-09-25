"""Konstanten der Demo Dualität und Sensitivität: Regler-Bereiche, feste Instanzen, Optionen (Werte aus der Vormessung), Presets."""
M_MIN, M_MAX, DEFAULT_M = 2, 12, 6
N_MIN, N_MAX, DEFAULT_N = 2, 12, 8
K_MIN, K_MAX, DEFAULT_K = 2, 5, 3                    # Lager im Transportproblem
L_MIN, L_MAX, DEFAULT_L = 2, 8, 5                    # Kunden im Transportproblem
DENSITY = 0.5                                        # Dichte der Zufalls- und Mischinstanzen (fest)
SEED_MAX = 999999
DEFAULT_SEED = 35
INDEX_MAX = 11                                       # Ressourcen- bzw. Dienst-Auswahl (wird auf die Instanzgröße begrenzt)
DELTA_OPTIONS = tuple(range(-50, 105, 5))           # Änderung der rechten Seite in Prozent
PRICE_OPTIONS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.5)     # Zukaufspreis je Einheit Kapazität
DEFAULT_PRICE = 2.0
STEPS = {1: "1 · Die Duale", 2: "2 · Schattenpreis", 3: "3 · Ranging", 4: "4 · Entartung"}
SWEEP_SEEDS = tuple(range(100000, 100005))
NOISE_SEEDS = tuple(range(200000, 200050))
NOISE_SIGMAS = (0.01, 0.05, 0.1)
PREDICTION_FACTORS = (0.5, 1.0, 2.0, 4.0)            # Erweiterung als Vielfaches der Bereichsgrenze
_BASE = {"kind": "centre", "m": 6, "n": 8, "k": 3, "l": 5, "seed": 35, "res": 0, "var": 0, "price": 2.0, "step": 1}
PRESETS = {
    "Lehrbuch: Duale von Hand": {**_BASE, "kind": "textbook"},
    "Zentrum: drei Preise, eine freie Ressource": dict(_BASE),
    "Mischung: Vorzeichen der Dualen": {**_BASE, "kind": "mixed", "m": 8, "n": 8},
    "Schattenpreis gilt nur im Bereich": {**_BASE, "res": 1, "step": 2, "delta": 75},
    "Zukauf: Kommissionierstunden": {**_BASE, "res": 0, "step": 2, "price": 2.0},
    "Ranging der Rampenzeit": {**_BASE, "res": 2, "step": 3},
    "Sperrgut lohnt ab 23.5": {**_BASE, "var": 3, "step": 3},
    "Kostenrauschen kippt die Basis": {**_BASE, "kind": "random", "m": 10, "n": 10, "step": 3},
    "Entartete Ecke: links ungleich rechts": {**_BASE, "kind": "degenerate", "step": 4},
    "Transport: Duale nicht eindeutig": {**_BASE, "kind": "transport", "step": 4},
}
PRESET_HELP = {
    "Lehrbuch: Duale von Hand": "Optimum 36 bei (2, 6), Schattenpreise y = (0, 1.5, 1): die Rampenzeit hat Schlupf 2 und den Preis 0, Kommissionierstunden und Lagerfläche binden. Starke Dualität: c·x = 36 = b·y (12·1.5 + 18·1). Die Duale sind eindeutig.",
    "Zentrum: drei Preise, eine freie Ressource": "Distributionszentrum: Optimum 720 bei x = (20, 10, 10, 0, 0), Schattenpreise y = (5, 0.5, 4, 0). Kommissionierstunden, Lagerfläche und Rampenzeit binden mit sehr verschiedenem Preis, die Fahrzeugkapazität hat Schlupf 160. Sperrgut und Retouren lohnen nicht (reduzierte Kosten 3.5).",
    "Mischung: Vorzeichen der Dualen": "Mischinstanz mit 8 Ressourcen und 8 Diensten (Seed 35): 4 bindende Ressourcen, drei ≤-Ressourcen mit den Preisen 2.73, 2.36 und 0.37 und eine ≥-Ressource mit dem Preis −1.29 (mehr Mindestmenge kostet). c·x = b·y = 320.67.",
    "Schattenpreis gilt nur im Bereich": "Lagerfläche im Zentrum (y = 0.5) gilt für b zwischen 172 und 217.5. Bei +75 % (Δ = 150) sagt y·Δ 75 voraus, die Neulösung findet 8.75: jenseits von 17.5 zusätzlichen Einheiten bringt Fläche nichts mehr.",
    "Zukauf: Kommissionierstunden": "Zukauf zum Preis 2: die Schattenpreis-Regel kauft in 3 Schritten 135 Einheiten (Schattenpreis 5, 4.125, 2.833), Gewinn nach Zukaufskosten 160, Optimalwert 880 wie im exakten LP mit Zukaufsspalte. Die naive Rechnung (y − Preis) mal Menge verspricht 405.",
    "Ranging der Rampenzeit": "Rampenzeit (y = 4): Bereich von 52.31 bis 58.89 (−2.69 bis +3.89). Knapp über der oberen Grenze wechselt die Basis und der Schattenpreis fällt auf 0, knapp unter der unteren steigt er auf 7.25.",
    "Sperrgut lohnt ab 23.5": "Sperrgut (c = 20, Menge 0, reduzierte Kosten 3.5): die Endbasis bleibt bis c = 23.5 optimal; bei 23.52 werden 7.78 Einheiten produziert und der Optimalwert steigt auf 720.18.",
    "Kostenrauschen kippt die Basis": "Zufallsinstanzen 10 × 10: bei 1 %, 5 % und 10 % multiplikativem Rauschen auf den Deckungsbeiträgen wechselt die optimale Basis in 4 %, 26 % und 40 % der 50 Läufe (Zentrum: 0 %, 50 %, 82 %). Auf Abruf im Schritt.",
    "Entartete Ecke: links ungleich rechts": "Das Lehrbuchbeispiel mit einer vierten Ressource durch das Optimum (2, 6): der Löser wählt y = (0, 1.5, 1, 0), optimal sind aber alle Duale mit y₂ ∈ [1, 1.5], y₃ ∈ [0, 1], y₄ ∈ [0, 3]. Links- und rechtsseitige Steigung der Kommissionierstunden: 1.5 und 1.0.",
    "Transport: Duale nicht eindeutig": "Transportproblem mit 3 Lagern und 5 Kunden: die Endbasis ist entartet, und die optimale Dual-Seitenfläche ist unbeschränkt (alle Potenziale lassen sich gemeinsam verschieben, weil Angebot gleich Nachfrage ist). In 50 Instanzen 3 × 5 und 4 × 8 je 100 %, bei Zufall 10 × 10 0 %, bei Mischung 10 × 10 4 %.",
}
