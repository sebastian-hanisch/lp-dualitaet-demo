# Dualität und Sensitivität – was ein Schattenpreis sagt und wie lange er gilt – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lp-dualitaet-demo.streamlit.app/)**

Fünftes Stück der **Lineare-Programmierung-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind der Wurzel [tableau-simplex-demo](https://github.com/sebastian-hanisch/tableau-simplex-demo). Bisher waren die Duale y nur ein Nebenprodukt der Endbasis (Stück 1: aus der Zielzeile, Stück 4: Ergebnis von BTRAN). Hier sind sie das Thema: **y_i ist der Wert einer zusätzlichen Einheit der Ressource i**, also das, was ein Distributionszentrum für eine Stunde mehr Kommissionierung oder einen Quadratmeter mehr Lagerfläche höchstens zahlen dürfte. Die Demo rechnet Duale, reduzierte Kosten und **Bereiche** (Ranging) aus dem Endtableau, prüft sie gegen Neulösungen und gegen HiGHS und zeigt, wo die Schattenpreis-Rechnung endet. Vier Fragen, alle gemessen: **(1) Die Duale** – stimmen starke Dualität und komplementärer Schlupf, und was heißt y geometrisch? **(2) Schattenpreis** – y ist die Steigung des Optimalwerts über die rechte Seite, aber wie weit trägt sie, und was passiert beim Zukauf von Kapazität? **(3) Ranging** – in welchem Bereich bleibt die Endbasis optimal, und wie oft kippt sie schon bei wenig Rauschen? **(4) Entartung** – wann sind die Duale nicht eindeutig?

**Einordnung in die Reihe:** die Reihe hat elf Stücke, dies ist das fünfte (Details in `lp-planung/PLAN.md` des Portfolio-Ordners):

```
Tableau-Simplex (Wurzel)                                                                  [gebaut: tableau-simplex-demo]
 ├─ Pivotregeln & Entartung ─ Simplex im schlimmsten und im typischen Fall (Klee-Minty)   [gebaut: pivotregeln-demo, klee-minty-demo]
 ├─ Revised Simplex ─ Präsolve, Skalierung & Numerik                                     [gebaut: revised-simplex-demo]  →  [nicht gebaut]
 ├─ Dualität & Sensitivität ─ Dualer Simplex & Neuoptimierung                            [DIESES STÜCK]  →  [nicht gebaut]
 ├─ Ellipsoid-Methode (Kontrast: polynomial in der Theorie)                              [nicht gebaut]
 └─ Innere Punkte ─ PDLP (Verfahren erster Ordnung) ─ Crossover & Simplex gegen Innere Punkte gegen PDLP  [nicht gebaut]
```

Ergebnis in Kürze: **Ein Schattenpreis ist eine Steigung, keine Zahl, die man beliebig weit hochrechnen darf, und bei Entartung ist er nicht einmal eindeutig.** Im Distributionszentrum (drei bindende Ressourcen mit den Preisen 5, 0.5 und 4, eine mit Schlupf 160 und Preis 0) sagt der Preis der Lagerfläche (0.5) den Gewinn einer Erweiterung **exakt bis 17.5 Einheiten** voraus; bei dem Vierfachen der Grenze verspricht y·Δ 35, die Neulösung findet 8.75. Die **naive Zukaufsrechnung** (y − Preis) mal Menge verspricht bei den Kommissionierstunden zum Preis 2 einen Gewinn von 405, tatsächlich sind es 160; die Regel "kaufe bis zur Bereichsgrenze und bepreise neu" trifft das exakte LP mit Zukaufsspalte (880). **Bereiche sind eng**: im Zentrum im Median +8.8 % und −12.5 % des Bestands, und **5 % Rauschen** auf den Deckungsbeiträgen wechselt die optimale Basis im Zentrum in 50 % und bei Zufall 10 × 10 in 26 % der Läufe. **Entartung:** Zufallsinstanzen sind nie entartet (0 %), Mischinstanzen fast nie (0 % bis 4 %), Transportprobleme **immer**, und dort sind die Duale nicht eindeutig (unbeschränkte Dual-Seitenfläche: alle Potenziale lassen sich gemeinsam verschieben).

| Frage | Ergebnis (Distributionszentrum, Standard-LP max c·x; Kreuzprüfung mit HiGHS und Neulösungen; Median über 5 feste Instanzen, Seeds 100000–100004, Rauschen und Entartungsanteile über 50 Instanzen, Seeds 200000–200049; vollständig deterministisch) |
|---|---|
| **Stimmen die Zertifikate?** | ✅ Auf 303 Instanzen (Zufall, Mischung mit ≥ und =, Transport, Zentrum, Lehrbuch, entartete Ecke): Optimum gleich HiGHS, primal und dual zulässig, **starke Dualität c·x = b·y**, komplementärer Schlupf (y_i · Schlupf_i = 0 und x_j · r_j = 0) jeweils bis 1e-7; bei nicht entarteter Endbasis sind die Duale gleich den HiGHS-Marginals. Das Dual-LP (min b·y, Aᵀy ≥ c) liefert mit dem eigenen Löser und mit HiGHS denselben Wert. Vorzeichen: ≤-Zeilen y ≥ 0, ≥-Zeilen y ≤ 0, =-Zeilen frei |
| **Lehrbuch von Hand** | Optimum 36 bei (2, 6), y = (0, 1.5, 1) (die Rampenzeit hat Schlupf 2), c·x = b·y = 12·1.5 + 18·1 = 36. Bereiche der rechten Seiten: Kommissionierstunden b ∈ [6, 18], Lagerfläche b ∈ [12, 24], Rampenzeit b ≥ 2; Deckungsbeiträge: c₁ ∈ [0, 7.5], c₂ ∈ [2, ∞) |
| **Zentrum (rückwärts konstruiert)** | Optimum 720 bei x = (20, 10, 10, 0, 0), y = (5, 0.5, 4, 0), Schlupf der Fahrzeugkapazität 160; Sperrgut und Retouren haben reduzierte Kosten 3.5. Bereiche: Kommissionierstunden [70, 90], Lagerfläche [172, 217.5], Rampenzeit [52.31, 58.89]; Kostenbereiche Express [14.8, 16.28], Palettenversand [13.25, 19.67], Kühlware [24.92, 26.88] |
| **Schattenpreis = Ableitung** | Im Bereich ist der Differenzenquotient der Neulösung gleich y (eigener Löser und HiGHS, 1e-6); knapp jenseits einer endlichen Grenze ist die Steigung eine andere. An 99.9 % der Grenze bleibt die Basis gleich, an 100.1 % wechselt sie (Rampenzeit: oben fällt y auf 0, unten steigt es auf 7.25). Die Wertfunktion ist konkav und stückweise linear (zweite Differenzen ≤ 1e-7), in c konvex |
| **Vorhersage y·Δ gegen Neulösung** | Lagerfläche (y = 0.5), Δ = 0.5 / 1 / 2 / 4 mal die Grenze 17.5: Vorhersage 4.4 / 8.8 / 17.5 / 35, Neulösung 4.4 / 8.8 / **8.75 / 8.75**: exakt bis zur Grenze, danach bringt zusätzliche Fläche nichts mehr. Kommissionierstunden (Grenze 10): 25 / 50 / 100 / 200 gegen 25 / 50 / 91.2 / 160.8 |
| **Zukauf** | Kommissionierstunden, Preis 2: in **3 Schritten 135 Einheiten** (Schattenpreis 5 → 4.125 → 2.833), Gewinn nach Zukaufskosten **160**, Optimalwert 880 – gleich dem exakten LP mit Zukaufsspalte. Die naive Rechnung (y − Preis) · Menge = 3 · 135 verspricht **405**. Zum Preis 3 kauft die Regel 30 Einheiten (2 Schritte) und gewinnt 42.5 statt der naiv erwarteten 60. Bei Preis ≥ y wird nichts gekauft |
| **Bereichsbreiten** | Median über bindende Ressourcen (Anteil des Bestands, Erhöhung / Senkung) und Basis-Dienste (Kostenbereich als Anteil von c): Zentrum **8.8 % / 12.5 %** und 9.5 %; Lehrbuch 41.7 % / 41.7 % und 250 %; Zufall 6 × 8 50 % / 42 % und 68 %; Zufall 10 × 10 16 % / 37 % und 76 %; Mischung 8 × 8 47 % / 48 % und 163 %; Transport: Erhöhung 0 % (entartet) |
| **Kostenrauschen kippt die Basis** | Anteil der 50 Läufe, in denen multiplikatives Rauschen der Stärke 1 % / 5 % / 10 % auf den Deckungsbeiträgen die optimale Basis ändert: Zentrum **0 % / 50 % / 82 %**, Zufall 10 × 10 **4 % / 26 % / 40 %**, Zufall 6 × 8 4 % / 14 % / 22 %, Mischung 8 × 8 6 % / 16 % / 22 %, Transport 3 × 5 14 % / 52 % / 76 %, Lehrbuch 0 % / 0 % / 0 % |
| **Entartung, Duale nicht eindeutig** | Entartete Ecke (Lehrbuch mit vierter Ressource durch das Optimum): der Löser wählt y = (0, 1.5, 1, 0), optimal sind alle Duale mit y₂ ∈ [1, 1.5], y₃ ∈ [0, 1], y₄ ∈ [0, 3]; links- und rechtsseitige Steigung der Kommissionierstunden **1.5 und 1.0**. Anteil der Instanzen mit entarteter Endbasis und nicht eindeutigen Dualen: Zufall 6 × 8 und 10 × 10 **0 % / 0 %**, Mischung 8 × 8 0 %, Mischung 10 × 10 **4 % / 4 %**, Transport 3 × 5 und 4 × 8 **100 % / 100 %** – nie eine nicht entartete Endbasis mit mehrdeutigen Dualen |

## Was die Demo zeigt

1. **Vier Schritte** (Schritt-Slider): **Die Duale** (Tabellen der Ressourcen mit Bestand, Nutzung, Schlupf, Schattenpreis und y·Schlupf sowie der Dienste mit Menge, Deckungsbeitrag, reduzierten Kosten; die Meldung "starke Dualität: c·x = b·y"; bei zwei Diensten die Zeichnung mit zulässiger Menge, bindenden Ressourcen und dem Zielvektor c als Summe der mit y gewichteten Normalen; sonst Balken der Schattenpreise) → **Schattenpreis** (Wertfunktion z*(b_i) durch Neulösung mit exakten Knickpunkten, Tangente, Bereichsband, Regler für die Änderung in Prozent, Tabelle Vorhersage gegen Neulösung, Zukauf mit Preisregler, Treppe des Grenznutzens) → **Ranging** (Balken und Tabellen der Bereiche für rechte Seiten und Deckungsbeiträge, "Grenze prüfen" mit Neulösung knapp innerhalb und außerhalb, Basiskipp unter Kostenrauschen auf Abruf) → **Entartung** (optimale Dual-Seitenfläche je Ressource, links-/rechtsseitige Steigung, Anteile je Instanzart auf Abruf).
2. **Kennzahlen der Instanz:** Optimalwert, bindende Ressourcen, größter Schattenpreis, entartet oder nicht.
3. **Instanzen:** Lehrbuchbeispiel, Zentrum (5 Dienste, 4 Ressourcen), entartete Ecke, Zufall, Mischung (≥ und =), Transport, dazu Unzulässig und Unbeschränkt (ohne Analyse, mit Erklärung).

Presets (10): Lehrbuch: Duale von Hand, Zentrum: drei Preise, eine freie Ressource, Mischung: Vorzeichen der Dualen, Schattenpreis gilt nur im Bereich, Zukauf: Kommissionierstunden, Ranging der Rampenzeit, Sperrgut lohnt ab 23.5, Kostenrauschen kippt die Basis, Entartete Ecke: links ungleich rechts, Transport: Duale nicht eindeutig.

## Messwerte der Presets

| Preset | Einstellungen | Ergebnis |
|---|---|---|
| **Lehrbuch: Duale von Hand** | 2 Dienste | Optimum 36, y = (0, 1.5, 1), c·x = b·y |
| **Zentrum: drei Preise, eine freie Ressource** | 5 Dienste, 4 Ressourcen | 720, y = (5, 0.5, 4, 0), Schlupf 160 der Fahrzeugkapazität |
| **Mischung: Vorzeichen der Dualen** | 8 × 8, Seed 35 | 4 bindende Ressourcen: 2.73, −1.29 (≥-Zeile), 2.36, 0.37; Optimum 320.67 |
| **Schattenpreis gilt nur im Bereich** | Lagerfläche, +75 % | y = 0.5 gilt für b ∈ [172, 217.5]; y·Δ = 75 gegen Neulösung 8.75 |
| **Zukauf: Kommissionierstunden** | Preis 2 | 3 Schritte, 135 Einheiten, Gewinn 160, Wert 880, naiv 405 |
| **Ranging der Rampenzeit** | y = 4 | Bereich [52.31, 58.89] (−2.69 bis +3.89); oberhalb y = 0, unterhalb y = 7.25 |
| **Sperrgut lohnt ab 23.5** | Sperrgut, c = 20 | bis c = 23.5 unverändert; bei 23.52 werden 7.78 Einheiten produziert, Optimalwert 720.18 |
| **Kostenrauschen kippt die Basis** | Zufall 10 × 10 | Basiswechsel in 4 % / 26 % / 40 % (Zentrum: 0 % / 50 % / 82 %) |
| **Entartete Ecke: links ungleich rechts** | 2 Dienste, 4 Ressourcen | y₂ ∈ [1, 1.5], y₃ ∈ [0, 1], y₄ ∈ [0, 3]; Steigungen 1.5 und 1.0 |
| **Transport: Duale nicht eindeutig** | 3 Lager, 5 Kunden | entartet, Dual-Seitenfläche unbeschränkt; 100 % der Instanzen 3 × 5 und 4 × 8 |

## Modell und Verfahren

- **Instanz** (`dua_scenario.py`): die Auslastungsplanung der Vorgängerstücke, dazu das **Zentrum** (rückwärts konstruiert: gewünschtes Optimum und gewünschte Schattenpreise vorgegeben, Deckungsbeiträge daraus berechnet, zwei Dienste bewusst knapp unrentabel) und die **entartete Ecke**.
- **Löser** (`dua_algorithm.py`): der dichte Tableau-Simplex der Stücke 1 bis 4 mit Zwei-Phasen-Start, Dantzig-Regel und Bland als Notbremse gegen Zyklen; das Ergebnis enthält das Endtableau. Die Anfangsbasis besteht aus Einheitsspalten, also steht B⁻¹ im Endtableau unter diesen Spalten und y = c_B B⁻¹ in der Zielzeile.
- **Sensitivität** (`dua_sensitivity.py`): Schattenpreise je Zeilenart, Schlupf, reduzierte Kosten; **RHS-Ranging** aus x_B + δ·B⁻¹e_i ≥ 0 (mit Vorzeichen bei gespiegelten Zeilen; eine künstliche Basisvariable in einer redundanten Gleichung setzt den Bereich auf 0), **Kosten-Ranging** (Nichtbasisdienst: c_j ≤ y·a_j; Basisdienst: r_k + δ·α_pk ≥ 0 für alle Nichtbasisspalten), **Wertfunktion** durch Neulösung entlang der Bereichsgrenzen (Verschiebung um 1e-7 an einer Grenze), **Dual-LP** als Instanz für denselben Löser (Vorzeichenbeschränkungen über Ersetzungen), **Dual-Seitenfläche** (2 m LPs über {Aᵀy ≥ c, b·y = z*}: kleinster und größter Schattenpreis je Ressource, unbeschränkt möglich), **Zukauf** als LP mit Zusatzspalte und als Schattenpreis-Regel in Schritten.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Hypothese "Transportprobleme haben nicht eindeutige Duale" – bestätigt, aber meine erste Messung hat es verfehlt.** Über links- und rechtsseitige Differenzenquotienten sah ich bei Transport 0 % nicht eindeutige Duale: eine Seite ist bei ausgeglichenem Angebot und Nachfrage unzulässig, der Vergleich entfällt. Erst die optimale Dual-Seitenfläche zeigt es: sie ist unbeschränkt (alle Potenziale gemeinsam verschiebbar), in 100 % der Instanzen. Die Demo rechnet deshalb über die Seitenfläche.
- **Vorab-Hypothese "Entartung kommt im Zufall selten vor" – bestätigt** (0 % bei Zufall, 0 % bis 4 % bei Mischung); bei Transport ist sie die Regel. Nicht entartete Endbasen hatten in allen Läufen eindeutige Duale (Test).
- **Ranging gilt für einen Parameter allein.** Gleichzeitige Änderungen (die 100-%-Regel) und parametrische Programmierung über mehrere Parameter sind nicht gebaut.
- **Zukauf nur für ≤-Ressourcen.** Bei ≥- und =-Ressourcen gibt es keine "Kapazität"; die Demo weist darauf hin.
- **Unzulässige und unbeschränkte Instanzen haben kein optimales Dual** (nur ein Zertifikat); die Demo zeigt dann keine Analyse.
- **Neuoptimierung ohne Warmstart.** Die Wertfunktion und die Grenzprüfungen lösen jeden Punkt neu; der Duale Simplex, der nach einer Änderung von der alten Basis aus weiterrechnet, ist das Folgestück.
- **Synthetische, kleine Instanzen** (bis 12 Ressourcen und 12 Dienste; Transport bis 5 × 8); Schattenpreise in echten Netzen sind viel schwerer zu deuten (Entartung, Nebenwirkungen, ganzzahlige Entscheidungen).
- **Schattenpreise sagen nur Grenzwerte, keine Gesamteffekte.** Das Zentrum zeigt es: erst die Regel mit Neubepreisung trifft den optimalen Zukauf.

## Verifikation

- `tests/test_algorithm.py`: Zertifikate auf 303 Instanzen (Optimum gegen HiGHS, starke Dualität, dual zulässig, komplementärer Schlupf beider Arten, Duale gleich HiGHS-Marginals bei nicht entarteter Endbasis); Dual-LP mit eigenem Löser und HiGHS; **Lehrbuch und Zentrum von Hand** (Duale, Schlupf, reduzierte Kosten, Bereiche); **Schattenpreis gleich Ableitung** im Bereich (eigene Neulösung und HiGHS) und ungleich jenseits der Grenze; **Bereichsgrenzen gegen Neulösung** (99.9 % gleiche Basis, 100.1 % andere); Konkavität in b und Konvexität in c; Kostenbereiche (Lösung unverändert innerhalb, geändert außerhalb; Nichtbasisdienst tritt genau über der Grenze ein); Vorzeichen bei ≤, ≥, =; Zeilenart-gemischte Instanzen; redundante Gleichung; **Dual-Seitenfläche gegen HiGHS** (über 100 Grenzen) und gegen die einseitigen Steigungen der entarteten Ecke; Zukauf (Regel in Schritten gleich exaktem LP, nichts bei Preis ≥ y, unbeschränkt erkannt).
- `tests/test_scenario.py`, `test_evaluation.py`, `test_presets.py` (jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz, jede Ressourcen- und Dienst-Auswahl, δ-Randwerte, Zukauf, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer).
- Für die Prüfung genügt **pytest**; `scipy` dient nur als Gegenprobe (`requirements-dev.txt`), die App braucht nur numpy.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Gale, D., Kuhn, H. W., & Tucker, A. W. (1951). *Linear programming and the theory of games.* In T. C. Koopmans (Hrsg.), Activity Analysis of Production and Allocation, 317–329. Wiley.
- Dantzig, G. B. (1963). *Linear Programming and Extensions.* Princeton University Press.
- Gal, T. (1979). *Postoptimal Analyses, Parametric Programming, and Related Topics.* McGraw-Hill (nur genannt).

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
