# Settling-Tool
Dieser Code kann den Asymmetrie-Koaleszenzparameter an experimentelle Daten von Henschke (in Excel-Files abgespeichert) fitten. Es besteht auch die Möglichkeit, das vereinfachte Koaleszenzmodell zu verwenden, um den Höhen-Koaleszenzparameter zu fitten.  Das Tool ist objektorientiert von Lukas Thiel aufgebaut worden und wurde im Rahmen dieser BA angepasst, um das vereinfachte Koaleszenzmodell anwenden zu können. Zum Ausführen ist die Datei sim_model zu verwenden (in allExamples sind beispielhafte Anweisungen aufgeführt, welche in sim_model hineinkopiert werden können). Messwerte von Henschke und Stoffdaten werden aus den Excel-Dateien im Input-Ordner ausgelesen. Hier ein kurzer Überblick über die Aufgaben der Dateien:
- "run.py" (auszuführende Datei)
- "model.py" (beinhaltet die Berechnungsvorschriften und Plotting-Funktionen)
- "settings.py" (beinhaltet Parameter zur Berechnung und liest Messwerte sowie Stoffdaten aus Excel-Dateien im Input-Ordner 		aus)
- "helper_functions.py" (beinhaltet Hilfsfunktionen zur Berechnung)
- "allExamples.py" (beinhaltet beispielhafte Anweisungen für die ausführende Datei)
