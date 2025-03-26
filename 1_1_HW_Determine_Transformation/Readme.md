## Berechnung von Transformationsparametern zwischen Punktwolken mit Rauschen

Das Ziel ist es, die Parameter einer Transformation `R` zu schätzen, die eine gegebene Menge von 2D-Punkten (Originalpunkte `Pi`) so transformiert, dass sie sich so gut wie möglich an eine korrespondierende Menge von 2D-Punkten (Zielpunkte `P'i`) annähern.

### 1. Mathematische Modellierung der Transformation:

Die Transformation eines Originalpunktes `Pi` mit Koordinaten (x, y) in einen transformierten Punkt mit Koordinaten (x', y') wird durch folgende Gleichungen beschrieben:

```
a = s * cos(0)
b = s * sin(0)

    | a  -b  Tx |
T = | b   a  Ty |
    | 0   0   1 |

x' = a * x - b * y + Tx
y' = b * x + a * y + Ty
```

Dabei sind:
*   (x, y): die Koordinaten des Originalpunktes `Pi`.
*   (x', y'): die Koordinaten des transformierten Punktes `P'i`.
*   `a` und `b`: Parameter, die die Kombination aus Rotation und Skalierung ist.
*   `Tx`: die Translation in x-Richtung.
*   `Ty`: die Translation in y-Richtung.

Diese Gleichungen definieren, wie jeder Punkt `Pi` durch Rotation, Skalierung und Translation in einen transformierten Punkt überführt wird. Das Ziel ist es, `a`, `b`, `Tx` und `Ty` so zu bestimmen, dass die transformierten Punkte möglichst nahe an den Zielpunkten `P'i` liegen.

### 2. Aufstellung eines linearen Gleichungssystems:

Die Transformationsgleichungen werden in ein lineares Gleichungssystem überführt, um die Lösung mit der Methode der kleinsten Quadrate zu ermöglichen.

Dieses Gleichungssystem hat die Form `Ax = b`, wobei:
*   `x`: Ein Vektor, der die zu schätzenden Transformationsparameter enthält: `x = [a, b, Tx, Ty]`.
*   `A`: Eine Koeffizientenmatrix, die aus den Koordinaten der Originalpunkte `Pi` aufgebaut ist. Für jeden Punkt `Pi` werden zwei Zeilen in `A` erzeugt. Die Form von `A` ist so aufgebaut, dass die erste Spalte mit dem Parameter `a`, die zweite Spalte mit dem Parameter `b`, die dritte mit dem Parameter `Tx` und die vierte Spalte mit dem Parameter `Ty` multipliziert wird.
*   `b`: Ein Vektor, der die Koordinaten der Zielpunkte `P'i` enthält.


### 3.  Lösen des linearen Gleichungssystems mit der Methode der kleinsten Quadrate:

Das überbestimmte lineare Gleichungssystem `Ax = b` wird mit der Methode der kleinsten Quadrate gelöst.

```python
x, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
```

Die Lösung liefert einen Vektor `x` mit den optimalen Parametern `a`, `b`, `Tx`, und `Ty`.

### 4.  Extrahieren der Transformationsparameter:

Der gelöste Vektor `x` liefert die optimalen Parameter:
*   `a`: Der erste Eintrag in `x` kombiniert Rotation und Skalierung.
*   `b`: Der zweite Eintrag in `x` kombiniert ebenfalls Rotation und Skalierung.
*   `Tx`: Der dritte Eintrag in `x` stellt die Translation in x-Richtung dar.
*   `Ty`: Der vierte Eintrag in `x` stellt die Translation in y-Richtung dar.

### 5.  Berechnung der endgültigen Transformationsparameter:

Die zuvor berechneten Parameter `a` und `b` werden in die endgültigen Transformationsparameter, Skalierung `s` und Rotation `rot`, umgewandelt.
*   Die Skalierung `s` wird aus `a` und `b` mit folgender Formel berechnet: `s = sqrt(a^2 + b^2)`.
*   Die Rotation `rot` wird aus `a` und `b` mit folgender Formel berechnet: `rot = arctan2(b, a)`. Die Rotation `rot` wird in Bogenmaß ausgegeben und kann in Grad umgerechnet werden.

Die Parameter `Tx` und `Ty` entsprechen direkt den Translationen in x- und y-Richtung.


### 6. Ergebnis

```
Optimierte Parameter:
  Rotation: -10.8741 Grad
  Skalierung: 0.5489
  Translation Tx: -2.2145
  Translation Ty: 1.1295
  Geschätztes Rauschen: 0.0893
```
