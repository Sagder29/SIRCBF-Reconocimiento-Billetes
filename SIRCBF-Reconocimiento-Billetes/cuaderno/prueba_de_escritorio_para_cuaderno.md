# Prueba de escritorio manuscrita — K-NN

**Tema:** Clasificación de especímenes mediante K-NN.  
**Propósito:** comprobar manualmente la función de distancia y la votación.  
**Nota:** estos son datos didácticos pequeños creados para verificar el algoritmo; no son registros reales del dataset UCI.

## Datos de entrenamiento de ejemplo

| Registro | x1 | x2 | Clase |
|---|---:|---:|---:|
| A | 0 | 0 | 0 |
| B | 1 | 0 | 0 |
| C | 0 | 1 | 0 |
| D | 5 | 5 | 1 |
| E | 6 | 5 | 1 |

La distancia euclidiana para dos características es:

\[
d(X,Y)=\sqrt{(x_1-y_1)^2+(x_2-y_2)^2}
\]

## Prueba 1 — K=1

Nuevo registro: X=(0,2; 0,2), donde la coma decimal indica 0.2.

- A: \(d=\sqrt{(0.2-0)^2+(0.2-0)^2}=\sqrt{0.08}=0.2828\)
- B: \(d=\sqrt{(0.2-1)^2+(0.2-0)^2}=\sqrt{0.68}=0.8246\)
- C: \(d=\sqrt{(0.2-0)^2+(0.2-1)^2}=\sqrt{0.68}=0.8246\)

El vecino más cercano es A, cuya etiqueta es Clase 0.

**Resultado esperado:** Clase 0.

## Prueba 2 — K=3

Nuevo registro: X=(5.5; 5.2).

- D: \(d=\sqrt{(5.5-5)^2+(5.2-5)^2}=\sqrt{0.29}=0.5385\)
- E: \(d=\sqrt{(5.5-6)^2+(5.2-5)^2}=\sqrt{0.29}=0.5385\)
- B: \(d=\sqrt{(5.5-1)^2+(5.2-0)^2}=\sqrt{47.29}=6.8768\)

Los tres vecinos más cercanos son D (Clase 1), E (Clase 1) y B (Clase 0).

Votación: Clase 1 = 2 votos; Clase 0 = 1 voto.

**Resultado esperado:** Clase 1.

## Prueba 3 — Empate con K=2

Datos: P=(0;0), Clase 0; Q=(2;0), Clase 1. Nuevo registro X=(1;0).

- Distancia a P: \(d=\sqrt{(1-0)^2+(0-0)^2}=1\)
- Distancia a Q: \(d=\sqrt{(1-2)^2+(0-0)^2}=1\)

Cada clase recibe un voto. Para que el programa sea reproducible se definió una regla de desempate: gana la etiqueta numéricamente menor.

**Resultado esperado:** Clase 0.

## Conclusión para completar después de ejecutar el programa

Las tres pruebas de escritorio deben coincidir con las tres pruebas unitarias ejecutadas en Python. Si algún resultado no coincide, se debe revisar la fórmula, la selección de vecinos o la regla de votación antes de continuar.
