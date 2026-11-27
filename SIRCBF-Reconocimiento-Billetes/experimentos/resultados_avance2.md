# Resultados del Avance 2 – Proyecto SIRCBF

## 1. Introducción

En esta etapa se desarrolló la preparación de los datos y la primera versión del algoritmo de clasificación K-Nearest Neighbors (K-NN), implementado desde cero en Python.

El objetivo fue analizar el conjunto de datos Banknote Authentication, construir un modelo de referencia (baseline), implementar el algoritmo K-NN y verificar su funcionamiento mediante pruebas unitarias.

## 2. Control de calidad de los datos

Se trabajó con el conjunto de datos del UCI Machine Learning Repository, compuesto por 1.372 registros y cuatro variables predictoras.

Los resultados del control de calidad fueron:

| Indicador | Resultado |
|---|---:|
| Registros originales | 1.372 |
| Valores faltantes | 0 |
| Valores infinitos | 0 |
| Filas duplicadas conservadas | 24 |
| Grupos únicos de características | 1.348 |
| Grupos con características repetidas | 11 |
| Filas adicionales en grupos repetidos | 24 |

Los registros duplicados fueron conservados. Para evitar filtración de información, las muestras con las mismas cuatro características se asignaron al mismo subconjunto.

## 3. Análisis exploratorio

Se analizaron las cuatro variables predictoras:

- Variance (varianza).
- Skewness (asimetría).
- Curtosis.
- Entropy (entropía).

La distribución de las clases fue la siguiente:

| Clase | Cantidad | Porcentaje |
|---|---:|---:|
| Clase 0 | 762 | 55,54 % |
| Clase 1 | 610 | 44,46 % |
| **Total** | **1.372** | **100 %** |

También se calcularon estadísticas descriptivas y correlaciones entre las variables. Se generaron histogramas, una gráfica del balance de clases y una matriz de correlaciones.

## 4. División de los datos

Se utilizó una división por grupos de características, con semilla fija 42.

| Subconjunto | Cantidad |
|---|---:|
| Entrenamiento | 961 |
| Validación | 205 |
| Prueba | 206 |
| **Total** | **1.372** |

Se verificó que no existiera solapamiento de grupos entre entrenamiento, validación y prueba. Los tres controles obtuvieron un resultado de cero.

El conjunto de prueba se mantuvo reservado y no fue evaluado durante este avance.

## 5. Implementación del baseline

Se implementó un modelo de referencia basado en la clasificación de todos los registros como la clase mayoritaria del conjunto de entrenamiento.

Resultados obtenidos en validación:

| Métrica | Resultado |
|---|---:|
| Clase mayoritaria | 0 |
| Exactitud (Accuracy) | 55,61 % |
| F1 macro | 0,3574 |

Este modelo permitió establecer una referencia sencilla para comparar el desempeño de nuestro algoritmo K-NN.

## 6. Implementación y evaluación de K-NN

Se desarrolló una implementación propia del algoritmo K-Nearest Neighbors (K-NN), utilizando la distancia euclidiana y la votación mayoritaria.

Se normalizaron las variables utilizando únicamente los datos de entrenamiento.

Se evaluaron los siguientes valores de K:

| Valor de K | Exactitud | F1 macro |
|---:|---:|---:|
| 1 | 97,07 % | 0,9705 |
| 3 | **97,56 %** | **0,9754** |
| 5 | 96,10 % | 0,9608 |
| 7 | 95,61 % | 0,9559 |
| 9 | 95,61 % | 0,9559 |
| 11 | 95,61 % | 0,9559 |

El valor seleccionado fue K = 3, debido a que presentó los mejores resultados en el conjunto de validación.

## 7. Matriz de confusión

Para K = 3 se obtuvo la siguiente matriz de confusión:

| Clase real / predicha | Clase 0 | Clase 1 |
|---|---:|---:|
| Clase 0 | 109 | 5 |
| Clase 1 | 0 | 91 |

El modelo clasificó correctamente 200 de los 205 registros de validación y presentó cinco clasificaciones incorrectas.

La exactitud obtenida fue de 97,56 % y el F1 macro fue de 0,9754.

## 8. Pruebas unitarias

Se realizaron tres pruebas unitarias para comprobar el funcionamiento del núcleo K-NN:

| Prueba | Descripción | Resultado |
|---|---|---|
| Prueba 1 | Clasificación con K = 1 | Aprobada |
| Prueba 2 | Votación mayoritaria con K = 3 | Aprobada |
| Prueba 3 | Resolución de empate | Aprobada |

Las tres pruebas finalizaron satisfactoriamente y produjeron las clasificaciones esperadas.

## 9. Conclusiones del avance

La preparación y el análisis exploratorio permitieron comprobar que el conjunto de datos dispone de información suficiente para desarrollar el prototipo de clasificación.

La implementación propia del algoritmo K-NN obtuvo mejores resultados que el baseline en el conjunto de validación. El valor K = 3 presentó el mejor desempeño entre los valores evaluados.

Asimismo, las pruebas unitarias confirmaron el funcionamiento de las operaciones principales del algoritmo, incluyendo el cálculo de distancias, la votación mayoritaria y la regla de desempate.

Como siguiente etapa se realizarán experimentos controlados, se profundizará el análisis de errores y se evaluará el modelo en el conjunto de prueba reservado.

## 10. Evidencias digitales

Las gráficas generadas durante el análisis exploratorio se encuentran en la carpeta `salidas/` del repositorio:

- `01_distribuciones.png`
- `02_balance_clases.png`
- `03_correlaciones.png`
- `resumen_avance2.json`

El notebook y el código fuente contienen la implementación y los procedimientos utilizados para obtener los resultados.

**Nota metodológica:** las etiquetas se mantienen como Clase 0 y Clase 1. No se les atribuye todavía el significado de auténtico o falsificado sin verificar la correspondencia documental. Los resultados presentados corresponden exclusivamente al conjunto de validación y no constituyen una evaluación final del sistema.
