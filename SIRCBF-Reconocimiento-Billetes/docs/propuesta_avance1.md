# Propuesta de proyecto — Avance 1

## Título
**Sistema inteligente de reconocimiento y clasificación de especímenes de billetes mediante aprendizaje automático (SIRCBF).**

## 1. Planteamiento del problema

La falsificación de moneda constituye un problema económico y de seguridad. La revisión de especímenes puede apoyarse en características observables y procedimientos especializados. En este proyecto se estudiará un problema acotado de clasificación: distinguir dos clases de especímenes a partir de características numéricas extraídas previamente de imágenes.

Se utilizará el conjunto Banknote Authentication del UCI Machine Learning Repository. Por tanto, el prototipo no analizará directamente fotografías nuevas ni estará validado para billetes bolivianos en circulación. Su propósito es construir y evaluar un clasificador académico sobre los datos disponibles.

## 2. Justificación

El proyecto permite aplicar conocimientos de programación, estadística y aprendizaje automático a un problema de clasificación con datos reales. Además, permite implementar el algoritmo K-NN desde sus fundamentos, verificar los cálculos mediante pruebas manuales y evaluar sus limitaciones. El trabajo también fortalece la capacidad del equipo para documentar decisiones técnicas y defenderlas oralmente.

## 3. Objetivo general

Desarrollar un prototipo de clasificación binaria basado en K vecinos más cercanos (K-NN), implementado desde cero en Python, que clasifique registros del dataset Banknote Authentication a partir de cuatro características numéricas.

## 4. Objetivos específicos

1. Obtener, describir y preparar el conjunto de datos Banknote Authentication.
2. Implementar el cálculo de distancias, selección de vecinos y votación mayoritaria del algoritmo K-NN.
3. Verificar el núcleo mediante pruebas unitarias y ejemplos numéricos resueltos a mano.
4. Evaluar el desempeño mediante métricas apropiadas y experimentos controlados.
5. Analizar errores, limitaciones, riesgos y alcance del prototipo.

## 5. Algoritmo seleccionado y justificación

Se selecciona K-Nearest Neighbors (K-NN), algoritmo supervisado de clasificación que asigna una clase a una muestra nueva según las clases de sus K vecinos más cercanos. Es apropiado para este proyecto porque permite una implementación propia comprensible: cálculo de distancias, selección de K y votación. Se utilizará distancia euclidiana y se estudiará el efecto de distintos valores de K.

## 6. Datos y variables

Fuente: UCI Machine Learning Repository, Banknote Authentication (dataset ID 267). La ficha reporta 1.372 instancias y cuatro variables predictoras: `variance`, `skewness`, `curtosis` y `entropy`. La variable objetivo es `class`, con dos categorías. Se mantendrán las etiquetas numéricas Clase 0 y Clase 1 hasta comprobar la correspondencia semántica en una fuente adecuada.

## 7. Métrica de éxito

Se utilizará la exactitud (accuracy) como métrica general, complementada por precisión, sensibilidad (recall), F1 y matriz de confusión. Se prestará atención a los errores por clase. La elección final se justificará con la distribución de clases y los objetivos del problema.

## 8. Alcance y limitaciones

El prototipo clasifica registros numéricos ya procesados; no es un sistema óptico de lectura de billetes ni un mecanismo oficial de autenticación. No se afirmará su validez para moneda boliviana sin datos locales representativos y evaluación independiente.

## 9. Herramientas previstas

Python, NumPy, pandas y Matplotlib. Scikit-learn podrá utilizarse solamente como comparación auxiliar al final, no como núcleo de la solución.
