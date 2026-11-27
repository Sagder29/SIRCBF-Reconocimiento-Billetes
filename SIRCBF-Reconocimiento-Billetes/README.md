# SIRCBF — Sistema inteligente de reconocimiento y clasificación de especímenes de billetes

**Carrera:** Ingeniería de Sistemas  
**Asignatura:** Inteligencia Artificial  
**Proyecto:** Crea tu propia Inteligencia Artificial  
**Algoritmo principal:** K vecinos más cercanos (K-NN), implementado desde cero en Python.

## 1. Descripción

Este proyecto desarrolla un prototipo académico de clasificación binaria de especímenes de billetes utilizando cuatro características numéricas derivadas de imágenes. El sistema compara una muestra nueva con los ejemplos del conjunto de entrenamiento mediante distancia euclidiana y determina la clase por votación mayoritaria de sus vecinos más cercanos.

**Alcance actual:** el prototipo trabaja con características numéricas del dataset Banknote Authentication. No recibe fotografías de billetes bolivianos ni constituye un dispositivo oficial de autenticación de moneda.

## 2. Problema y objetivo

Se busca estudiar si un algoritmo K-NN implementado por el equipo puede clasificar correctamente registros del conjunto Banknote Authentication en sus dos clases. El objetivo es comprender, programar, verificar y evaluar el núcleo del algoritmo, documentando sus aciertos y limitaciones.

## 3. Datos

- Fuente: UCI Machine Learning Repository — Banknote Authentication.
- Dataset ID: 267.
- 1.372 instancias y cuatro variables predictoras, según la ficha de UCI.
- Variables: `variance`, `skewness`, `curtosis`, `entropy`.
- Objetivo: `class`.
- Se conservarán las etiquetas como **Clase 0** y **Clase 1** hasta verificar documentalmente el significado de cada etiqueta.
- Ficha y enlace: https://doi.org/10.24432/C55P57

## 4. Algoritmo

El núcleo K-NN se implementa sin `fit()` ni `KNeighborsClassifier` de scikit-learn. El equipo programa el cálculo de distancias, búsqueda de vecinos, votación y predicción. Las librerías permitidas se usan como apoyo para datos, cálculos auxiliares y gráficas.

## 5. Estructura del repositorio

```text
SIRCBF-Reconocimiento-Billetes/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── propuesta_avance1.md
│   ├── ficha_datos.md
│   └── plan_trabajo.md
├── data/
│   └── README.md
├── src/
│   └── avance2_sircbf.py
├── notebooks/
│   └── Avance2_SIRCBF.ipynb
├── cuaderno/
│   ├── README.md
│   ├── bitacora_avance2.md
│   └── prueba_de_escritorio_para_cuaderno.md
├── experimentos/
│   └── README.md
└── salidas/
    └── README.md
```

## 6. Cómo ejecutar

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar desde la carpeta raíz:

```bash
python src/avance2_sircbf.py
```

También se puede abrir `notebooks/Avance2_SIRCBF.ipynb` en Google Colab o Jupyter y ejecutar sus celdas en orden.

## 7. Estado del trabajo

Este repositorio es la base inicial del proyecto. Antes de presentar resultados, el equipo debe ejecutar el notebook o script, guardar las salidas reales, revisar las gráficas y completar la bitácora con lo que efectivamente ocurrió. No se deben inventar métricas ni conclusiones.

## 8. Uso de IA generativa

Las herramientas generativas se utilizan como apoyo para explicar conceptos, depurar errores y revisar la estructura. El equipo debe comprender, verificar y poder defender todo el código. La declaración de uso de IA se completará conforme a la consigna de la asignatura.

## 9. Fuente de datos

Lohweg, V. (2012). *Banknote Authentication* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C55P57
