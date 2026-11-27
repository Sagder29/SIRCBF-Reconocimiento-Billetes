"""SIRCBF - Avance 2
Clasificador K-NN implementado desde cero para el dataset UCI Banknote Authentication.

Uso:
    pip install numpy pandas matplotlib ucimlrepo
    python avance2_sircbf.py

Importante:
- El nucleo K-NN, la division estratificada y la normalizacion son codigo propio.
- Scikit-learn no se utiliza para entrenar, dividir ni normalizar.
- En este avance se evalua con VALIDACION. El conjunto TEST queda reservado
  para la evaluacion final y no se consulta aqui.
- Se conservan las etiquetas como Clase 0 y Clase 1. Verificar el significado
  semantico de cada etiqueta antes de nombrarlas autentico/falsificado.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
FEATURES = ["variance", "skewness", "curtosis", "entropy"]
K_CANDIDATES = [1, 3, 5, 7, 9, 11]
OUT_DIR = Path("salidas_sircbf")


def load_uci_banknote():
    """Descarga el dataset oficial UCI (ID 267) y devuelve un DataFrame."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit(
            "Falta ucimlrepo. Instala dependencias con: "
            "pip install numpy pandas matplotlib ucimlrepo"
        ) from exc

    dataset = fetch_ucirepo(id=267)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if y.shape[1] != 1:
        raise ValueError("Se esperaba una sola columna objetivo en el dataset UCI.")

    X.columns = [str(c).strip().lower() for c in X.columns]
    X = X[FEATURES]
    y = y.iloc[:, 0].rename("class")
    df = pd.concat([X, y], axis=1)
    return df


def clean_and_report(df):
    """Convierte columnas a numericas, informa faltantes/duplicados e infinitos."""
    df = df.copy()
    for col in FEATURES + ["class"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    infinite_mask = np.isinf(df[FEATURES].to_numpy(dtype=float)).any(axis=1)
    n_infinite = int(infinite_mask.sum())
    if n_infinite:
        df = df.loc[~infinite_mask].copy()

    n_missing_before = int(df[FEATURES + ["class"]].isna().sum().sum())
    df = df.dropna(subset=FEATURES + ["class"]).copy()
    n_duplicates = int(df.duplicated().sum())

    # No eliminamos duplicados automaticamente: se reportan para evitar
    # borrar observaciones sin una justificacion metodologica.
    df["class"] = df["class"].astype(int)

    print("\n=== CONTROL DE CALIDAD ===")
    print(f"Filas despues de retirar infinitos/faltantes: {len(df)}")
    print(f"Valores faltantes detectados antes de eliminarlos: {n_missing_before}")
    print(f"Filas con infinitos detectadas y retiradas: {n_infinite}")
    print(f"Filas duplicadas detectadas (se conservan y reportan): {n_duplicates}")
    print("\nValores faltantes por columna:")
    print(df[FEATURES + ["class"]].isna().sum().to_string())
    return df.reset_index(drop=True)


def create_eda(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== DIMENSIONES DEL DATASET ===")
    print(f"Muestras: {len(df)}")
    print(f"Variables predictoras: {len(FEATURES)}")
    print(f"Clases: {sorted(df['class'].unique().tolist())}")

    print("\n=== ESTADISTICAS DESCRIPTIVAS ===")
    print(df[FEATURES].describe().round(4).to_string())

    print("\n=== BALANCE DE CLASES ===")
    counts = df["class"].value_counts().sort_index()
    print(counts.to_string())
    print("Porcentajes:")
    print((100 * counts / counts.sum()).round(2).astype(str).add("%").to_string())

    print("\n=== CORRELACIONES ENTRE VARIABLES ===")
    corr = df[FEATURES].corr(numeric_only=True)
    print(corr.round(3).to_string())

    # Distribucion de cada variable.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, feature in zip(axes.flat, FEATURES):
        ax.hist(df[feature], bins=30, edgecolor="black")
        ax.set_title(f"Distribucion: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_distribuciones.png", dpi=160)
    plt.close(fig)

    # Balance de clases.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(x) for x in counts.index], counts.values)
    ax.set_title("Balance de clases")
    ax.set_xlabel("Etiqueta de clase")
    ax.set_ylabel("Cantidad de muestras")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_balance_clases.png", dpi=160)
    plt.close(fig)

    # Matriz de correlacion.
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr.to_numpy(), aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=35, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.set_title("Correlacion entre variables")
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Correlacion")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_correlaciones.png", dpi=160)
    plt.close(fig)
    print(f"\nGraficas guardadas en: {OUT_DIR.resolve()}")


def stratified_split_indices(y, train_ratio=0.70, val_ratio=0.15, seed=SEED):
    """Division estratificada propia. El conjunto de prueba queda reservado."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        rng.shuffle(class_indices)
        n = len(class_indices)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            raise ValueError("Las proporciones de division no son validas.")
        train_parts.extend(class_indices[:n_train])
        val_parts.extend(class_indices[n_train:n_train + n_val])
        test_parts.extend(class_indices[n_train + n_val:])

    train_idx = np.array(train_parts, dtype=int)
    val_idx = np.array(val_parts, dtype=int)
    test_idx = np.array(test_parts, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def standardize_fit(X_train):
    """Calcula media y desviacion solo con entrenamiento (evita fuga de datos)."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0, ddof=0)
    std[std == 0] = 1.0
    return mean, std


def standardize_transform(X, mean, std):
    return (np.asarray(X, dtype=float) - mean) / std


def knn_predict_one(X_train, y_train, x_new, k):
    """Predice una muestra por distancia euclidiana y votacion mayoritaria.

    Desempate: si dos o mas clases tienen igual numero de votos, se escoge
    la etiqueta numericamente menor. Es una regla determinista documentada.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    x_new = np.asarray(x_new, dtype=float)

    if X_train.ndim != 2 or x_new.ndim != 1:
        raise ValueError("Dimensiones incorrectas para X_train o x_new.")
    if X_train.shape[1] != x_new.shape[0]:
        raise ValueError("La muestra nueva debe tener el mismo numero de variables.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train e y_train deben tener la misma cantidad de filas.")
    if not (1 <= k <= len(X_train)):
        raise ValueError("K debe estar entre 1 y la cantidad de muestras de entrenamiento.")

    distances = np.sqrt(np.sum((X_train - x_new) ** 2, axis=1))
    nearest_idx = np.argsort(distances, kind="stable")[:k]
    neighbor_labels = y_train[nearest_idx]
    labels, votes = np.unique(neighbor_labels, return_counts=True)
    max_votes = votes.max()
    tied_labels = labels[votes == max_votes]
    prediction = np.min(tied_labels)
    return prediction


def knn_predict(X_train, y_train, X_query, k):
    X_query = np.asarray(X_query, dtype=float)
    return np.array([knn_predict_one(X_train, y_train, row, k) for row in X_query])


def classification_metrics(y_true, y_pred):
    """Metricas para etiquetas binarias 0/1, sin usar librerias de ML."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = [0, 1]
    cm = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if actual in labels and predicted in labels:
            cm[labels.index(actual), labels.index(predicted)] += 1

    accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    per_class_f1 = []
    per_class = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        per_class_f1.append(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(per_class_f1)),
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def run_unit_tests():
    """Tres pruebas pequenas con resultados conocidos/calculables a mano."""
    X_demo = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
    ])
    y_demo = np.array([0, 0, 0, 1, 1])

    # Prueba 1: un vecino mas cercano, cercano al grupo de clase 0.
    p1 = knn_predict_one(X_demo, y_demo, np.array([0.2, 0.2]), k=1)
    assert p1 == 0, f"Prueba 1 fallo: se esperaba 0 y salio {p1}"

    # Prueba 2: con K=3, dos vecinos de clase 1 y uno de clase 0.
    p2 = knn_predict_one(X_demo, y_demo, np.array([5.5, 5.2]), k=3)
    assert p2 == 1, f"Prueba 2 fallo: se esperaba 1 y salio {p2}"

    # Prueba 3: empate 1 a 1; regla estable: gana la etiqueta menor (clase 0).
    X_tie = np.array([[0.0, 0.0], [2.0, 0.0]])
    y_tie = np.array([0, 1])
    p3 = knn_predict_one(X_tie, y_tie, np.array([1.0, 0.0]), k=2)
    assert p3 == 0, f"Prueba 3 fallo: se esperaba 0 por desempate y salio {p3}"

    print("\n=== PRUEBAS UNITARIAS DEL NUCLEO K-NN ===")
    print("Prueba 1: OK - K=1, prediccion esperada Clase 0")
    print("Prueba 2: OK - K=3, prediccion esperada Clase 1")
    print("Prueba 3: OK - empate, regla de desempate Clase 0")


def main():
    print("SIRCBF - AVANCE 2 | K-NN implementado desde cero")
    df = load_uci_banknote()
    df = clean_and_report(df)
    create_eda(df)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["class"].to_numpy(dtype=int)
    train_idx, val_idx, test_idx = stratified_split_indices(y)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    # X_test/y_test se separan, pero deliberadamente NO se consultan en Avance 2.
    print("\n=== DIVISION ESTRATIFICADA (SEMILLA 42) ===")
    print(f"Entrenamiento: {len(train_idx)} muestras")
    print(f"Validacion:    {len(val_idx)} muestras")
    print(f"Prueba:        {len(test_idx)} muestras (reservadas; no evaluadas ahora)")

    # Baseline: siempre predice la clase mas frecuente del conjunto de entrenamiento.
    majority_class = int(pd.Series(y_train).value_counts().idxmax())
    baseline_pred = np.full(shape=len(y_val), fill_value=majority_class, dtype=int)
    baseline_metrics = classification_metrics(y_val, baseline_pred)
    print("\n=== BASELINE: CLASE MAYORITARIA ===")
    print(f"Clase mayoritaria en entrenamiento: {majority_class}")
    print(f"Exactitud en validacion: {baseline_metrics['accuracy']:.4f}")
    print(f"F1 macro en validacion: {baseline_metrics['macro_f1']:.4f}")
    print("Matriz de confusion [filas=real 0/1, columnas=predicho 0/1]:")
    print(baseline_metrics["confusion_matrix"])

    # Normalizacion calculada unicamente con entrenamiento.
    mean, std = standardize_fit(X_train)
    X_train_scaled = standardize_transform(X_train, mean, std)
    X_val_scaled = standardize_transform(X_val, mean, std)

    print("\n=== SELECCION DE K EN VALIDACION ===")
    results = []
    for k in K_CANDIDATES:
        predictions = knn_predict(X_train_scaled, y_train, X_val_scaled, k)
        metrics = classification_metrics(y_val, predictions)
        results.append((k, metrics["accuracy"], metrics["macro_f1"]))
        print(f"K={k:2d} | accuracy={metrics['accuracy']:.4f} | macro-F1={metrics['macro_f1']:.4f}")

    best_k, best_acc, best_macro_f1 = sorted(
        results, key=lambda row: (-row[1], -row[2], row[0])
    )[0]
    print(f"\nK seleccionado con validacion: {best_k}")
    print(f"Exactitud K-NN en validacion: {best_acc:.4f}")
    print(f"F1 macro K-NN en validacion: {best_macro_f1:.4f}")
    print("Comparacion: baseline accuracy = "
          f"{baseline_metrics['accuracy']:.4f}; K-NN accuracy = {best_acc:.4f}")
    print("NOTA: el conjunto de prueba permanece reservado para la entrega final.")

    run_unit_tests()
    print("\n=== FIN DEL AVANCE 2 ===")
    print("Guarda las salidas y graficas como evidencias. No presentes numeros "
          "hasta haber ejecutado este script y revisado sus resultados.")


if __name__ == "__main__":
    main()
