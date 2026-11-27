# Ficha técnica de datos

| Campo | Descripción |
|---|---|
| Nombre | Banknote Authentication |
| Repositorio | UCI Machine Learning Repository |
| ID | 267 |
| Referencia | https://doi.org/10.24432/C55P57 |
| Instancias | 1.372 |
| Variables predictoras | 4 |
| Variable objetivo | `class` |
| Tipo de problema | Clasificación supervisada binaria |
| Valores faltantes | La ficha de UCI reporta que no hay valores faltantes |

## Variables predictoras

| Variable | Significado general |
|---|---|
| `variance` | Varianza de la imagen transformada |
| `skewness` | Asimetría de la distribución |
| `curtosis` | Curtosis de la distribución |
| `entropy` | Entropía de la imagen |

Las características fueron calculadas a partir de imágenes en escala de grises de especímenes mediante una transformada wavelet, según la descripción del repositorio UCI.

**Precaución:** la tabla del repositorio presenta la variable `class` como objetivo, pero antes de traducir Clase 0 y Clase 1 a «auténtico» y «falsificado» se debe verificar la asignación explícita de etiquetas. Hasta entonces, se mantendrán sus nombres numéricos.
