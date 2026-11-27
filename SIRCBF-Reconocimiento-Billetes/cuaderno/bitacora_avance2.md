# Bitácora manuscrita — Avance 2

Copiar a mano y completar con información real del equipo.

**Equipo:** __________________________  
**Fecha:** 27/11/2026  
**Integrantes presentes y firmas:** ______________________________________

### Esta semana intentamos
Preparar el conjunto Banknote Authentication de UCI, explorar las cuatro variables, separar entrenamiento/validación/prueba y desarrollar una primera implementación propia de K-NN.

### Qué funcionó
(Completar después de ejecutar el programa. Anotar los resultados que realmente se observaron.)

### Qué falló y por qué
(Registrar errores reales, mensajes de Python, problemas de instalación o decisiones corregidas.)

### Decisiones tomadas y razón
- Se eligió K-NN porque está incluido entre los algoritmos aceptados en la consigna.
- Se calcula la normalización usando únicamente el conjunto de entrenamiento para evitar fuga de información.
- Se utiliza una división estratificada con semilla fija 42 para reproducibilidad.
- Se elige K usando validación y se reserva prueba para la evaluación final.
- Se conservan temporalmente las etiquetas como Clase 0 y Clase 1 hasta verificar su significado documental.

### Qué aprendimos
(Completar con palabras propias después de ejecutar y explicar el programa.)

### Próximos pasos y responsable
- Ejecutar el script y guardar capturas: __________________
- Revisar histogramas, balance y correlaciones: __________________
- Comparar baseline y K-NN en validación: __________________
- Copiar y resolver la prueba de escritorio: __________________
- Subir commits de cada integrante: __________________

### Evidencia adjunta
Adjuntar capturas de la ejecución, gráficas generadas y tres pruebas unitarias aprobadas.
