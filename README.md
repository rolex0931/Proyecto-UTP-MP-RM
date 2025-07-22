# Proyecto-UTP-MP-RM

# 🎓 Predicción del Rendimiento Académico en Educación Superior

**Proyecto Final - Maestría en Analítica de Datos**  
**Autor:** Rolando Mora (9-756-1619)  
**Repositorio:** `ProjectMP`

## 📁 Contenido del Repositorio

Este repositorio contiene el desarrollo completo del proyecto final para la asignatura de Modelos Predictivos, cuyo objetivo principal es analizar y predecir el rendimiento académico de estudiantes universitarios mediante técnicas de aprendizaje automático y análisis estadístico.

### 📊 Datos y Preprocesamiento
- `DATA.csv`: Dataset original utilizado (fuente: UCI Machine Learning Repository).
- `Cleaned_Dataset_Sin_CourseID.csv`: Dataset limpio y preprocesado para análisis y modelado.
- Scripts para imputación, codificación, limpieza y reducción de variables.

### 📈 Análisis Estadístico
- `Resumen_Descriptivo_Ordinal_Correcto.csv`: Estadísticas descriptivas de variables ordinales por nivel de rendimiento.
- `Kruskal_Wallis_Correcto.csv`: Resultados de prueba de Kruskal-Wallis.
- `Chi_Cuadrado_Correcto.csv`: Resultados de prueba de Chi-cuadrado.
- Gráficas de boxplots e histogramas de distribución.

### 🤖 Modelado Predictivo
- `Clasificacion Binaria.py`: Script principal de entrenamiento y evaluación de modelos.
- Modelos utilizados:
  - Regresión Logística
  - Árbol de Decisión (Decision Tree)
  - Bosque Aleatorio (Random Forest)
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
  - SVM
- Dos escenarios comparativos:
  - Escenario 1: con todas las variables.
  - Escenario 2: solo con variables estadísticamente significativas.

### 📊 Resultados y Visualizaciones
- `Comparacion_Modelos_Todos_vs_Significativos.xlsx`: Comparación completa de métricas entre ambos escenarios.
- `Escenario2_Variables_Significativas_Resultados.xlsx`: Resultados detallados por modelo en el segundo escenario.
- Matrices de confusión y gráficos de importancia de variables por modelo y escenario.

### 📝 Documentación
- `Reporte de Avance Proyecto Final MP - Rolando Mora.docx`: Documento técnico con introducción, justificación, metodología, análisis, resultados, conclusiones y recomendaciones.

## 🔍 Temas Abordados
- Selección estadística de variables
- Evaluación de modelos mediante métricas de clasificación (accuracy, recall, precision, F1-score)
- Interpretabilidad de modelos y aplicación educativa
- Uso de herramientas como `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`

## ✅ Conclusiones
Los resultados muestran que el uso de modelos como Random Forest o Árbol de Decisión, en conjunto con una selección cuidadosa de variables relevantes, permite predecir con buena precisión el desempeño académico. Además, se evidencia la viabilidad técnica y pedagógica de implementar estas soluciones en sistemas de alerta temprana en instituciones de educación superior.
