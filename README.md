# Proyecto-UTP-MP-RM | Proyect-UTP-MP-RM

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

**Predicting Academic Performance in Higher Education**  
Final Project – Master’s in Data Analytics  
**Author:** Rolando Mora (9-756-1619)  
**Repository:** ProjectMP  

## 📁 Repository Contents  
This repository contains the complete development of the final project for the Predictive Models course. The main goal is to analyze and predict university students’ academic performance using machine learning techniques and statistical analysis.

## 📊 Data and Preprocessing  
- `DATA.csv`: Original dataset used (source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation))  
- `Cleaned_Dataset_Sin_CourseID.csv`: Cleaned and preprocessed dataset for analysis and modeling  
- Scripts for:
  - Missing value imputation  
  - Categorical encoding  
  - Data cleaning  
  - Variable reduction  

---

## 📈 Statistical Analysis  
- `Resumen_Descriptivo_Ordinal_Correcto.csv`: Descriptive statistics of ordinal variables by performance level  
- `Kruskal_Wallis_Correcto.csv`: Kruskal-Wallis test results  
- `Chi_Cuadrado_Correcto.csv`: Chi-square test results  
- Boxplots and distribution histograms

---

## 🤖 Predictive Modeling  
- `Clasificacion Binaria.py`: Main script for model training and evaluation  
- **Models used:**
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Naive Bayes  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Machine (SVM)  

- **Two comparative scenarios:**
  1. **Scenario 1:** Using all available variables  
  2. **Scenario 2:** Using only statistically significant variables  

---

## 📊 Results and Visualizations  
- `Comparacion_Modelos_Todos_vs_Significativos.xlsx`: Comparative metrics across both scenarios  
- `Escenario2_Variables_Significativas_Resultados.xlsx`: Detailed results per model for Scenario 2  
- Confusion matrices  
- Variable importance plots by model and scenario  

---

## 📝 Documentation  
- `Reporte de Avance Proyecto Final MP - Rolando Mora.docx`: Technical report including:
  - Introduction  
  - Justification  
  - Methodology  
  - Analysis  
  - Results  
  - Conclusions  
  - Recommendations  

---

## 🔍 Topics Covered  
- Statistical variable selection  
- Classification metrics (Accuracy, Recall, Precision, F1-Score)  
- Model interpretability and educational application  
- Tools used:  
  - `scikit-learn`  
  - `matplotlib`  
  - `seaborn`  
  - `pandas`  
  - `numpy`  

---

## ✅ Conclusions  
Models like Random Forest and Decision Tree, combined with careful variable selection, allow for reliable prediction of academic performance. The results also highlight the technical and pedagogical feasibility of implementing such models in early warning systems within higher education institutions.
