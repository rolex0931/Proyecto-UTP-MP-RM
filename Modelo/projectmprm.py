#PROYECTO FINAL ROLANDO MORA 9-756-1619

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import kruskal, chi2_contingency

# Cargar datos
df = pd.read_csv("DATA.csv")

# Columnas nombradas
df.columns = [
    "Student_ID", "Student_Age", "Sex", "High_School_Type", "Scholarship_Type",
    "Additional_Work", "Art_Sport_Activity", "Have_Partner", "Total_Salary",
    "Transport_Type", "Accommodation_Type", "Mother_Education", "Father_Education",
    "Number_Siblings", "Parental_Status", "Mother_Occupation", "Father_Occupation",
    "Weekly_Study_Hours", "Reading_Freq_NonScientific", "Reading_Freq_Scientific",
    "Department_Seminar_Attendance", "Project_Impact_Success", "Class_Attendance",
    "Preparation_Midterm1", "Preparation_Midterm2", "Taking_Notes",
    "Listening_Classes", "Discussion_Interest_Success", "Flip_Classroom",
    "GPA_Last_Semester", "Expected_GPA_Graduation", "Course_ID", "GRADE"
]

# Agrupar GRADE en dos niveles
# Bajo: 0-3 | Alto: 4-7
df["Grade_Grouped"] = pd.cut(df["GRADE"], bins=[-1, 3, 7], labels=["Bajo", "Alto"])

# Eliminar valores nulos
df.dropna(inplace=True)

# Visualización de distribución de clases
os.makedirs("graficos_distribucion", exist_ok=True)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Grade_Grouped", hue="Grade_Grouped", palette="pastel", legend=False)
plt.title("Distribución de Clases - Grade_Grouped")
plt.xlabel("Clase")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("graficos_distribucion/Distribucion_Clases.png")
plt.close()

# Codificar variables categóricas
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    if col not in ["Student_ID", "Grade_Grouped"]:
        df[col] = le.fit_transform(df[col])

# Guardar base de datos limpia
df.drop(columns=["Course_ID"]).to_csv("Cleaned_Dataset_Sin_CourseID.csv", index=False)

# Clasificación de variables
ordinal_vars = [
    "Student_Age", "Mother_Education", "Father_Education", "Number_Siblings", "Total_Salary",
    "Weekly_Study_Hours", "Reading_Freq_NonScientific", "Reading_Freq_Scientific",
    "Department_Seminar_Attendance", "Project_Impact_Success", "Class_Attendance",
    "Preparation_Midterm1", "Preparation_Midterm2", "Taking_Notes", "Listening_Classes",
    "Discussion_Interest_Success", "Flip_Classroom", "GPA_Last_Semester", "Expected_GPA_Graduation"
]

nominal_vars = [
    "Sex", "High_School_Type", "Scholarship_Type", "Additional_Work", "Art_Sport_Activity",
    "Have_Partner", "Accommodation_Type", "Mother_Occupation", "Father_Occupation",
    "Parental_Status", "Transport_Type"
]

# Estadísticas descriptivas para ordinales
print("Ejecutando estadísticas descriptivas")
df_ord = df[ordinal_vars + ["Grade_Grouped"]]
desc_stats = df_ord.groupby("Grade_Grouped", observed=False).agg(['mean', 'std', 'min', 'max'])
desc_stats.columns = ['_'.join(col).strip() for col in desc_stats.columns.values]
desc_stats.to_csv("Resumen_Descriptivo_Ordinal_Correcto.csv")

# Boxplots
os.makedirs("boxplots_corr", exist_ok=True)
for col in ordinal_vars:
    plt.figure()
    sns.boxplot(data=df_ord, x="Grade_Grouped", y=col)
    plt.title(f"Boxplot - {col}")
    plt.savefig(f"boxplots_corr/{col}_boxplot.png")
    plt.close()

# Kruskal-Wallis
kruskal_df = []
for col in ordinal_vars:
    groups = [group[col].values for _, group in df_ord.groupby("Grade_Grouped", observed=False)]
    stat, p = kruskal(*groups)
    kruskal_df.append({"Variable": col, "H_statistic": stat, "p_value": p})
kruskal_results = pd.DataFrame(kruskal_df)
kruskal_results.to_csv("Kruskal_Wallis_Correcto.csv", index=False)
print("\nResultados de la prueba Kruskal-Wallis:")
print(kruskal_results.to_string(index=False))

# Frecuencias nominales y Chi-cuadrado
os.makedirs("frecuencias_nominales_corr", exist_ok=True)
chi2_results = []
for col in nominal_vars:
    freq = pd.crosstab(df["Grade_Grouped"], df[col])
    freq.to_csv(f"frecuencias_nominales_corr/Frecuencia_{col}.csv")
    contingency = pd.crosstab(df[col], df["Grade_Grouped"])
    chi2, p, _, _ = chi2_contingency(contingency)
    chi2_results.append({"Variable": col, "Chi2_statistic": chi2, "p_value": p})
chi2_df = pd.DataFrame(chi2_results)
chi2_df.to_csv("Chi_Cuadrado_Correcto.csv", index=False)

print("\nResultados de la prueba Chi-cuadrado:")
print(chi2_df.to_string(index=False))

# Modelado
print("Ejecutando modelado")

# Variables significativas identificadas previamente
selected_features = [
    "GPA_Last_Semester", "Expected_GPA_Graduation", "Reading_Freq_NonScientific",
    "Project_Impact_Success", "Mother_Education", "Student_Age", "Sex", "Scholarship_Type"
]

# Función de evaluación de modelos
modelos = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

os.makedirs("conf_matrices_corr", exist_ok=True)

def evaluar_modelos(X, y, descripcion):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    resultados = []

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = np.mean(y_pred == y_test)

        # Guardar matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=["Bajo", "Alto"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Bajo", "Alto"])
        disp.plot()
        plt.title(f"Matriz de Confusión - {nombre} ({descripcion})")
        plt.tight_layout()
        fname = f"conf_matrices_corr/Confusion_{nombre.replace(' ', '_')}_{descripcion.replace(' ', '_')}.png"
        plt.savefig(fname)
        plt.close()

        resultados.append({
            "Modelo": nombre,
            "Escenario": descripcion,
            "Accuracy": round(accuracy, 3),
            "Macro F1 Score": round(reporte["macro avg"]["f1-score"], 3),
            "F1 Bajo": round(reporte.get("Bajo", {}).get("f1-score", 0), 3),
            "Precision Bajo": round(reporte.get("Bajo", {}).get("precision", 0), 3),
            "Recall Bajo": round(reporte.get("Bajo", {}).get("recall", 0), 3),
            "F1 Alto": round(reporte.get("Alto", {}).get("f1-score", 0), 3),
            "Precision Alto": round(reporte.get("Alto", {}).get("precision", 0), 3),
            "Recall Alto": round(reporte.get("Alto", {}).get("recall", 0), 3)
        })

    df_resultados = pd.DataFrame(resultados)
    print(f"\nResultados del escenario: {descripcion}")
    print(df_resultados)
    return df_resultados

# Escenario 1: Todas las variables excepto identificadores
columnas_excluir = ["Student_ID", "GRADE", "Grade_Grouped", "Course_ID"]
columnas_validas = [col for col in df.columns if col not in columnas_excluir]
X_all = df[columnas_validas]
y = df["Grade_Grouped"]
res_all = evaluar_modelos(X_all, y, "Todas las variables")

# Escenario 2: Solo variables significativas
X_sig = df[selected_features]
res_sig = evaluar_modelos(X_sig, y, "Variables significativas")

# Combinar y guardar resultados
resumen_comparativo = pd.concat([res_all, res_sig], ignore_index=True)
resumen_comparativo.to_csv("Comparacion_Modelos_Todos_vs_Significativos.csv", index=False)
print("\nResumen comparativo de modelos (Todas las variables vs. Significativas):")
print(resumen_comparativo)

# Importancia de variables – RF y DT
print("Generando importancia de variables")

X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(X_sig, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_sig, y_train_sig)
importancia_rf = pd.Series(rf.feature_importances_, index=X_sig.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=importancia_rf.values, y=importancia_rf.index)
plt.title("Importancia de Variables – Random Forest")
plt.xlabel("Importancia")
plt.tight_layout()
plt.savefig("Importancia_RF_Significativas.png")
plt.close()

print("\nImportancia de variables - Random Forest:")
print(importancia_rf)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_sig, y_train_sig)
importancia_dt = pd.Series(dt.feature_importances_, index=X_sig.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=importancia_dt.values, y=importancia_dt.index)
plt.title("Importancia de Variables – Decision Tree")
plt.xlabel("Importancia")
plt.tight_layout()
plt.savefig("Importancia_DT_Significativas.png")
plt.close()

print("\nImportancia de variables - Decision Tree:")
print(importancia_dt)

# Guardar en CSV
pd.DataFrame({
    "Variable": X_sig.columns,
    "Importancia_RF": rf.feature_importances_,
    "Importancia_DT": dt.feature_importances_
}).to_csv("Importancia_RF_DT_Significativas.csv", index=False)