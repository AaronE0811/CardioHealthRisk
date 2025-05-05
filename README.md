# ❤️ Evaluación de Riesgo Cardíaco (Cardio Health Risk Assessment)

**Autor:** Aaron Elizondo

Este proyecto utiliza técnicas de Machine Learning para predecir el riesgo de enfermedad cardíaca basado en diversas variables clínicas extraídas de un conjunto de datos médico.

---

## 🧰 Bibliotecas Utilizadas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    GradientBoostingClassifier, AdaBoostClassifier,
    RandomForestClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


## 📂 Descripción del Conjunto de Datos

El conjunto de datos incluye las siguientes características:

- **Edad**: Edad del paciente
- **Sexo**: Género del paciente
- **Tipo de dolor de pecho**: Clasificación del dolor experimentado (angina típica, atípica, no anginal)
- **Presión arterial (BP)**: Presión arterial del paciente
- **Colesterol**: Nivel de colesterol en sangre
- **Glucosa en ayunas > 120**: Si el nivel de glucosa en ayunas es mayor a 120 mg/dl
- **Resultados del EKG**: Resultados del electrocardiograma
- **Frecuencia cardíaca máxima (Max HR)**: Alcanzada durante el ejercicio
- **Angina por ejercicio**: Si experimentó dolor durante el ejercicio
- **Depresión del ST**: Depresión del segmento ST posterior al ejercicio
- **Pendiente del ST**: Indicador de la gravedad de la enfermedad arterial coronaria
- **Vasos visibles por fluoroscopía**: Número de vasos visibles mediante fluoroscopía
- **Talio**: Resultado del test de esfuerzo con talio
- **Enfermedad cardíaca**: Diagnóstico final (presencia o no de enfermedad)

---

## 🤖 Modelos Entrenados

Se entrenaron los siguientes modelos para evaluar su rendimiento:

- `logistic`: Regresión Logística  
- `decision_tree`: Árbol de Decisión  
- `svm`: Máquina de Vectores de Soporte  
- `knn`: K-Vecinos más Cercanos  
- `random_forest`: Bosques Aleatorios  
- `gradient_boosting`: Gradient Boosting  
- `adaboost`: AdaBoost  
- `naive_bayes`: Naive Bayes  
- `mlp`: Perceptrón Multicapa  
- `xgboost`: XGBoost  

---

## ✅ Modelos Finales Seleccionados

### 🔹 Regresión Logística

- **Curva ROC AUC**: 0.95  
- **Métricas (Antes → Después de mejora):**
  - **F1**: `0.9275 → 0.9429`
  - **Recall**: `0.9697 → 1.0`
  - **Precisión**: `0.8889 → 0.8919`
  - **Accuracy**: `0.9074 → 0.9259`
- **Observación**: Solo 1 error en 20 predicciones. Muy alto rendimiento.

### 🔸 Naive Bayes

- **Error** en 3 de 20 predicciones.  
- No se logró mejorar la curva ROC ni otras métricas ajustando hiperparámetros.  
- Desempeño inferior comparado con la regresión logística.

---

## 📌 Conclusión

La **Regresión Logística** mostró un mejor desempeño general para predecir enfermedades cardíacas en comparación con **Naive Bayes**, destacando por su precisión, recall perfecto y una ROC AUC de 0.95.
