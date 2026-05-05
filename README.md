# 🍷 Wine Quality ML Pipeline - CI/CD

## 📌 Descripción

Este proyecto implementa un pipeline completo de Machine Learning automatizado utilizando prácticas de MLOps. El flujo incluye preprocesamiento de datos, entrenamiento, evaluación y registro del modelo usando MLflow, además de integración continua mediante GitHub Actions.

---

## 🎯 Objetivo

Automatizar el ciclo de vida de un modelo de Machine Learning, garantizando reproducibilidad, trazabilidad y ejecución automática en la nube.

---

## 📊 Dataset

Se utilizó el dataset de calidad de vino de la UCI:

- Variables fisicoquímicas del vino
- Variable objetivo: calidad (convertida a clasificación binaria)

---

## ⚙️ Pipeline

### 1. Preprocesamiento
- Conversión a clasificación binaria
- División train/test
- Escalamiento de datos

### 2. Entrenamiento
- Modelo: Logistic Regression
- Librería: scikit-learn

### 3. Evaluación
- Accuracy
- F1 Score

### 4. Tracking
- Uso de MLflow para:
  - Registro de métricas
  - Registro de parámetros
  - Guardado del modelo

---

## 🚀 CI/CD

Se implementó un pipeline en GitHub Actions que:

- Instala dependencias
- Ejecuta entrenamiento
- Evalúa el modelo
- Guarda artefactos (mlruns)

---

## 🛠️ Ejecución local

```bash
make install
make train
make test
