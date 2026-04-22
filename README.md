# 🚀 Detección de Fraude en Salud - Proyecto MLOps

## 📌 Descripción General

Este proyecto implementa un **pipeline completo de MLOps** para la detección de fraude en reclamaciones del sector salud utilizando técnicas de Machine Learning.

El sistema simula un entorno real de producción e incluye:

* Ingesta y procesamiento de datos (ETL)
* Ingeniería de características
* Entrenamiento y evaluación de modelos
* Seguimiento de experimentos con MLflow
* Orquestación de procesos con Prefect
* Despliegue del modelo mediante API (FastAPI)
* Monitoreo del rendimiento del modelo

---

## 🎯 Problema de Negocio

El fraude en sistemas de salud genera pérdidas millonarias cada año.

Este proyecto busca:

> Identificar reclamaciones médicas potencialmente fraudulentas para reducir pérdidas económicas y mejorar los procesos de auditoría.

---

## 📊 Dataset

Fuente: Kaggle - Healthcare Fraud Detection Dataset

El dataset contiene múltiples entidades:

* Pacientes
* Proveedores
* Reclamaciones (claims)
* Pagos

Esto permite simular un sistema real con múltiples relaciones entre datos.

---

## 🧠 Objetivo del Modelo

Predecir si una reclamación es fraudulenta:

* `1` → Fraude
* `0` → No fraude

---

## 🏗️ Arquitectura del Proyecto

```bash
data → ETL → features → modelo → MLflow → API → monitoreo
```

---



## ⚙️ Tecnologías Utilizadas

* Python
* Pandas / NumPy
* Scikit-learn
* MLflow
* Prefect
* FastAPI
* Uvicorn
* SHAP (interpretabilidad)

---

## 🚀 Instalación

```bash
git clone https://github.com/sergioduran93/fraude-mlops.git
cd fraude-mlops

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## ▶️ Ejecución del Proyecto

### 1. Procesamiento de datos

```bash
python src/data/load_data.py
```

### 2. Entrenamiento del modelo

```bash
python src/models/train.py
```

### 3. Ejecutar MLflow

```bash
mlflow ui
```

Acceder en:

```
http://localhost:5000
```

---

## 🌐 Despliegue del Modelo (API)

```bash
uvicorn src.api.main:app --reload
```

Endpoint ejemplo:

```
POST /predict
```

---

## 📈 Métricas de Evaluación

Debido a la naturaleza del problema, se priorizan:

* Recall (detección de fraude)
* F1-score
* ROC-AUC

---

## 📊 Monitoreo

El sistema permite:

* Seguimiento de métricas del modelo
* Detección de data drift
* Reentrenamiento futuro

---

## 💡 Valor del Proyecto

Este proyecto demuestra:

* Implementación real de MLOps
* Integración de múltiples herramientas
* Resolución de un problema de negocio crítico
* Capacidad de llevar modelos a producción

---

## 👨‍💻 Autor

**Sergio Andrés Durán Vásquez**
**Ivan Stiven Castrillon**
**Diego Castaneda**

---

## 📌 Notas

Este proyecto tiene fines académicos y de portafolio, simulando un entorno real de producción en el sector salud.

---
