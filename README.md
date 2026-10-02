# CardioRiesgo IA

Modelo de inteligencia artificial orientado a identificar el riesgo de enfermedades cardiovasculares a partir de datos clínicos y características de los pacientes, construido y desplegado con **Azure Machine Learning**.

**Microproyecto 3**
*Prototipo académico de apoyo a la decisión clínica. No reemplaza el criterio médico ni constituye un diagnóstico.*

## El problema

Las enfermedades cardiovasculares son la principal causa de muerte en Colombia: cerca de 219 personas mueren cada día por esta causa, y el país cuenta con apenas unos 1.200 cardiólogos. CardioRiesgo IA le ofrece al cardiólogo una segunda opinión rápida y basada en datos para identificar a los pacientes con mayor riesgo y priorizar su atención.

## Datos

Se usa el **Framingham Heart Study** ([Kaggle](https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study)), con unos 4.000 participantes. La variable objetivo es `TenYearCHD`, el riesgo de enfermedad coronaria a 10 años.

Variables utilizadas: `age`, `sysBP`, `diaBP`, `totChol`, `currentSmoker`, `cigsPerDay`, `glucose`, `BMI` y `diabetes`.

El dataset está **desbalanceado** (solo cerca del 15 % de los casos son positivos), por lo que el *accuracy* no se usa como métrica principal. El modelo se evalúa con **recall, AUC y F1**, y el desbalance se corrige con **SMOTE** aplicado solo a los datos de entrenamiento.

> El CSV no se incluye en el repositorio. Descárgalo de Kaggle y súbelo a Azure ML como *Data asset*.

## Recursos en Azure

| Recurso | Nombre / configuración |
|---|---|
| Suscripción | Azure for Students |
| Región | **Canada Central** (la suscripción de estudiante no permite East US ni Korea Central) |
| Grupo de recursos | `rg-cardioriesgo` |
| Workspace de Azure ML | `ws-cardioriesgo` |
| Compute cluster | `cpu-cardio` · Standard_DS11_v2 (2 núcleos, 14 GB) · mín. 0 / máx. 1 nodos · 120 s de inactividad |
| Data asset | `framingham` · Tabular (API v1) · 17 columnas |
| Servicios asociados | Storage Account, Key Vault, Application Insights, Log Analytics (creados con el workspace); Container Registry (se crea al desplegar) |

**Nota sobre el esquema del dataset:** en el CSV original los valores faltantes vienen como `NA`, por lo que Azure detectó como texto las columnas `education`, `cigsPerDay`, `BPMeds`, `totChol`, `BMI` y `glucose`. Se cambiaron a **Decimal (punto)** al crear el data asset; los `NA` quedan como nulos y se imputan con la mediana en el pipeline.

## Solución en Azure

Todo el modelo se construye en **Azure ML Designer**, sin código:

1. Select Columns in Dataset: las 9 variables y la etiqueta.
2. Clean Missing Data: reemplazo con la mediana en `totChol`, `cigsPerDay`, `glucose` y `BMI` (la mediana es robusta a valores extremos, como glucosas muy altas).
3. Edit Metadata: `currentSmoker` y `diabetes` como categóricas.
4. Normalize Data: ZScore en `age`, `sysBP`, `diaBP`, `totChol`, `cigsPerDay`, `glucose` y `BMI`.
5. Split Data: 70 % / 30 %, estratificado por `TenYearCHD`, semilla 123.
6. SMOTE: balanceo solo del conjunto de entrenamiento (300 %, 5 vecinos, semilla 123). El 30 % de prueba conserva la proporción real.
7. Entrenamiento y comparación de 4 algoritmos: Logistic Regression, Boosted Decision Tree, Decision Forest y Neural Network (Two-Class).
8. Evaluate Model (recall, AUC, F1) y Permutation Feature Importance.
9. Despliegue del mejor modelo como **endpoint en tiempo real** en Azure Container Instances.

Los diagramas de arquitectura y del pipeline están en `docs/diagramas/` (se abren en [app.diagrams.net](https://app.diagrams.net)).

## Estructura del repositorio

```
CardioRiesgo-IA/
├── README.md
├── app/                          Formulario web que consume el endpoint
│   ├── app.py
│   ├── requirements.txt
│   └── .streamlit/
│       └── secrets.toml.example  Plantilla para la URL y la llave
├── docs/
│   ├── diagramas/                Arquitectura y pipeline (draw.io)
│   └── costos/                   Estimación de la calculadora de Azure
├── evidencias/                   Pantallazos del pipeline, métricas y demo
└── data/                         (vacía: el CSV se descarga de Kaggle)
```

## Cómo ejecutar el formulario

1. Despliega el endpoint en Azure ML y copia, desde la pestaña **Consume**, la URL REST y la *Primary key*.
2. Copia `app/.streamlit/secrets.toml.example` como `app/.streamlit/secrets.toml` y pega ahí la URL y la llave.
3. Ejecuta:

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

El formulario pide los datos del paciente, consulta el endpoint y muestra la probabilidad de riesgo. Incluye un **umbral de alerta** ajustable (por defecto 0,30) para priorizar el recall.

> **Importante:** `secrets.toml` está en `.gitignore`. Nunca subas la llave del endpoint al repositorio.


## Costos estimados

| Escenario | Costo mensual (USD) |
|---|---|
| Desarrollo y demo | 12,60 |
| Producción (proyección, endpoint 24/7) | 51,53 |

> La cotización se hizo en East US, pero los recursos se desplegaron en **Canada Central** por la restricción de regiones de la suscripción de estudiante. Los precios en Canada Central son ligeramente distintos.

El detalle está en `docs/costos/estimacion_costos_azure.xlsx`.


