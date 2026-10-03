"""
CardioRiesgo IA
Formulario de demo que consume el endpoint en tiempo real de Azure ML.

Cómo ejecutarlo:
    pip install streamlit requests
    streamlit run app.py

Antes de ejecutarlo, copia la URL REST y la llave del endpoint
(Azure ML Studio -> Endpoints -> tu endpoint -> pestaña "Consume")
en el archivo app/.streamlit/secrets.toml (ver secrets.toml.example).
"""

import json

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# 1. Datos del endpoint (pestaña "Consume" del endpoint en Azure ML Studio)
# ---------------------------------------------------------------------------
# La URL y la llave NO se escriben aquí para no subirlas a Git.
# Se leen del archivo app/.streamlit/secrets.toml (que está en .gitignore)
# o de variables de entorno con el mismo nombre.
import os


def leer_secreto(nombre):
    try:
        return st.secrets[nombre]
    except Exception:
        return os.environ.get(nombre, "")


ENDPOINT_URL = leer_secreto("ENDPOINT_URL")   # termina en /score
API_KEY = leer_secreto("API_KEY")

# Nombre del componente Web Service Input. Revísalo en el ejemplo de la
# pestaña "Consume"; casi siempre es "WebServiceInput0".
NOMBRE_INPUT = "WebServiceInput0"

# Columnas que el endpoint espera y que el formulario NO pide.
# Si en la pestaña "Consume" aparecen columnas extra (male, education,
# BPMeds, prevalentStroke, prevalentHyp, heartRate...), agrégalas aquí
# con un valor por defecto. La etiqueta se envía en 0 porque el modelo
# no la usa para predecir.
COLUMNAS_EXTRA = {
    "TenYearCHD": 0,
}

# ---------------------------------------------------------------------------
# 2. Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CardioRiesgo IA", page_icon="🫀", layout="centered")
st.title("CardioRiesgo IA")
st.caption("Apoyo a la decisión clínica · Prototipo académico")

st.write(
    "Ingrese los datos del paciente para estimar su riesgo de enfermedad "
    "coronaria a 10 años. El resultado es un apoyo para el cardiólogo y no "
    "reemplaza el criterio médico."
)

with st.form("paciente"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Edad (años)", min_value=18, max_value=100, value=55)
        sysBP = st.number_input("Presión sistólica (mmHg)", min_value=80.0, max_value=260.0, value=130.0)
        diaBP = st.number_input("Presión diastólica (mmHg)", min_value=40.0, max_value=160.0, value=85.0)
        totChol = st.number_input("Colesterol total (mg/dL)", min_value=100.0, max_value=600.0, value=230.0)
        glucose = st.number_input("Glucosa (mg/dL)", min_value=40.0, max_value=400.0, value=85.0)
    with col2:
        BMI = st.number_input("Índice de masa corporal (IMC)", min_value=15.0, max_value=60.0, value=26.0)
        currentSmoker = st.selectbox("¿Fuma actualmente?", ["No", "Sí"])
        cigsPerDay = st.number_input("Cigarrillos al día", min_value=0, max_value=70, value=0)
        diabetes = st.selectbox("¿Tiene diabetes?", ["No", "Sí"])

    umbral = st.slider(
        "Umbral de alerta",
        min_value=0.10, max_value=0.90, value=0.30, step=0.05,
        help="Un umbral bajo prioriza el recall: detecta más pacientes en riesgo "
             "a cambio de algunas falsas alarmas.",
    )
    enviar = st.form_submit_button("Calcular riesgo")


def buscar_valor(obj, nombres):
    """Busca recursivamente la primera clave que coincida en la respuesta JSON."""
    if isinstance(obj, dict):
        for clave, valor in obj.items():
            if clave in nombres:
                return valor
            encontrado = buscar_valor(valor, nombres)
            if encontrado is not None:
                return encontrado
    elif isinstance(obj, list):
        for item in obj:
            encontrado = buscar_valor(item, nombres)
            if encontrado is not None:
                return encontrado
    return None


if enviar:
    if not ENDPOINT_URL or not API_KEY:
        st.error("Falta configurar ENDPOINT_URL y API_KEY en app/.streamlit/secrets.toml")
        st.stop()
    if currentSmoker == "No" and cigsPerDay > 0:
        st.warning("Indicó que el paciente no fuma, pero registró cigarrillos al día. Revise los datos.")

    paciente = {
        "age": int(age),
        "sysBP": float(sysBP),
        "diaBP": float(diaBP),
        "totChol": float(totChol),
        "currentSmoker": 1 if currentSmoker == "Sí" else 0,
        "cigsPerDay": float(cigsPerDay),
        "glucose": float(glucose),
        "BMI": float(BMI),
        "diabetes": 1 if diabetes == "Sí" else 0,
        **COLUMNAS_EXTRA,
    }
    cuerpo = {"Inputs": {NOMBRE_INPUT: [paciente]}, "GlobalParameters": {}}
    encabezados = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        with st.spinner("Consultando el modelo en Azure..."):
            r = requests.post(ENDPOINT_URL, data=json.dumps(cuerpo), headers=encabezados, timeout=30)
        r.raise_for_status()
        respuesta = r.json()
        if isinstance(respuesta, str):  # algunos endpoints devuelven JSON como texto
            respuesta = json.loads(respuesta)
    except requests.exceptions.HTTPError:
        st.error(f"El endpoint respondió con error {r.status_code}: {r.text[:500]}")
        st.stop()
    except Exception as e:
        st.error(f"No se pudo conectar con el endpoint: {e}")
        st.stop()

    prob = buscar_valor(respuesta, {"Scored Probabilities", "Scored Probabilities_1", "probability"})
    if prob is None:
        st.error("No se encontró la probabilidad en la respuesta. Respuesta recibida:")
        st.json(respuesta)
        st.stop()

    prob = float(prob)
    riesgo_alto = prob >= umbral

    st.subheader("Resultado")
    st.metric("Probabilidad de enfermedad coronaria a 10 años", f"{prob:.0%}")
    if riesgo_alto:
        st.error(
            f"Riesgo ALTO (probabilidad {prob:.0%} ≥ umbral {umbral:.0%}). "
            "Se sugiere priorizar la valoración cardiológica."
        )
    else:
        st.success(
            f"Riesgo BAJO (probabilidad {prob:.0%} < umbral {umbral:.0%}). "
            "Continuar con controles de rutina."
        )

    with st.expander("Ver respuesta completa del endpoint"):
        st.json(respuesta)
