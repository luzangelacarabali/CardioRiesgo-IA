import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def make_app_icon():
    
    try:
        import math
        from PIL import Image, ImageDraw
        S, OUT = 512, 128
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, S - 1, S - 1], radius=110, fill=(29, 78, 216, 255))
        pts = []
        for i in range(361):
            t = math.radians(i)
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((S / 2 + x * 12.5, S / 2 - y * 12.5 + 8))
        d.polygon(pts, fill=(255, 255, 255, 255))
        ecg = [(110, 262), (190, 262), (225, 190), (272, 340), (312, 235), (336, 262), (402, 262)]
        d.line(ecg, fill=(37, 99, 235, 255), width=22, joint="curve")
        for x, y in (ecg[0], ecg[-1]):
            d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=(37, 99, 235, 255))
        return img.resize((OUT, OUT), Image.LANCZOS)
    except Exception:
        return "🩺"


st.set_page_config(page_title="CardioRiesgo IA", page_icon=make_app_icon(), layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
MODELS = {
    "Two-Class Logistic Regression": "LR",
    "Two-Class Boosted Decision Tree": "BDT",
    "Two-Class Decision Forest": "DF",
    "Two-Class Neural Network": "NN",
}
DEMO_VALUES = {  # valores SIMULADOS (los de tu versión anterior); no vienen de ningún modelo
    "Two-Class Logistic Regression": 0.18,
    "Two-Class Boosted Decision Tree": 0.23,
    "Two-Class Decision Forest": 0.20,
    "Two-Class Neural Network": 0.25,
}
FEATURES = ["age", "sysBP", "diaBP", "totChol", "currentSmoker",
            "cigsPerDay", "glucose", "BMI", "diabetes"]
EXAMPLE = dict(age=58, smoker="Sí", cigs=15, diabetes="No",
               sysBP=148, diaBP=92, totChol=245, glucose=96, BMI=29.0)
DEFAULTS = dict(age=55, smoker="No", cigs=0, diabetes="No",
                sysBP=130, diaBP=85, totChol=230, glucose=85, BMI=26.0)


def get_setting(name, default=""):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def endpoint_for(model_name):
    slug = MODELS[model_name]
    url = get_setting(f"AZURE_ENDPOINT_URL_{slug}") or get_setting("AZURE_ENDPOINT_URL")
    key = get_setting(f"AZURE_ENDPOINT_KEY_{slug}") or get_setting("AZURE_ENDPOINT_KEY")
    return url, key


# ----------------------------------------------------------------------------
# Cliente Azure ML (formato típico de Designer; verifica con la pestaña Consume)
# ----------------------------------------------------------------------------
def call_endpoint(url, key, input_name, record, timeout=40):
    payload = {"Inputs": {input_name: [record]}, "GlobalParameters": {}}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, str):
        data = json.loads(data)
    return payload, data


def parse_response(resp):
    results = resp.get("Results", resp.get("results", resp)) if isinstance(resp, dict) else resp
    rows = next(iter(results.values())) if isinstance(results, dict) else results
    if isinstance(rows, dict):
        rows = [rows]
    row = rows[0]
    norm = {str(k).lower().replace(" ", "").replace("_", ""): v for k, v in row.items()}
    label = norm.get("scoredlabels", norm.get("scoredlabel"))
    prob = norm.get("scoredprobabilities", norm.get("scoredprobability"))
    if label is None or prob is None:
        raise KeyError(f"No encontré 'Scored Labels'/'Scored Probabilities'. Columnas: {list(row.keys())}")
    label = int(float(str(label).strip().lower().replace("true", "1").replace("false", "0")))
    return {"label": label, "prob": float(prob)}


def http_hint(e):
    code = e.response.status_code if e.response is not None else None
    hints = {401: "Llave rechazada (401).", 403: "Acceso denegado (403).",
             404: "URL no encontrada (404): ¿el endpoint sigue desplegado?",
             502: "El endpoint no respondió (502).", 503: "Servicio no disponible (503): puede estar iniciando."}
    try:
        body = e.response.text[:500]
    except Exception:
        body = ""
    return f"{hints.get(code, f'Error HTTP {code}.')} Detalle: {body}"


# ----------------------------------------------------------------------------
# Estado y callbacks
# ----------------------------------------------------------------------------
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_example():
    for k, v in EXAMPLE.items():
        st.session_state[k] = v


def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.pop("result", None)


# ----------------------------------------------------------------------------
# CSS (cada bloque HTML se renderiza completo en UNA sola llamada de markdown)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
html {color-scheme: light;}
.stApp {background:#f4f7fb; color:#0f172a;}

/* ---- Texto general (colores explícitos para que siempre haya contraste) ---- */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {color:#0f172a;}
h1, h2, h3, h4 {color:#0f172a !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:#475569 !important;}
[data-testid="stWidgetLabel"] p, label p {color:#1e293b !important; font-weight:600;}
code {color:#be123c; background:#f1f5f9;}
pre code {color:#0f172a;}

/* ---- Campos de entrada: fondo blanco, texto oscuro ---- */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div
  {background:#ffffff !important; border-color:#cbd5e1 !important;}
input, textarea {color:#0f172a !important; -webkit-text-fill-color:#0f172a !important; background:#ffffff !important;}
div[data-baseweb="select"] span, div[data-baseweb="select"] div {color:#0f172a;}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {background:#ffffff !important; color:#0f172a !important;}
div[data-baseweb="popover"] li:hover {background:#eff6ff !important;}
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"]
  {background:#f1f5f9 !important; color:#0f172a !important;}

/* ---- Botones ---- */
div.stButton > button {background:#ffffff; color:#1e293b; border:1px solid #cbd5e1;}
div.stButton > button p {color:#1e293b !important;}
div.stButton > button:hover {border-color:#2563eb; color:#1d4ed8;}
div.stButton > button:hover p {color:#1d4ed8 !important;}
div[data-testid="stFormSubmitButton"] button, div[data-testid="stFormSubmitButton"] button p
  {color:#ffffff !important;}

/* ---- Pestañas, slider, expanders, alertas ---- */
button[data-baseweb="tab"] p {color:#475569 !important; font-weight:600;}
button[data-baseweb="tab"][aria-selected="true"] p {color:#1d4ed8 !important;}
div[data-baseweb="tab-highlight"] {background:#2563eb !important;}
div[data-testid="stSlider"] [role="slider"] {background:#2563eb !important;}
div[data-testid="stSlider"] [data-testid="stThumbValue"] {color:#1d4ed8 !important;}
details {background:#ffffff; border-color:#e2e8f0 !important; border-radius:12px;}
details summary, details summary p {color:#0f172a !important;}
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {color:#1e293b !important;}

/* ---- Header (debe ir después de las reglas generales) ---- */
.hero h1 {color:#ffffff !important;}
.hero p {color:#dbe7f5 !important;}
.pill {color:#e0ecff !important;}
.block-container {max-width:1280px; padding-top:1.6rem; padding-bottom:3rem;}
section[data-testid="stSidebar"] {background:#ffffff; border-right:1px solid #e5e7eb;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden; height:0;}

.hero {background:linear-gradient(135deg,#0b1220 0%,#14306b 55%,#2563eb 100%);
  padding:30px 36px; border-radius:22px; margin-bottom:22px;
  box-shadow:0 14px 34px rgba(15,23,42,.18); display:flex; justify-content:space-between; align-items:center; gap:24px;}
.hero h1 {color:#fff; font-size:2.3rem; font-weight:800; margin:0; letter-spacing:-.03em;}
.hero p {color:#cbd5e1; margin:.5rem 0 0 0; font-size:1.02rem; max-width:720px;}
.pill {display:inline-flex; align-items:center; gap:8px; margin-top:16px; padding:6px 14px; border-radius:999px;
  background:rgba(255,255,255,.12); color:#e0ecff; font-size:.8rem; font-weight:600;}
.dot {width:8px; height:8px; border-radius:50%; display:inline-block;}
.dot-green {background:#22c55e; box-shadow:0 0 0 4px rgba(34,197,94,.25);}
.dot-amber {background:#f59e0b; box-shadow:0 0 0 4px rgba(245,158,11,.25);}
.hero-icon {font-size:3.6rem; background:rgba(255,255,255,.10); width:96px; height:96px; border-radius:24px;
  display:flex; align-items:center; justify-content:center; flex-shrink:0;}

.step {display:flex; align-items:center; gap:10px; font-weight:700; color:#0f172a; font-size:1.05rem; margin-bottom:2px;}
.step-n {background:#2563eb; color:#fff; width:26px; height:26px; border-radius:50%; font-size:.8rem;
  display:inline-flex; align-items:center; justify-content:center;}
.step-d {color:#64748b; font-size:.85rem; margin:0 0 10px 36px;}

div[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:16px;}

div.stButton > button, div[data-testid="stFormSubmitButton"] button {
  border-radius:12px; height:3rem; font-weight:700; transition:all .15s ease;}
div[data-testid="stFormSubmitButton"] button {
  background:linear-gradient(135deg,#1d4ed8,#2563eb); color:#fff; border:none; box-shadow:0 6px 16px rgba(37,99,235,.35);}
div[data-testid="stFormSubmitButton"] button:hover {transform:translateY(-1px); box-shadow:0 10px 22px rgba(37,99,235,.42);}

.res {border-radius:20px; padding:26px; background:#fff; border:1px solid #dbe3ef;
  box-shadow:0 10px 30px rgba(15,23,42,.08); position:relative; overflow:hidden;}
.res-high {border-top:6px solid #dc2626;} .res-low {border-top:6px solid #16a34a;}
.res-label {color:#64748b; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em;}
.res-num {font-size:4rem; font-weight:800; color:#0f172a; line-height:1.05; margin-top:4px; letter-spacing:-.03em;}
.res-sub {color:#64748b; font-size:.9rem;}
.tag {display:inline-block; padding:6px 12px; border-radius:8px; font-size:.78rem; font-weight:700; margin:10px 6px 0 0;}
.tag-model {background:#eff6ff; color:#1d4ed8;}
.tag-demo {background:#fef3c7; color:#92400e;} .tag-live {background:#dcfce7; color:#166534;}
.verdict {margin-top:16px; padding:14px 16px; border-radius:12px; font-weight:600; font-size:.95rem; line-height:1.45;}
.v-high {background:#fef2f2; color:#991b1b; border:1px solid #fecaca;}
.v-low {background:#f0fdf4; color:#166534; border:1px solid #bbf7d0;}
.bar {position:relative; height:14px; border-radius:999px; margin:26px 0 8px 0;
  background:linear-gradient(90deg,#22c55e 0%,#facc15 45%,#ef4444 100%);}
.bar-fill-mask {position:absolute; top:0; right:0; bottom:0; background:#e5e7eb; border-radius:0 999px 999px 0; opacity:.78;}
.marker {position:absolute; top:-7px; width:4px; height:28px; background:#0f172a; border-radius:3px;}
.marker-prob {position:absolute; top:-26px; transform:translateX(-50%); font-size:.72rem; font-weight:700; color:#0f172a; white-space:nowrap;}
.marker-thr {position:absolute; top:24px; transform:translateX(-50%); font-size:.7rem; font-weight:600; color:#475569; white-space:nowrap;}
.scale {display:flex; justify-content:space-between; color:#64748b; font-size:.72rem; margin-top:24px;}
.watermark {position:absolute; top:22px; right:-38px; transform:rotate(35deg); background:#b45309; color:#fff;
  font-size:.7rem; font-weight:800; letter-spacing:.14em; padding:4px 44px;}
.empty {text-align:center; padding:42px 18px; color:#64748b;}
.empty .big {font-size:3rem;}
.note {font-size:.82rem; color:#475569; background:#f8fafc; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; margin-top:14px; line-height:1.55;}
.foot {text-align:center; color:#64748b; font-size:.8rem; margin-top:34px; padding-top:16px; border-top:1px solid #e5e7eb;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    model = st.selectbox("🤖 Modelo de IA", list(MODELS.keys()))
    st.markdown("### 🎚️ Umbral de clasificación")
    thr = st.slider("Umbral", 0.10, 0.90, 0.30, 0.05, format="%.2f", label_visibility="collapsed")
    st.caption(f"Se marca predicción positiva si la probabilidad ≥ **{thr:.0%}**. "
               "Bajar el umbral detecta más casos (más Recall) pero genera más falsas alarmas.")

    url, key = endpoint_for(model)
    with st.expander("☁️ Conexión Azure ML", expanded=not (url and key)):
        url = st.text_input("URL del endpoint", value=url)
        key = st.text_input("Llave", value=key, type="password")
        input_name = st.text_input("Input del servicio", value=get_setting("AZURE_INPUT_NAME", "WebServiceInput0"))
        dummy_target = st.toggle("Enviar TenYearCHD = 0", value=get_setting("INCLUDE_DUMMY_TARGET", "true").lower() == "true",
                                 help="Algunos endpoints de Designer esperan el esquema completo con la columna objetivo.")
    live = bool(url and key)
    st.divider()
    st.info("Prototipo académico. No sustituye la valoración de un profesional de la salud.")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
pill = ('<span class="dot dot-green"></span> Conectado a Azure ML' if live
        else '<span class="dot dot-amber"></span> Modo demostración (valores simulados)')
st.markdown(f"""
<div class="hero">
  <div>
    <h1>CardioRiesgo IA</h1>
    <p>Apoyo a la decisión clínica: estimación del riesgo de enfermedad coronaria a 10 años con modelos de
    Machine Learning entrenados con el Framingham Heart Study y desplegados en Azure.</p>
    <span class="pill">{pill}</span>
  </div>
  <div class="hero-icon">
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M32 57C11 41 4 29 4 20.5C4 12 10.5 6 18 6C24 6 29 9 32 14C35 9 40 6 46 6C53.5 6 60 12 60 20.5C60 29 53 41 32 57Z"
            fill="rgba(255,255,255,0.16)" stroke="#ffffff" stroke-width="3" stroke-linejoin="round"/>
      <polyline points="9,33 22,33 27,22 34,45 39,28 43,33 55,33" stroke="#7dd3fc" stroke-width="3.6"
                stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_models, tab_about = st.tabs(["🩺 Predicción", "📊 Modelos y métricas", "ℹ️ Acerca del proyecto"])

# ----------------------------------------------------------------------------
# Tab Predicción
# ----------------------------------------------------------------------------
with tab_pred:
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        b1, b2, _ = st.columns([1, 1, 1.4])
        b1.button("🧪 Cargar ejemplo", on_click=load_example, use_container_width=True)
        b2.button("🧹 Restablecer", on_click=reset_form, use_container_width=True)

        with st.form("paciente"):
            with st.container(border=True):
                st.markdown('<div class="step"><span class="step-n">1</span> Datos del paciente</div>'
                            '<div class="step-d">Información demográfica y hábitos</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.number_input("Edad (años)", 18, 100, step=1, key="age")
                c2.selectbox("¿Tiene diabetes?", ["No", "Sí"], key="diabetes")
                c3, c4 = st.columns(2)
                c3.selectbox("¿Fuma actualmente?", ["No", "Sí"], key="smoker")
                c4.number_input("Cigarrillos por día", 0, 70, step=1, key="cigs",
                                help="Debe ser 0 si no fuma.")

            with st.container(border=True):
                st.markdown('<div class="step"><span class="step-n">2</span> Presión arterial</div>'
                            '<div class="step-d">Valores en mmHg</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.number_input("Sistólica (sysBP)", 80, 260, step=1, key="sysBP")
                c2.number_input("Diastólica (diaBP)", 40, 160, step=1, key="diaBP")

            with st.container(border=True):
                st.markdown('<div class="step"><span class="step-n">3</span> Variables clínicas</div>'
                            '<div class="step-d">Colesterol y glucosa en mg/dL</div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.number_input("Colesterol total", 100, 600, step=1, key="totChol")
                c2.number_input("Glucosa", 40, 400, step=1, key="glucose")
                c3.number_input("IMC (BMI)", 15.0, 60.0, step=0.1, format="%.1f", key="BMI")

            go = st.form_submit_button("🧠 Calcular estimación", use_container_width=True)

    with right:
        ss = st.session_state
        if go:
            errors = []
            if ss.smoker == "No" and ss.cigs > 0:
                errors.append("Marcaste **no fumador** pero con cigarrillos por día > 0.")
            if ss.smoker == "Sí" and ss.cigs == 0:
                errors.append("Marcaste **fumador** pero con 0 cigarrillos por día.")
            if ss.diaBP >= ss.sysBP:
                errors.append("La presión diastólica debe ser menor que la sistólica.")
            if errors:
                st.session_state.pop("result", None)
                for e in errors:
                    st.error(e)
            else:
                record = {"age": int(ss.age), "sysBP": float(ss.sysBP), "diaBP": float(ss.diaBP),
                          "totChol": float(ss.totChol), "currentSmoker": 1 if ss.smoker == "Sí" else 0,
                          "cigsPerDay": int(ss.cigs), "glucose": float(ss.glucose),
                          "BMI": float(ss.BMI), "diabetes": 1 if ss.diabetes == "Sí" else 0}
                record = {k: record[k] for k in FEATURES}
                try:
                    if live:
                        to_send = dict(record)
                        if dummy_target:
                            to_send["TenYearCHD"] = 0
                        with st.spinner("Consultando el modelo en Azure…"):
                            payload, raw = call_endpoint(url, key, input_name, to_send)
                            parsed = parse_response(raw)
                        ss["result"] = dict(mode="azure", model=model, prob=parsed["prob"],
                                            label=parsed["label"], record=record, payload=payload, raw=raw)
                    else:
                        ss["result"] = dict(mode="demo", model=model, prob=DEMO_VALUES[model],
                                            label=None, record=record, payload=None, raw=None)
                except requests.HTTPError as e:
                    ss.pop("result", None); st.error(http_hint(e))
                except requests.RequestException as e:
                    ss.pop("result", None); st.error(f"No se pudo conectar con el endpoint: {e}")
                except (KeyError, ValueError, IndexError, StopIteration) as e:
                    ss.pop("result", None); st.error(f"No pude interpretar la respuesta: {e}")

        res = ss.get("result")
        if not res:
            st.markdown('<div class="res"><div class="empty"><div class="big">📋</div>'
                        '<b>Aún no hay resultado</b><br>Completa los datos y pulsa '
                        '<i>Calcular estimación</i>.</div></div>', unsafe_allow_html=True)
        else:
            p = res["prob"]
            positive = p >= thr          # la clasificación usa el umbral de la barra lateral
            cls = "res-high" if positive else "res-low"
            vcls = "v-high" if positive else "v-low"
            verdict = ("🔴 <b>Predicción positiva</b>: la probabilidad ({:.1%}) alcanza o supera el umbral ({:.0%}). "
                       "Sugiere priorizar la valoración clínica.").format(p, thr) if positive else \
                      ("🟢 <b>Predicción negativa</b>: la probabilidad ({:.1%}) está por debajo del umbral ({:.0%}). "
                       "No descarta riesgo.").format(p, thr)
            tag_mode = ('<span class="tag tag-live">● Azure ML en vivo</span>' if res["mode"] == "azure"
                        else '<span class="tag tag-demo">⚠ Valor simulado</span>')
            watermark = '<div class="watermark">SIMULADO</div>' if res["mode"] == "demo" else ""
            pp, tt = min(max(p, 0), 1) * 100, thr * 100
            st.markdown(f"""
<div class="res {cls}">
  {watermark}
  <div class="res-label">Probabilidad estimada por el modelo</div>
  <div class="res-num">{p:.1%}</div>
  <div class="res-sub">Evento: enfermedad coronaria a 10 años (TenYearCHD)</div>
  <span class="tag tag-model">🤖 {res['model']}</span>{tag_mode}
  <div class="bar">
    <div class="bar-fill-mask" style="left:{pp}%"></div>
    <div class="marker" style="left:calc({pp}% - 2px)"></div>
    <div class="marker-prob" style="left:{pp}%">Paciente</div>
    <div class="marker" style="left:calc({tt}% - 2px); background:#64748b; height:20px; top:-3px; opacity:.8"></div>
    <div class="marker-thr" style="left:{tt}%">Umbral</div>
  </div>
  <div class="scale"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
  <div class="verdict {vcls}">{verdict}</div>
  <div class="note">La probabilidad proviene de un modelo experimental entrenado con SMOTE: sirve para
  <b>priorizar</b>, no equivale al riesgo clínico absoluto ni a un diagnóstico.</div>
</div>
""", unsafe_allow_html=True)
            if res["mode"] == "demo":
                st.warning("Estos valores son **simulados** y no dependen de los datos ingresados. "
                           "Configura el endpoint en la barra lateral para ver la predicción real.")
            with st.expander("🔎 Ver datos enviados / respuesta del endpoint"):
                st.json(res["payload"] if res["payload"] else res["record"], expanded=False)
                if res["raw"]:
                    st.caption(f"Etiqueta del modelo (umbral interno de Azure): {res['label']}")
                    st.json(res["raw"], expanded=False)

# ----------------------------------------------------------------------------
# Tab Modelos
# ----------------------------------------------------------------------------
with tab_models:
    st.subheader("Comparación de modelos")
    mpath = Path(__file__).parent / "metrics.json"
    if not mpath.exists():
        st.info("Aún no hay métricas cargadas. Crea un archivo `metrics.json` junto a `interfaz.py` con los valores reales de **Evaluate Model** "
                "de Azure ML Designer. Formato: lista de objetos como "
                "`{\"model\": \"Logistic Regression\", \"Accuracy\": 0.0, \"Recall\": 0.0, \"F1\": 0.0, \"AUC\": 0.0}`. "
                "Esta app no inventa métricas.")
        st.dataframe(pd.DataFrame({"Modelo": list(MODELS.keys())}), hide_index=True, use_container_width=True)
    else:
        dfm = pd.DataFrame(json.loads(mpath.read_text(encoding="utf-8")))
        st.dataframe(dfm, hide_index=True, use_container_width=True)
        metric_cols = [c for c in dfm.columns if c != "model"]
        chart = dfm.set_index("model")[metric_cols].apply(pd.to_numeric, errors="coerce")
        st.bar_chart(chart.T)
        st.caption("Con clases desbalanceadas, el accuracy no debe ser la métrica principal: "
                   "prioriza Recall, F1 y AUC.")

# ----------------------------------------------------------------------------
# Tab Acerca
# ----------------------------------------------------------------------------
with tab_about:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("### 📊 Dataset")
            st.write("**Framingham Heart Study** · variable objetivo `TenYearCHD` (clasificación binaria).")
            st.markdown("### 🧾 Variables que recibe el modelo")
            st.code(", ".join(FEATURES))
    with c2:
        with st.container(border=True):
            st.markdown("### 🤖 Modelos del pipeline")
            for m in MODELS:
                st.markdown(f"- {m}")
            st.markdown("### ⚖️ Uso responsable")
            st.markdown("- Herramienta de **apoyo**, la decisión es del profesional.\n"
                        "- Un **falso negativo** (paciente de riesgo clasificado como bajo) es el error más costoso.\n"
                        "- Datos históricos de EE. UU.: pueden no generalizar a otras poblaciones.\n"
                        "- No ingreses datos identificables de pacientes reales.")

st.markdown('<div class="foot"><b>CardioRiesgo IA</b> · Prototipo académico de Inteligencia Artificial para apoyo '
            'a la decisión clínica</div>', unsafe_allow_html=True)