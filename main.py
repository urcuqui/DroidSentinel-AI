import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================
# 1. Entrenamiento del modelo (cacheado)
# ============================

@st.cache_resource
def train_model(csv_path: str = "train.csv"):
    # Cargar dataset
    df = pd.read_csv(csv_path)

    # La columna única se separa en columnas de permisos
    data = df[df.columns[0]].str.split(';', expand=True)

    # Encabezados reales (permisos + type)
    header = df.columns[0].split(';')
    data.columns = header

    # Convertir a numérico
    data = data.apply(pd.to_numeric, errors='coerce')

    # Labels y features
    y = data["type"]
    X = data.drop(columns=["type"])

    # Train / test (para el entrenamiento básico)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
        )

    # Modelo
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Importancia de características para elegir las más relevantes
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_perms = importances.sort_values(ascending=False).head(20).index.tolist()

    return model, X.columns.tolist(), top_perms

# ============================
# 2. Interfaz Streamlit
# ============================

st.set_page_config(page_title="Clasificador de Malware Android", page_icon="🤖", layout="wide")

st.title("🤖 Clasificador de Malware Android basado en Permisos")
st.write(
    "Esta app entrena un modelo de **RandomForest** sobre tu dataset de permisos "
    "y permite enviar nuevas combinaciones de permisos para predecir si la app es **benigna (0)** o **malware (1)**."
)

# Entrenar / cargar modelo
with st.spinner("Entrenando modelo y cargando permisos..."):
    model, all_features, top_perms = train_model("train.csv")

st.success("Modelo entrenado y listo para usar ✅")

st.sidebar.header("Configuración de entrada")
st.sidebar.write("Selecciona los permisos que la app solicita (1 = accedido).")

st.sidebar.subheader("Permisos más relevantes (Top 20)")
selected_perms = []
for perm in top_perms:
    if st.sidebar.checkbox(perm, value=False):
        selected_perms.append(perm)

st.sidebar.subheader("Otros parámetros")
default_zeros = st.sidebar.checkbox(
    "Poner todos los permisos no seleccionados en 0 (recomendado)", value=True
)

st.markdown("### 🧩 Entrada de ejemplo")

st.write(
    "El modelo utiliza **todos los permisos** del dataset internamente, "
    "pero aquí controlas principalmente los **Top 20 más importantes**. "
    "Los demás se asumirán en 0 (sin permiso) si dejas activada la opción de arriba."
)

# ============================
# 3. Construcción del vector de características
# ============================

# Inicializar todos en 0 o NaN (pero lo normal es 0)
if default_zeros:
    input_vector = {feat: 0 for feat in all_features}
else:
    input_vector = {feat: 0 for feat in all_features}  # igual 0, pero podrías modificar luego

# Activar los permisos seleccionados
for perm in selected_perms:
    if perm in input_vector:
        input_vector[perm] = 1

# Crear DataFrame de una sola instancia
X_new = pd.DataFrame([input_vector])[all_features]  # respetar el orden

st.code(
    "Permisos activos (1) en esta instancia:\n" +
    "\n".join(f"- {p}" for p in selected_perms) if selected_perms else
    "No has seleccionado ningún permiso (todos en 0).",
    language="text"
)

# ============================
# 4. Predicción
# ============================

if st.button("🔮 Predecir tipo de aplicación"):
    # Predicción
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]

    etiqueta = "Malware (1)" if pred == 1 else "Benigna (0)"
    prob_malware = proba[1]
    prob_benigno = proba[0]

    if pred == 1:
        st.error(f"Resultado: **{etiqueta}**")
    else:
        st.success(f"Resultado: **{etiqueta}**")

    st.markdown("### 📊 Probabilidades")
    st.write(f"- Probabilidad de **Benigno (0)**: `{prob_benigno:.3f}`")
    st.write(f"- Probabilidad de **Malware (1)**: `{prob_malware:.3f}`")

    st.markdown("### 🔎 Interpretación rápida")
    if prob_malware > 0.8:
        st.write("⚠️ Alta probabilidad de que la app sea **maliciosa**.")
    elif prob_malware > 0.5:
        st.write("⚠️ La app tiene **cierta sospecha** de comportamiento malicioso.")
    else:
        st.write("✅ La app parece **más probable que sea benigna**, según estos permisos.")

else:
    st.info("Configura los permisos en la barra lateral y pulsa **“Predecir tipo de aplicación”** para ver el resultado.")

# ============================
# 5. Info adicional
# ============================

with st.expander("Ver lista completa de permisos usados por el modelo"):
    st.write(pd.Series(all_features).to_frame("Permisos / Features"))
