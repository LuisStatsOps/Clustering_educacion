import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import squarify
import networkx as nx
from io import BytesIO

# -----------------------------
# Configuración de la app
# -----------------------------
st.set_page_config(page_title="Higher education in Ecuador", layout="wide")
st.title("Higher education in Ecuador")

# -----------------------------
# Carga de datos (cache)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")

EXCEL_PATH = "base_matricula_datosabiertos.xlsx"

try:
    df_matricula = load_data(EXCEL_PATH)
except Exception as e:
    st.error(f"No se pudo cargar el archivo: {EXCEL_PATH}. Verifica que esté en el repo.")
    st.exception(e)
    st.stop()

df_matriculas = df_matricula[df_matricula["AÑO"] == 2022].copy()

# -----------------------------
# Limpieza / normalización
# -----------------------------
df_matriculas["MODALIDAD"] = df_matriculas["MODALIDAD"].replace(["HIBRIDA", "DUAL"], "SEMIPRESENCIAL")
df_matriculas["NIVEL_FORMACIÓN"] = df_matriculas["NIVEL_FORMACIÓN"].replace(
    ["TERCER NIVEL O PREGRADO", "CUARTO NIVEL O POSGRADO"],
    ["PREGRADO", "POSGRADO"],
)
df_matriculas["CAMPO_AMPLIO"] = df_matriculas["CAMPO_AMPLIO"].replace(
    [
        "CIENCIAS SOCIALES, PERIODISMO, INFORMACION Y DERECHO",
        "AGRICULTURA, SILVICULTURA, PESCA Y VETERINARIA",
        "CIENCIAS NATURALES, MATEMATICAS Y ESTADISTICA",
        "INGENIERIA, INDUSTRIA Y CONSTRUCCION",
        "TECNOLOGIAS DE LA INFORMACION Y LA COMUNICACION (TIC)",
    ],
    [
        "CIENCIAS SOCIALES Y DERECHO",
        "AGRICULTURA Y VETERINARIA",
        "CIENCIAS NATURALES Y MATEMATICAS",
        "INGENIERIA E INDUSTRIA",
        "TECNOLOGIAS DE LA INFORMACION",
    ],
)
df_matriculas["TIPO_FINANCIAMIENTO"] = df_matriculas["TIPO_FINANCIAMIENTO"].replace(
    ["PARTICULAR COFINANCIADA", "PARTICULAR AUTOFINANCIADA"], "PARTICULAR"
)

df_matriculas = df_matriculas[
    (df_matriculas["CAMPO_AMPLIO"] != "NO_REGISTRA")
    & (df_matriculas["PROVINCIA_RESIDENCIA"] != "NO_REGISTRA")
    & (df_matriculas["PROVINCIA_RESIDENCIA"] != "ZONAS NO DELIMITADAS")
    & (df_matriculas["NIVEL_FORMACIÓN"] != "TERCER NIVEL TECNICO-TECNOLOGICO SUPERIOR")
].copy()

# -----------------------------
# Helpers: exportar plotly a PNG (opcional)
# -----------------------------
def plotly_png_bytes(fig) -> bytes | None:
    """
    Intenta generar PNG bytes desde un fig de plotly.
    Si no hay kaleido (o falla), devuelve None.
    """
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None

# -----------------------------
# 1) SUNBURST: Programas/Carreras (nunique CODIGO_CARRERA)
# -----------------------------
st.subheader("Programs/Courses")

color_sequence = ["rgba(255, 215, 0, 0.3)", "rgba(128, 128, 128, 0.5)", "white"]

sunburst_prog = (
    df_matriculas.groupby(["TIPO_FINANCIAMIENTO", "NIVEL_FORMACIÓN", "MODALIDAD", "CAMPO_AMPLIO"])["CODIGO_CARRERA"]
    .nunique()
    .reset_index()
    .rename(columns={"CODIGO_CARRERA": "Cantidad de Carreras"})
)

top_3_campo = sunburst_prog.groupby("CAMPO_AMPLIO")["Cantidad de Carreras"].sum().nlargest(3).index
sunburst_prog["CAMPO_AMPLIO"] = sunburst_prog["CAMPO_AMPLIO"].apply(lambda x: x if x in top_3_campo else "Otros")

sunburst_prog = (
    sunburst_prog.groupby(["TIPO_FINANCIAMIENTO", "NIVEL_FORMACIÓN", "MODALIDAD", "CAMPO_AMPLIO"])["Cantidad de Carreras"]
    .sum()
    .reset_index()
)
sunburst_prog["Total"] = "CARRERAS/PROGRAMAS"

fig_prog = px.sunburst(
    sunburst_prog,
    path=["Total", "TIPO_FINANCIAMIENTO", "NIVEL_FORMACIÓN", "MODALIDAD", "CAMPO_AMPLIO"],
    values="Cantidad de Carreras",
    color="TIPO_FINANCIAMIENTO",
    color_discrete_map={
        "PUBLICA": "#D4AF37",
        "PARTICULAR": "#D3D3D3",
    },     
    title="Programs/Courses",
)

fig_prog.update_traces(
    textinfo="label+percent entry",
    hoverinfo="label+percent entry",
    insidetextorientation="radial",
    textfont_size=12,
    marker=dict(line=dict(color="white", width=1), opacity=0.95),
   
)

total_carreras = int(sunburst_prog["Cantidad de Carreras"].sum())
fig_prog.update_layout(
    title_x=0.5,
    annotations=[dict(text=f"{total_carreras}", x=0.5, y=0.48, showarrow=False, font=dict(size=24, color="black"))],
    paper_bgcolor="black",
    font=dict(size=20),
    margin=dict(t=50, l=10, r=10, b=10),
)

st.plotly_chart(fig_prog, use_container_width=True)

png1 = plotly_png_bytes(fig_prog)
if png1:
    st.download_button(
        "Descargar PNG (Programs/Courses)",
        data=png1,
        file_name="sunburst_oferta_carreras.png",
        mime="image/png",
    )
else:
    st.info("Exportar a PNG requiere kaleido. Si lo necesitas, agrega `kaleido` en requirements.txt.")

# -----------------------------
# 2) SUNBURST: Estudiantes (sum tot)
# -----------------------------
st.subheader("Students distribution")

sunburst_stu = (
    df_matriculas.groupby(["TIPO_FINANCIAMIENTO", "NIVEL_FORMACIÓN", "MODALIDAD", "CAMPO_AMPLIO"])["tot"]
    .sum()
    .reset_index()
    .rename(columns={"tot": "Total Estudiantes"})
)
sunburst_stu["Total"] = "ESTUDIANTES"

fig_stu = px.sunburst(
    sunburst_stu,
    path=["Total", "TIPO_FINANCIAMIENTO", "NIVEL_FORMACIÓN", "MODALIDAD", "CAMPO_AMPLIO"],
    values="Total Estudiantes",
    color="TIPO_FINANCIAMIENTO",
    color_discrete_map={
        "PUBLICA": "#D4AF37",
        "PARTICULAR": "#D3D3D3",
    },
    title="Students distribution",
)

fig_stu.update_traces(
    textinfo="label+percent entry",
    hoverinfo="label+percent entry",
    insidetextorientation="radial",
    textfont_size=12,
)

total_est = int(sunburst_stu["Total Estudiantes"].sum())
fig_stu.update_layout(
    title_x=0.5,
    annotations=[dict(text=f"{total_est}", x=0.5, y=0.48, showarrow=False, font=dict(size=24, color="black"))],
    paper_bgcolor="black",
    font=dict(size=20),
    margin=dict(t=50, l=10, r=10, b=10),
)

st.plotly_chart(fig_stu, use_container_width=True)

png2 = plotly_png_bytes(fig_stu)
if png2:
    st.download_button(
        "Descargar PNG (Students distribution)",
        data=png2,
        file_name="sunburst_estudiantes.png",
        mime="image/png",
    )

# -----------------------------
# 3) TREEMAP: Top 10 provincias + Otros (matplotlib)
# -----------------------------
st.subheader("Estudiantes por Provincia de residencia (Top 10)")

prov_tot = df_matriculas.groupby("PROVINCIA_RESIDENCIA")["tot"].sum().reset_index()
top_10 = prov_tot.nlargest(10, "tot")

otros = pd.DataFrame(
    {"PROVINCIA_RESIDENCIA": ["Otros"], "tot": [prov_tot["tot"].sum() - top_10["tot"].sum()]}
)

treemap_data = pd.concat([top_10, otros], ignore_index=True).sort_values(by="tot", ascending=False)

colors = []
for v in treemap_data["tot"]:
    if v > 100000:
        colors.append("#D4AF37")
    elif 30000 <= v <= 100000:
        colors.append("#808080")
    else:
        colors.append("#D3D3D3")

fig_tm = plt.figure(figsize=(12, 8))
squarify.plot(
    sizes=treemap_data["tot"],
    label=treemap_data["PROVINCIA_RESIDENCIA"] + "\n" + treemap_data["tot"].apply(lambda x: f"{int(x):,}"),
    color=colors,
    alpha=0.8,
    text_kwargs={"fontsize": 12, "color": "black"},
)
plt.title("Estudiantes por Provincia de residencia (Top 10)", fontsize=16)
plt.axis("off")
st.pyplot(fig_tm, clear_figure=True)

# Descarga del treemap como PNG (en memoria)
buf = BytesIO()
fig_tm.savefig(buf, format="png", dpi=200, bbox_inches="tight")
buf.seek(0)
st.download_button(
    "Descargar PNG (Treemap provincias)",
    data=buf.getvalue(),
    file_name="top_10_estudiantes_provincia_treemap.png",
    mime="image/png",
)
plt.close(fig_tm)

# -----------------------------
# 4) GRAFO: Provincia -> Campos amplios (networkx)
# -----------------------------
st.subheader("Grafo de distribución de estudiantes por provincia")

df_grouped = df_matriculas.groupby(["PROVINCIA_RESIDENCIA", "CAMPO_AMPLIO"])["tot"].sum().reset_index()
provincias = sorted(df_grouped["PROVINCIA_RESIDENCIA"].unique().tolist())
secondary_nodes = sorted(df_grouped["CAMPO_AMPLIO"].unique().tolist())

tot_values_by_province = {
    prov: (
        df_grouped[df_grouped["PROVINCIA_RESIDENCIA"] == prov]
        .set_index("CAMPO_AMPLIO")["tot"]
        .reindex(secondary_nodes, fill_value=0)
        .tolist()
    )
    for prov in provincias
}

default_index = min(13, max(0, len(provincias) - 1))
provincia_sel = st.selectbox("Province:", provincias, index=default_index)

def split_text_with_tot(text: str, tot_value: float, max_length: int = 15) -> str:
    words = text.split()
    half = len(words) // 2
    if len(text) > max_length and len(words) > 1:
        label = "\n".join([" ".join(words[:half]), " ".join(words[half:])])
    else:
        label = text
    return f"{label}\nEstudiantes: {int(tot_value):,}"

def update_graph(provincia: str):
    tot_values = tot_values_by_province[provincia]

    G = nx.Graph()
    G.add_node(provincia)

    for node in secondary_nodes:
        G.add_node(node)
        G.add_edge(provincia, node)

    node_sizes = [4500] + [max(50, v * 0.5) for v in tot_values]
    node_colors = ["#1A237E"] + ["#D3D3D3"] * len(secondary_nodes)

    labels = {node: split_text_with_tot(node, tot_values[i]) for i, node in enumerate(secondary_nodes)}
    labels[provincia] = provincia

    fig_g = plt.figure(figsize=(14, 12))
    pos = nx.shell_layout(G, [list(G.neighbors(provincia)), [provincia]])
    nx.draw(G, pos, node_size=node_sizes, node_color=node_colors, edge_color="gray")

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color="darkblue")
    nx.draw_networkx_labels(G, pos, labels={provincia: provincia}, font_size=12, font_weight="bold", font_color="white")

    plt.title(f"Grafo de distribución de estudiantes en {provincia}", fontsize=20)
    st.pyplot(fig_g, clear_figure=True)
    plt.close(fig_g)

update_graph(provincia_sel)
