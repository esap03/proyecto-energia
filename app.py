import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import io
import xlsxwriter


# Configurar layout ancho
st.set_page_config(layout="wide")

# Leer datos del CSV
df = pd.read_csv('datos.csv')

if 'semanas' not in st.session_state:
    st.session_state.semanas = df['Semana'].tolist()
    st.session_state.hidraulica = df['Hidráulica'].tolist()
    st.session_state.termica = df['Térmica'].tolist()

# Función para generar nueva etiqueta de semana automáticamente
def generar_nueva_semana(ultima_semana):
    dias, mes = ultima_semana.split()
    dia_fin = int(dias.split('-')[1])
    meses = {'Ene': 1, 'Feb': 2, 'Mar': 3, 'Abr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Ago': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dic': 12}
    num_mes = meses.get(mes)
    fecha_fin = datetime.date(2025, num_mes, dia_fin)
    nueva_inicio = fecha_fin + datetime.timedelta(days=1)
    nueva_fin = nueva_inicio + datetime.timedelta(days=6)
    mes_corto = nueva_fin.strftime('%b').capitalize()
    if mes_corto.endswith('.'):
        mes_corto = mes_corto[:-1]
    return f"{nueva_inicio.day:02d}-{nueva_fin.day:02d} {mes_corto}"

st.title("Generación de Energía Semanal")

# Preparar datos (últimas 12 semanas)
ultimas_semanas = st.session_state.semanas[-12:]
ultimas_hidraulica = st.session_state.hidraulica[-12:]
ultimas_termica = st.session_state.termica[-12:]
ultimas_total = [h + t for h, t in zip(ultimas_hidraulica, ultimas_termica)]

# Calcular totales y promedios
total_hidraulica = sum(ultimas_hidraulica)
total_termica = sum(ultimas_termica)
total_general = sum(ultimas_total)

promedio_hidraulica = total_hidraulica / len(ultimas_hidraulica)
promedio_termica = total_termica / len(ultimas_termica)
promedio_general = total_general / len(ultimas_total)

# Crear DataFrame resumen
resumen_df = pd.DataFrame({
    "Semana": ultimas_semanas,
    "Hidráulica": ultimas_hidraulica,
    "Térmica": ultimas_termica,
    "Total": ultimas_total
})

fila_total = pd.DataFrame({
    "Semana": ["TOTAL"],
    "Hidráulica": [total_hidraulica],
    "Térmica": [total_termica],
})

fila_promedio = pd.DataFrame({
    "Semana": ["PROMEDIO"],
    "Hidráulica": [promedio_hidraulica],
    "Térmica": [promedio_termica],
    "Total": [promedio_general]
})

resumen_final = pd.concat([resumen_df, fila_promedio], ignore_index=True)

st.subheader("Agregar / Corregir / Eliminar Semana")

col_a, col_b, col_c, col_d = st.columns(4)

# -------- AGREGAR SEMANA --------
with col_a:
    if "mostrar_agregar" not in st.session_state:
        st.session_state.mostrar_agregar = False

    if st.session_state.mostrar_agregar:
        if st.button("➖ AGREGAR SEMANA", key="toggle_ocultar_agregar"):
            st.session_state.mostrar_agregar = False
            st.rerun()
    else:
        if st.button("➕ AGREGAR SEMANA", key="toggle_mostrar_agregar"):
            st.session_state.mostrar_agregar = True
            st.rerun()

    if st.session_state.mostrar_agregar:
        with st.form("agregar_semana"):
            nueva_hidraulica = st.number_input("Energía Hidráulica (MWh)", min_value=0.0, value=0.0, key="input_hidraulica")
            nueva_termica = st.number_input("Energía Térmica (MWh)", min_value=0.0, value=0.0, key="input_termica")
            agregar = st.form_submit_button("Agregar Semana")

        if agregar:
            nueva_semana = generar_nueva_semana(st.session_state.semanas[-1])
            st.session_state.semanas.append(nueva_semana)
            st.session_state.hidraulica.append(nueva_hidraulica)
            st.session_state.termica.append(nueva_termica)

            pd.DataFrame({
                'Semana': st.session_state.semanas,
                'Hidráulica': st.session_state.hidraulica,
                'Térmica': st.session_state.termica
            }).to_csv('datos.csv', index=False)

            st.success("✅ Semana agregada correctamente.")
            st.session_state.mostrar_agregar = False
            st.rerun()


# -------- CORREGIR SEMANA --------
with col_b:
    if "mostrar_corregir" not in st.session_state:
        st.session_state.mostrar_corregir = False

    if st.session_state.mostrar_corregir:
        if st.button("➖ CORREGIR SEMANA", key="toggle_ocultar_corregir"):
            st.session_state.mostrar_corregir = False
            st.rerun()
    else:
        if st.button("➕ CORREGIR SEMANA", key="toggle_mostrar_corregir"):
            st.session_state.mostrar_corregir = True
            st.rerun()

    if st.session_state.mostrar_corregir:
        with st.form("form_corregir"):
            indices_reales = list(range(len(st.session_state.semanas)))
            opciones = [f"{i}: {st.session_state.semanas[i]}" for i in indices_reales]

            indice_str = st.selectbox("Selecciona la semana para corregir", opciones, index=len(opciones) - 1, key="indice_corregir")
            indice = int(indice_str.split(":")[0])

            nuevo_h = st.number_input("Nuevo Hidráulica (MWh)", min_value=0.0, value=st.session_state.hidraulica[indice], key="nuevo_hidraulica")
            nuevo_t = st.number_input("Nuevo Térmica (MWh)", min_value=0.0, value=st.session_state.termica[indice], key="nuevo_termica")

            corregir = st.form_submit_button("Guardar Corrección")

        if corregir:
            st.session_state.hidraulica[indice] = nuevo_h
            st.session_state.termica[indice] = nuevo_t

            pd.DataFrame({
                'Semana': st.session_state.semanas,
                'Hidráulica': st.session_state.hidraulica,
                'Térmica': st.session_state.termica
            }).to_csv('datos.csv', index=False)

            st.success("✅ Semana corregida correctamente.")
            st.session_state.mostrar_corregir = False
            st.rerun()


# -------- ELIMINAR SEMANA --------
with col_c:
    if "mostrar_eliminar" not in st.session_state:
        st.session_state.mostrar_eliminar = False

    if st.session_state.mostrar_eliminar:
        if st.button("➖ ELIMINAR SEMANA", key="toggle_ocultar_eliminar"):
            st.session_state.mostrar_eliminar = False
            st.rerun()
    else:
        if st.button("➕ ELIMINAR SEMANA", key="toggle_mostrar_eliminar"):
            st.session_state.mostrar_eliminar = True
            st.rerun()

    if st.session_state.mostrar_eliminar:
        with st.form("form_eliminar"):
            indices_reales = list(range(len(st.session_state.semanas)))
            opciones = [f"{i}: {st.session_state.semanas[i]}" for i in indices_reales]

            indice_str = st.selectbox("Selecciona la semana para eliminar", opciones, index=len(opciones) - 1, key="indice_eliminar")
            indice = int(indice_str.split(":")[0])

            eliminar = st.form_submit_button("Eliminar Semana")

        if eliminar:
            st.session_state.semanas.pop(indice)
            st.session_state.hidraulica.pop(indice)
            st.session_state.termica.pop(indice)

            pd.DataFrame({
                'Semana': st.session_state.semanas,
                'Hidráulica': st.session_state.hidraulica,
                'Térmica': st.session_state.termica
            }).to_csv('datos.csv', index=False)

            st.success("✅ Semana eliminada correctamente.")
            st.session_state.mostrar_eliminar = False
            st.rerun()





    semanas_np = np.array(ultimas_semanas)
    hidraulica_np = np.array(ultimas_hidraulica)
    termica_np = np.array(ultimas_termica)

    fig, ax = plt.subplots(figsize=(13, 6))
    x_pos = np.arange(len(semanas_np))
    bar_width = 0.7

    ax.bar(x_pos, hidraulica_np, label='E. Hidráulica (MWh)', color='blue', width=bar_width)
    ax.bar(x_pos, termica_np, bottom=hidraulica_np, label='E. Térmica (MWh)', color='red', width=bar_width)

    for i, (h, t) in enumerate(zip(hidraulica_np, termica_np)):
        ax.text(i, h / 2, f'{h:.1f}', ha='center', va='center', color='white', fontsize=8)
        ax.text(i, h + t / 2, f'{t:.1f}', ha='center', va='center', color='white', fontsize=8)

    ax.axhline(promedio_hidraulica, color='gray', linestyle='--', linewidth=2, label=f'Prom. Hidráulico ({promedio_hidraulica:.1f} MWh)')
    ax.axhline(promedio_termica, color='orange', linestyle='--', linewidth=2, label=f'Prom. Térmico ({promedio_termica:.1f} MWh)')

    ax.set_title('GENERACIÓN DE ENERGÍA POR SEMANA')
    ax.set_ylabel('Energía (MWh)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(semanas_np, rotation=45)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0, 1, 1])




    # Crear Excel en memoria
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet('ENERGIA SEMANAL')

    # Formatos
    bold_center = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
    center = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
    title_format = workbook.add_format({'bold': True, 'bg_color': '#B7DEE8', 'align': 'center', 'valign': 'vcenter', 'border': 1})
    note_title_format = workbook.add_format({'bold': True, 'bg_color': '#00B0F0', 'font_color': 'white', 'align': 'center', 'valign': 'vcenter'})
    note_format = workbook.add_format({'text_wrap': True, 'bg_color': '#FFE699', 'border': 1})

    # Encabezado principal
    worksheet.merge_range('B2:O2', 'Generación de energía', title_format)
    worksheet.merge_range('B3:O3', 'ENERGIA EN (MWH )', title_format)

    # Encabezados de semanas
    headers = resumen_df['Semana'].tolist() + ['PROM']
    worksheet.write('B4', 'Semana', bold_center)
    worksheet.write_row('C4', headers, bold_center)

    # Datos
    worksheet.write('B5', 'E. HIDRAULICA', bold_center)
    worksheet.write_row('C5', resumen_df['Hidráulica'].tolist() + [round(promedio_hidraulica, 1)], center)

    worksheet.write('B6', 'E. TERMICA', bold_center)
    worksheet.write_row('C6', resumen_df['Térmica'].tolist() + [round(promedio_termica, 1)], center)

    worksheet.write('B7', 'E. TOTAL', bold_center)
    worksheet.write_row('C7', [round(h + t, 1) for h, t in zip(resumen_df['Hidráulica'], resumen_df['Térmica'])] + [round(promedio_general, 1)], center)

    # Insertar gráfico como imagen (ajustando el tamaño)
    chart_buffer = io.BytesIO()
    fig.savefig(chart_buffer, format='png', dpi=300)  # puedes aumentar dpi para mayor calidad
    chart_buffer.seek(0)

    worksheet.insert_image(
        'B10',                 # posición inicial
        'grafico.png',        # nombre referencial
        {
            'image_data': chart_buffer,
            'x_scale': 0.7,   # escala horizontal (1.0 es tamaño original)
            'y_scale': 0.7    # escala vertical (1.0 es tamaño original)
        }
    )


    # Lista de notas
    notes = [
        "*La MCH genera 80.3 MWh de energía hidráulica con 166 horas de trabajo efectivo.",
        "*La máxima demanda fue de _____ kW con la hidroeléctrica.",
        "*Caudal promedio de la quebrada Cuncush fue de _____ m³/s.",
        "*Los grupos electrogenos durante la semana generaron _____ MWh de energía térmica en paralelo con la MCH.",
        "*Hidroeléctrica operativo.",
        "*Grupos electrógenos operativos.",
        "*Consumo de combustible durante la semana, _____ galones."
    ]

    # Formato del título de notas
    note_title_format = workbook.add_format({
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'font_size': 14,
        'fg_color': '#00B0F0',
        'border': 1
    })

    # Formato de cada nota (ajustando texto y centrado)
    note_format = workbook.add_format({
        'text_wrap': True,
        'align': 'center',
        'align': 'left',         # alineación a la izquierda
        'valign': 'vcenter',
        'font_size': 11,
        'fg_color': '#FFD966',
        'border': 1
    })

    # Ajustar ancho de columna P
    worksheet.set_column('P:P', 50)

    # Ajustar altura de filas (P12 en adelante → fila índice 11)
    worksheet.set_row(11, 30)  # fila 12 para el título
    for i in range(len(notes)):
        worksheet.set_row(12 + i, 30)  # filas 13, 14, 15...

    # Escribir título en P12
    worksheet.write('P12', 'Notas de generación de energía:', note_title_format)

    # Escribir cada nota en P13, P14, P15, ...
    for i, note in enumerate(notes):
        worksheet.write(f'P{13 + i}', note, note_format)

    # Ajustar anchos de columna
    worksheet.set_column('B:B', 15)
    worksheet.set_column('C:O', 9)
    worksheet.set_column('R:V', 9)

    workbook.close()
    output.seek(0)

# Botón de descarga en Streamlit
with col_d:
    st.download_button(
        label="📥 Descargar Reporte Excel",
        data=output,
        file_name='REPORTE_SEMANAL.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)





# Dividir en dos columnas
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Resumen Últimas 12 Semanas")
    st.dataframe(resumen_final, use_container_width=True)

with col2:
    st.subheader("Gráfico de Generación")
    
    st.pyplot(fig)

   



















