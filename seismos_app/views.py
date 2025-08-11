import os
import json
import logging
from math import pi, sqrt, sin, cos, atan2, log10
from datetime import datetime, timedelta
import random

import pandas as pd
import numpy as np
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from sqlalchemy import create_engine, text, exc
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re

# Setup logging
logging.basicConfig(level=logging.INFO, filename='seismic_app.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DATE_COLUMN = 'Date'
TIME_COLUMN = 'Time'
LATITUDE_COLUMN = 'Latitude'
LONGITUDE_COLUMN = 'Longitude'

MAIN_MAGNITUDE_COLUMN = 'Mb'
SECONDARY_MAGNITUDE_COLUMN = 'Ml'

DEFAULT_ELEMENTS_GROUPS = {
    'gazli': ['He', 'H2', 'O2', 'N2', 'CH4', 'CO2'],
    'kimyoviy': ['F', 'C2H6', 'pH', 'Eh', 'HCO3', 'Cl2'],
    'fizikaviy': ['T0', 'Q', 'P', 'EOCC']
}

# --- Yangi ranglar palitrasi ---
COLOR_PALETTE = [
    'blue', 'green', 'orange', 'purple', 'yellow', 'brown', 'pink',
    'cyan', 'lime', 'teal', 'gold', 'navy', 'magenta', 'olive',
    'indigo', 'turquoise', 'plum'
]

# Database Utilities
def get_db_config():
    """Reads database configuration from user_info.json."""
    config_path = os.path.join(settings.BASE_DIR, 'user_info.json')
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {config_path}. Check file format.")
        raise

def connect_db():
    """Establishes and returns a database engine connection."""
    try:
        config = get_db_config()
        engine = create_engine(f"mysql+mysqlconnector://{config['user']}:{config['psw']}@{config['ip']}/{config['db']}")
        logging.info("Successfully created DB engine.")
        return engine
    except Exception as e:
        logging.error(f"DB engine creation error: {e}")
        raise

# --- Data Fetching ---
def fetch_data():
    """
    Fetches station, well, measurement, and coordinates data from the database.
    Returns:
        tuple: (dict of station measurements, dict of well coordinates)
    """
    engine = None
    try:
        engine = connect_db()
        
        query_izmereniya = "SELECT stansiya, skvajina, izmereniya, ssdi_id FROM all_izmereniya"
        df_izmereniya = pd.read_sql(query_izmereniya, engine)

        lst_stansiya = {}
        for (st, sk), group in df_izmereniya.groupby(['stansiya', 'skvajina']):
            lst_stansiya[f"{st} | {sk}"] = dict(zip(group['izmereniya'], group['ssdi_id']))

        coords_query = "SELECT naim, Latitude, Longitude FROM skvajina"
        coords_df = pd.read_sql(coords_query, engine)
        well_coords = {row['naim'].strip(): (row['Latitude'], row['Longitude']) for _, row in coords_df.iterrows()}

        return lst_stansiya, well_coords
    except exc.SQLAlchemyError as e:
        logging.error(f"Database query error in fetch_data: {e}")
        return {}, {}
    except Exception as e:
        logging.error(f"An unexpected error occurred in fetch_data: {e}")
        return {}, {}
    finally:
        if engine:
            engine.dispose()

# --- Utility Functions ---
def destenc_vectorized(lat1, lon1, lat2_series, lon2_series):
    """
    Calculates the Haversine distance in kilometers between a single point
    and a series of points.
    """
    deg_to_rad = pi / 180.0
    d_lat = (lat2_series - lat1) * deg_to_rad
    d_lon = (lon2_series - lon1) * deg_to_rad
    a = np.sin(d_lat / 2)**2 + np.cos(lat1 * deg_to_rad) * np.cos(lat2_series * deg_to_rad) * np.sin(d_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c

def process_dataframe(df, min_mag, min_mlgr, well_lat, well_lon,
                      date_col, time_col, lat_col, lon_col, main_mag_col, secondary_mag_col):
    """
    Processes the earthquake DataFrame to filter by main magnitude, calculate distance
    and M/lgR, and format for plotting.
    Now only considers MAIN_MAGNITUDE_COLUMN for M/lgR calculation and filtering.
    """
    try:
        required_cols = [date_col, time_col, lat_col, lon_col, main_mag_col, secondary_mag_col]
        if not all(col in df.columns for col in required_cols):
            logging.error(f"Missing required columns in Excel file. Expected: {required_cols}")
            return None

        df[main_mag_col] = pd.to_numeric(df[main_mag_col], errors='coerce')
        df[secondary_mag_col] = pd.to_numeric(df[secondary_mag_col], errors='coerce')

        df.dropna(subset=[main_mag_col], inplace=True)
        df = df[df[main_mag_col] >= min_mag].copy()

        df['R(km)'] = np.round(destenc_vectorized(well_lat, well_lon, df[lat_col], df[lon_col]))
        df['M/lgR'] = np.where(df['R(km)'] > 1, df[main_mag_col] / np.log10(df['R(km)']), np.nan)

        df = df[df['M/lgR'] >= min_mlgr].copy()
        
        rows = []

        df['parsed_date'] = pd.to_datetime(df[date_col], format='%d.%m.%Y', errors='coerce')
        
        df['time_str'] = df[time_col].astype(str)
        df['time_delta'] = pd.to_timedelta(df['time_str'].apply(lambda x: x if ':' in x else '00:00:00'), errors='coerce')
        
        df['combined_datetime'] = df['parsed_date'] + df['time_delta']
        
        df.sort_values(by=['combined_datetime'], inplace=True)
        df.dropna(subset=['combined_datetime'], inplace=True)

        for _, row_data in df.iterrows():
            current_datetime = row_data['combined_datetime']
            
            rows.append([current_datetime.strftime('%d.%m.%Y'), current_datetime.strftime('%H:%M:%S'), 
                         row_data[main_mag_col], row_data[secondary_mag_col]])
            rows.append([current_datetime.strftime('%d.%m.%Y'), (current_datetime + timedelta(seconds=1)).strftime('%H:%M:%S'), 0, 0])

        result = pd.DataFrame(rows, columns=[date_col, time_col, main_mag_col, secondary_mag_col])
        result['datetime_combined'] = pd.to_datetime(result[date_col] + ' ' + result[time_col], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        result.sort_values(by=['datetime_combined'], inplace=True)
        result.dropna(subset=['datetime_combined'], inplace=True)
        
        return result
    except KeyError as e:
        logging.error(f"Missing expected column in DataFrame: {e}. Check your DATE_COLUMN, TIME_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN, MAIN_MAGNITUDE_COLUMN, and SECONDARY_MAGNITUDE_COLUMN constants.")
        return None
    except Exception as e:
        logging.error(f"DataFrame processing error: {e}")
        return None


def generate_colors(n, cmap_name='tab20'):
    """
    Matplotlib colormap orqali `n` ta rang hosil qiladi.
    """
    cmap = plt.get_cmap(cmap_name)
    return [f'rgb{tuple(int(c * 255) for c in cmap(i % cmap.N)[:3])}' for i in range(n)]

def plot_data_with_anomalies(fig, x_val, y_val, mean, sigma, btn_value, row_idx, col_idx, trace_color, element_name, key_name):
    """
    Ma'lumotlarning butun chizig'ini chizadi va anomaliya qismlarini qizil rangda belgilaydi.
    """

    upper_bound = mean + btn_value * sigma
    lower_bound = mean - btn_value * sigma

    y_all_values = list(y_val)
    y_all_values.extend([upper_bound, lower_bound, mean])

    yaxis_index = (row_idx - 1) * 1 + col_idx
    yref = 'y' if yaxis_index == 1 else f'y{2 * row_idx - 1}'

    # UB (Upper Bound) chizig'i
    fig.add_shape(
        type='line',
        x0=min(x_val), x1=max(x_val),
        y0=upper_bound, y1=upper_bound,
        line=dict(color='green', width=1.5,),
        row=row_idx, col=col_idx,
        yref=yref, xref='x'
    )
    fig.add_annotation(
        x=max(x_val), y=upper_bound,
        text=f"UB ({btn_value}σ)",
        showarrow=False,
        font=dict(color='green', size=10),
        xanchor='right',
        yanchor='bottom',
        row=row_idx, col=col_idx
    )

    # MEAN chizig'i
    fig.add_shape(
        type='line',
        x0=min(x_val), x1=max(x_val),
        y0=mean, y1=mean,
        line=dict(color='magenta', width=1.5,),
        row=row_idx, col=col_idx,
        yref=yref, xref='x'
    )
    fig.add_annotation(
        x=max(x_val), y=mean,
        text="Mean",
        showarrow=False,
        font=dict(color='magenta', size=10),
        xanchor='right',
        yanchor='bottom',
        row=row_idx, col=col_idx
    )

    # LB (Lower Bound) chizig'i
    fig.add_shape(
        type='line',
        x0=min(x_val), x1=max(x_val),
        y0=lower_bound, y1=lower_bound,
        line=dict(color='blue', width=1.5,),
        row=row_idx, col=col_idx,
        yref=yref, xref='x'
    )
    fig.add_annotation(
        x=max(x_val), y=lower_bound,
        text=f"LB ({-btn_value}σ)",
        showarrow=False,
        font=dict(color='blue', size=10),
        xanchor='right',
        yanchor='top',
        row=row_idx, col=col_idx
    )

    # Asosiy grafik
    fig.add_trace(go.Scatter(
        x=x_val,
        y=y_val,
        mode='lines',
        line=dict(color=trace_color, width=1.5),
        name=f'{element_name} ({key_name})',
        showlegend=True,
        hoverinfo='x+y',
        hovertemplate=f'Vaqt: %{{x|%Y-%m-%d %H:%M}}<br>{element_name} Qiymati: %{{y}}<extra></extra>'
    ), row=row_idx, col=col_idx, secondary_y=False)

    # Anomaliya chiziqlari
    current_anomalous_segment_x = []
    current_anomalous_segment_y = []
    is_anomalous_prev = False

    for i in range(len(x_val)):
        x_curr, y_curr = x_val[i], y_val[i]
        is_anomalous_curr = (y_curr > upper_bound) or (y_curr < lower_bound)

        if i == 0:
            if is_anomalous_curr:
                current_anomalous_segment_x.append(x_curr)
                current_anomalous_segment_y.append(y_curr)
            is_anomalous_prev = is_anomalous_curr
            continue

        x_prev, y_prev = x_val[i - 1], y_val[i - 1]

        intersect_x = None
        intersect_y = None

        if (y_prev < upper_bound <= y_curr) or (y_curr < upper_bound <= y_prev):
            if abs(y_curr - y_prev) > 1e-9:
                ratio = (upper_bound - y_prev) / (y_curr - y_prev)
                intersect_x = x_prev + (x_curr - x_prev) * ratio
                intersect_y = upper_bound

        if (y_prev > lower_bound >= y_curr) or (y_curr > lower_bound >= y_prev):
            if abs(y_curr - y_prev) > 1e-9:
                ratio = (lower_bound - y_prev) / (y_curr - y_prev)
                new_intersect_x = x_prev + (x_curr - x_prev) * ratio
                new_intersect_y = lower_bound

                if intersect_x is None or (intersect_x and abs((new_intersect_x - x_prev).total_seconds()) < abs((intersect_x - x_prev).total_seconds())):
                    intersect_x = new_intersect_x
                    intersect_y = new_intersect_y

        if is_anomalous_curr != is_anomalous_prev and intersect_x is not None:
            if is_anomalous_prev:
                current_anomalous_segment_x.append(intersect_x)
                current_anomalous_segment_y.append(intersect_y)
                if current_anomalous_segment_x:
                    fig.add_trace(go.Scatter(
                        x=current_anomalous_segment_x,
                        y=current_anomalous_segment_y,
                        mode='lines',
                        line=dict(color='red', width=3),
                        showlegend=False,
                        hoverinfo='x+y',
                        hovertemplate='Vaqt: %{x|%Y-%m-%d %H:%M}<br>Anomaliya: %{y}<extra></extra>'
                    ), row=row_idx, col=col_idx, secondary_y=False)
                current_anomalous_segment_x = []
                current_anomalous_segment_y = []

            if is_anomalous_curr:
                current_anomalous_segment_x.append(intersect_x)
                current_anomalous_segment_y.append(intersect_y)
                current_anomalous_segment_x.append(x_curr)
                current_anomalous_segment_y.append(y_curr)

        elif is_anomalous_curr:
            current_anomalous_segment_x.append(x_curr)
            current_anomalous_segment_y.append(y_curr)
        elif not is_anomalous_curr and is_anomalous_prev:
            current_anomalous_segment_x.append(x_curr)
            current_anomalous_segment_y.append(y_curr)
            if current_anomalous_segment_x:
                fig.add_trace(go.Scatter(
                    x=current_anomalous_segment_x,
                    y=current_anomalous_segment_y,
                    mode='lines',
                    line=dict(color='red', width=3),
                    showlegend=False,
                    hoverinfo='x+y',
                    hovertemplate='Vaqt: %{x|%Y-%m-%d %H:%M}<br>Anomaliya: %{y}<extra></extra>'
                ), row=row_idx, col=col_idx, secondary_y=False)
            current_anomalous_segment_x = []
            current_anomalous_segment_y = []
        else:
            current_anomalous_segment_x = []
            current_anomalous_segment_y = []

        is_anomalous_prev = is_anomalous_curr

    if current_anomalous_segment_x:
        fig.add_trace(go.Scatter(
            x=current_anomalous_segment_x,
            y=current_anomalous_segment_y,
            mode='lines',
            line=dict(color='red', width=3),
            showlegend=False,
            hoverinfo='x+y',
            hovertemplate='Vaqt: %{x|%Y-%m-%d %H:%M}<br>Anomaliya: %{y}<extra></extra>'
        ), row=row_idx, col=col_idx, secondary_y=False)

    return y_all_values

def draw_magnitude_values(fig, result_df, row_index, col_index=1):
    """
    Seysmik Mb magnitudalarni ikkinchi Y-o'qda vertikal zangori chiziqlar orqali chizadi.
    """
    if result_df is None or result_df.empty:
        logging.info(f"draw_magnitude_values: result_df is empty for row {row_index}")
        return [0, 1]

    filtered_main_mag = result_df[result_df[MAIN_MAGNITUDE_COLUMN] > 0][MAIN_MAGNITUDE_COLUMN]
    max_mag_for_y_axis = filtered_main_mag.max() * 1.1 if not filtered_main_mag.empty else 0.1
    min_mag_for_y_axis = 0

    fig.update_yaxes(
        range=[min_mag_for_y_axis, max_mag_for_y_axis],
        secondary_y=True,
        title_text="Magnituda (Mb)",
        row=row_index,
        col=col_index
    )

    valid_mb_events = result_df[result_df[MAIN_MAGNITUDE_COLUMN] > 0].copy()

    stem_x = []
    stem_y = []
    hover_texts = []

    for _, row in valid_mb_events.iterrows():
        time_str = row['datetime_combined'].strftime('%d.%m.%Y %H:%M:%S')
        mag_val = row[MAIN_MAGNITUDE_COLUMN]

        stem_x.extend([row['datetime_combined'], row['datetime_combined'], None])
        stem_y.extend([0, mag_val, None])

        hover_texts.extend(["", f"Vaqt: {time_str}<br>Mb: {mag_val:.2f}", ""])

    if stem_x:
        fig.add_trace(go.Scatter(
            x=stem_x,
            y=stem_y,
            mode='lines',
            line=dict(color='navy', width=2),
            name=f'{MAIN_MAGNITUDE_COLUMN} Magnituda',
            hoverinfo='text',
            text=hover_texts,
            showlegend=True,
            legendgroup='magnitudes_mb',
            yaxis=f'y{2 * row_index}'
        ), row=row_index, col=col_index, secondary_y=True)

    fig.update_xaxes(showgrid=True, gridwidth=0.15, gridcolor='black', griddash='dot',
                     row=row_index, col=col_index)
    fig.update_yaxes(showgrid=True, gridwidth=0.15, gridcolor='black', griddash='dot',
                     row=row_index, col=col_index, secondary_y=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.15, gridcolor='gray', griddash='dot',
                     row=row_index, col=col_index, secondary_y=True)

    return [min_mag_for_y_axis, max_mag_for_y_axis]

# --- Views ---
def selection_view(request):
    """
    Handles the selection of wells and parametrs for analysis.
    """
    lst_stansiya, _ = fetch_data()
    all_params = []
    for group_name, params_list in DEFAULT_ELEMENTS_GROUPS.items():
        all_params.extend(params_list)
    all_params = sorted(list(set(all_params))) 

    if request.method == 'POST':
        selected_keys = request.POST.getlist('wells')
        selected_params = request.POST.getlist('params')

        if not selected_keys:
            return render(request, 'selection.html', {
                'wells': lst_stansiya.keys(),
                'params': all_params,
                'error': 'Kamida bitta quduq tanlang.'
            })

        if not selected_params:
            selected_params = sorted(list(set(sum(DEFAULT_ELEMENTS_GROUPS.values(), []))))

        request.session['selected_keys'] = selected_keys
        request.session['selected_params'] = selected_params
        return redirect('parametrs')

    return render(request, 'selection.html', {'wells': lst_stansiya.keys(), 'params': all_params})
   
def parametrs_view(request):
    """
    Handles input for seismic parametrs and file upload.
    """
    if request.method == 'POST':
        try:
            min_mag = float(request.POST['min_mag'])
            btn_value = float(request.POST['sigma'])
            min_mlgr = float(request.POST['min_mlgr'])
            
            if 'excel_file' not in request.FILES:
                return render(request, 'parametrs.html', {'error': 'Iltimos, Excel faylni yuklang.'})

            file = request.FILES['excel_file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file) 
            
            request.session.update({
                'excel_file': filename,
                'min_mag': min_mag,
                'btn_value': btn_value,
                'min_mlgr': min_mlgr
            })
            return redirect('results')
        except ValueError:
            logging.error("Invalid input for numeric parametrs.")
            return render(request, 'parametrs.html', {'error': 'Iltimos, barcha sonli maydonlarga to‘g‘ri qiymat kiriting.'})
        except Exception as e:
            logging.error(f"Parameter input or file upload error: {e}")
            return render(request, 'parametrs.html', {'error': f'Xato yuz berdi: {e}. Iltimos, qayta urinib ko‘ring.'})
    return render(request, 'parametrs.html')


def results_view(request):
    """
    Generates and displays seismic analysis results with graphs, and a separate map
    at the bottom of the page.
    """
    selected_keys = request.session.get('selected_keys', [])
    selected_params = request.session.get('selected_params', [])
    min_mag = request.session.get('min_mag')
    btn_value = request.session.get('btn_value')
    min_mlgr = request.session.get('min_mlgr')
    excel_file = request.session.get('excel_file')

    if not all([selected_keys, excel_file, min_mag is not None, btn_value is not None, min_mlgr is not None]):
        return render(request, 'results.html', {'error': 'To‘liq maʼlumotlar mavjud emas. Iltimos, oldingi qadamlarga qayting.'})

    calculated_R_km = None
    if min_mlgr != 0:
        try:
            calculated_R_km = 10**(min_mag / min_mlgr)
            logging.info(f"Hisoblangan radius(R): {calculated_R_km} km")
        except ValueError:
            logging.warning("M/lgr qiymati noto'g'ri, R ni hisoblab bo'lmadi.")
    else:
        logging.warning("M/lgr ning qiymati nolga teng, R ni hisoblab bo'lmaydi")

    if not selected_params:
        selected_params = sorted(list(set(sum(DEFAULT_ELEMENTS_GROUPS.values(), []))))

    try:
        file_path = os.path.join(settings.MEDIA_ROOT, excel_file)
        if not os.path.exists(file_path):
            return render(request, 'results.html', {'error': 'Yuklangan Excel fayli topilmadi.'})

        dfe = pd.read_excel(file_path)
        required_cols = [DATE_COLUMN, TIME_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN, MAIN_MAGNITUDE_COLUMN, SECONDARY_MAGNITUDE_COLUMN]
        if not all(col in dfe.columns for col in required_cols):
            missing = [col for col in required_cols if col not in dfe.columns]
            return render(request, 'results.html', {'error': f"Excel faylida kerakli ustunlar yo‘q: {', '.join(missing)}."})

        lst_stansiya, well_coords = fetch_data()
        if not lst_stansiya or not well_coords:
            return render(request, 'results.html', {'error': 'Bazadan maʼlumotlar olinmadi.'})

        engine = connect_db()
        conn = engine.connect()

        graph_data = []
        for key in selected_keys:
            for param in selected_params:
                ssdi_id = lst_stansiya.get(key, {}).get(param)
                if not ssdi_id:
                    continue
                query = text(f"SELECT date, `{ssdi_id}` FROM alldata WHERE `{ssdi_id}` IS NOT NULL")
                data = conn.execute(query).fetchall()
                if not data:
                    continue
                x_val = pd.to_datetime([row[0] for row in data])
                y_val = [row[1] for row in data]
                mean, sigma = np.mean(y_val), np.std(y_val)
                stansiya, skvajina = key.split(' | ')
                graph_data.append((x_val, y_val, mean, sigma, param, key, skvajina))

        if not graph_data:
            return render(request, 'results.html', {'error': 'Hech qanday mos keluvchi maʼlumot topilmadi.'})

        # --- Grafiklar uchun subplot yaratish ---
        num_graphs = len(graph_data)
        subplot_titles = [f"{key} - {param}" for (_, _, _, _, param, key, _) in graph_data]
        
        # Xarita uchun oxiriga bitta bo'sh joy qo'shamiz
        subplot_titles.append("Skvajinalar joylashgan xarita")
        
        # Xarita subplotining balandligini oshirish uchun row_heights listini yaratish
        row_heights = [1] * num_graphs + [2]  # Xarita 2 birlik balandlikda, boshqalar 1 birlikda
        
        fig = make_subplots(
            rows=num_graphs + 1, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.03, # Bo'shliqni biroz kamaytirish
            row_heights=row_heights,
            specs=[[{"secondary_y": True}]] * num_graphs + [[{"type": "mapbox"}]]
        )

        # --- Grafiklar chizish qismi ---
        color_pool = generate_colors(num_graphs)
        for idx, (x, y, mean, sigma, param, key, skv) in enumerate(graph_data):
            row, col = idx + 1, 1
            trace_color = color_pool[idx]
            y_all = plot_data_with_anomalies(fig, x, y, mean, sigma, btn_value, row, col, trace_color, param, key)
            fig.update_yaxes(title_text=f"{param} Qiymati", range=[min(y_all)*0.9, max(y_all)*1.1], row=row, col=col)
            
            lat, lon = well_coords.get(skv, (0, 0))
            result_df = process_dataframe(
                dfe, min_mag, min_mlgr, lat, lon,
                DATE_COLUMN, TIME_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN,
                MAIN_MAGNITUDE_COLUMN, SECONDARY_MAGNITUDE_COLUMN
            )
            draw_magnitude_values(fig, result_df, row, col)
            fig.update_xaxes(tickformat="%Y", showgrid=True, griddash='dot', dtick="M12", row=row, col=col)

        # --- Xaritani chizish qismi (eng oxirgi subplotga) ---
        map_row = num_graphs + 1
        
        map_df = pd.DataFrame(columns=['lat', 'lon', 'name'])
        for key in selected_keys:
            stansiya, skvajina = key.split(' | ')
            lat, lon = well_coords.get(skvajina, (None, None))
            if lat is not None and lon is not None:
                map_df = pd.concat([map_df, pd.DataFrame([{'lat': lat, 'lon': lon, 'name': skvajina}])], ignore_index=True)

        if not map_df.empty:
            # Skvajina nuqtalari
            fig.add_trace(go.Scattermapbox(
                lat=map_df['lat'],
                lon=map_df['lon'],
                mode='markers+text',
                marker=go.scattermapbox.Marker(
                    size=14,
                    color='red',
                    symbol='marker'
                ),
                text=map_df['name'],
                textposition="bottom right",
                hoverinfo='text',
                hovertemplate="Skvajina: %{text}",
                name="Skvajinalar",
                showlegend=False
            ), row=map_row, col=1)

            # Radiuslarni chizish
            if calculated_R_km is not None:
                for _, row_data in map_df.iterrows():
                    lats, lons, hover_texts = [], [], []
                    for i in range(0, 361, 10):
                        angle = i * pi / 180
                        lat_offset = (calculated_R_km / 111.32) * cos(angle)
                        lon_offset = (calculated_R_km / 111.32) * sin(angle) / cos(row_data['lat'] * pi / 180)
                        
                        lats.append(row_data['lat'] + lat_offset)
                        lons.append(row_data['lon'] + lon_offset)
                        hover_texts.append(f"Skvajina: {row_data['name']}<br>Radius: {calculated_R_km:.2f} km")
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode='lines',
                        line=go.scattermapbox.Line(color='blue', width=2),
                        hoverinfo='text',
                        text=hover_texts,
                        name=f"Ta'sir radiusi ({row_data['name']})",
                        opacity=0.5
                    ), row=map_row, col=1)

        # --- Layoutni sozlash ---
        fig.update_layout(
            title_text="Tahlil natijalari",
            height=max(700, (num_graphs * 400) + 800), # Rasm balandligini oshirish
            width=2000,
            showlegend=True,
            plot_bgcolor='gainsboro',
            hovermode='x unified',
            mapbox_style="open-street-map",
            mapbox_center=go.layout.mapbox.Center(
                lat=map_df['lat'].mean() if not map_df.empty else 41.2995,
                lon=map_df['lon'].mean() if not map_df.empty else 69.2401
            ),
            mapbox_zoom=6,
            legend=dict(x=0.01, y=1.0)
        )

        # ... (Natijani saqlash va ko'rsatish qismi o'zgarishsiz) ...
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"full_analysis_{timestamp}.html"
        filepath = os.path.join(settings.MEDIA_ROOT, file_name)

        try:
            fig.write_html(filepath, include_plotlyjs='cdn')
            plot_file = {'element': 'Barcha tahlillar', 'file_url': f"{settings.MEDIA_URL}{file_name}"}
        except BrokenPipeError:
            logging.warning("Broken pipe while writing plot.")
            plot_file = {'error': 'Grafik faylini yozishda xato. Iltimos, qayta urinib ko‘ring.'}

        return render(request, 'results.html', {'plot_files': [plot_file]})

    except Exception as e:
        logging.error(f"Results view error: {e}")
        return render(request, 'results.html', {'error': f"Xatolik: {e}"})

    finally:
        if 'conn' in locals(): conn.close()
        if 'engine' in locals(): engine.dispose()