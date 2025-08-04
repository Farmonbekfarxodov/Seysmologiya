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

# Faqat Mb ni magnitudaning qiymati deb qabul qilamiz.
# Excel faylingizdagi ustun nomlariga mos kelishini ta'minlang.
# Sizning image_eb9809.png rasmida "Mk" va "Ml" ko'rsatilgan.
# Agar "Mb" va "Ml" ustunlari bo'lsa, quyidagicha qiling:
MAIN_MAGNITUDE_COLUMN = 'Mb' # Siz aytganingizdek Mb
SECONDARY_MAGNITUDE_COLUMN = 'Ml' # Ml ustuni ma'lumotini o'qiymiz, lekin grafikda alohida ko'rsatmaymiz

# Parametrlar guruhlari
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
        df[secondary_mag_col] = pd.to_numeric(df[secondary_mag_col], errors='coerce') # Ml ham o'qiladi, lekin hozirda ishlatilmaydi

        df.dropna(subset=[main_mag_col], inplace=True)
        df = df[df[main_mag_col] >= min_mag].copy()

        df['R(km)'] = np.round(destenc_vectorized(well_lat, well_lon, df[lat_col], df[lon_col]))
        # M/lgR hisoblashda faqat MAIN_MAGNITUDE_COLUMN (Mb) ishlatiladi
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
                         row_data[main_mag_col], row_data[secondary_mag_col]]) # Ml qiymatini ham saqlaymiz, lekin ishlatmaymiz
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


# --- Anomaliyalarni aniqlash va chizish funksiyasi (FAQAT ANOMALIYALAR CHIZILADI) ---
def plot_data_with_anomalies(fig, x_val, y_val, mean, sigma, btn_value, row_idx, col_idx, trace_color, element_name, key_name):
    """
    Anomaliyalarni (chegaradan tashqaridagi nuqtalarni yoki segmentlarni) qizil rangda chizadi.
    Faqat anomaliya holatdagi ma'lumotlarni ko'rsatadi.
    """
    upper_bound = mean + btn_value * sigma
    lower_bound = mean - btn_value * sigma

    # Chegaralarni qo'shish (faqat monitoring maqsadida ko'rsatiladi)
    fig.add_hline(y=upper_bound, line=dict(color='green', width=1.1, dash='dash'),
                  annotation_text=f"UB ({btn_value}σ)", annotation_position="top right",
                  row=row_idx, col=col_idx, secondary_y=False, annotation_font_color="green")
    fig.add_hline(y=mean, line=dict(color='magenta', width=1.1, dash='dot'),
                  annotation_text="Mean", annotation_position="bottom right",
                  row=row_idx, col=col_idx, secondary_y=False, annotation_font_color="magenta")
    fig.add_hline(y=lower_bound, line=dict(color='blue', width=1.1, dash='dash'),
                  annotation_text=f"LB ({-btn_value}σ)", annotation_position="top left",
                  row=row_idx, col=col_idx, secondary_y=False, annotation_font_color="blue")

    current_anomalous_segment_x = []
    current_anomalous_segment_y = []
    
    for i in range(len(x_val)):
        x_curr, y_curr = x_val[i], y_val[i]
        is_anomalous_curr = (y_curr > upper_bound) or (y_curr < lower_bound)

        if i == 0:
            if is_anomalous_curr:
                current_anomalous_segment_x.append(x_curr)
                current_anomalous_segment_y.append(y_curr)
            continue

        x_prev, y_prev = x_val[i-1], y_val[i-1]
        is_anomalous_prev = (y_prev > upper_bound) or (y_prev < lower_bound)

        # Chegarani kesib o'tishni tekshirish
        intersect_x = None
        intersect_y = None

        # Yuqori chegarani kesib o'tish
        if (y_prev < upper_bound <= y_curr) or (y_curr < upper_bound <= y_prev):
            if abs(y_curr - y_prev) > 1e-9:
                ratio = (upper_bound - y_prev) / (y_curr - y_prev)
                intersect_x = x_prev + (x_curr - x_prev) * ratio
                intersect_y = upper_bound

        # Pastki chegarani kesib o'tish
        if (y_prev > lower_bound >= y_curr) or (y_curr > lower_bound >= y_prev):
            if abs(y_curr - y_prev) > 1e-9:
                ratio = (lower_bound - y_prev) / (y_curr - y_prev)
                new_intersect_x = x_prev + (x_curr - x_prev) * ratio
                new_intersect_y = lower_bound
                
                # Agar bir nechta kesishish bo'lsa, eng yaqinini tanlang
                if intersect_x is None or (intersect_x and abs((new_intersect_x - x_prev).total_seconds()) < abs((intersect_x - x_prev).total_seconds())):
                    intersect_x = new_intersect_x
                    intersect_y = new_intersect_y

        # Agar anomaliya holati o'zgarsa va kesishish nuqtasi bo'lsa
        if is_anomalous_curr != is_anomalous_prev and intersect_x is not None:
            if is_anomalous_prev: # Oldingi segment anomaliya bo'lsa, tugatamiz
                current_anomalous_segment_x.append(intersect_x)
                current_anomalous_segment_y.append(intersect_y)
                fig.add_trace(go.Scatter(x=current_anomalous_segment_x, y=current_anomalous_segment_y,
                                         mode='lines', line=dict(color='red', width=3),
                                         showlegend=False # Legend yo'q
                                        ),
                              row=row_idx, col=col_idx, secondary_y=False)
                current_anomalous_segment_x = []
                current_anomalous_segment_y = []
            
            if is_anomalous_curr: # Yangi segment anomaliya bo'lsa, boshlaymiz
                current_anomalous_segment_x.append(intersect_x)
                current_anomalous_segment_y.append(intersect_y)
                current_anomalous_segment_x.append(x_curr)
                current_anomalous_segment_y.append(y_curr)
            
        elif is_anomalous_curr: # Anomaliya davom etsa yoki yangi anomaliya boshlansa
            current_anomalous_segment_x.append(x_curr)
            current_anomalous_segment_y.append(y_curr)
        elif not is_anomalous_curr and is_anomalous_prev: # Anomaliya tugasa, segmentni chizamiz
            current_anomalous_segment_x.append(x_curr)
            current_anomalous_segment_y.append(y_curr)
            fig.add_trace(go.Scatter(x=current_anomalous_segment_x, y=current_anomalous_segment_y,
                                     mode='lines', line=dict(color='red', width=3),
                                     showlegend=False # Legend yo'q
                                    ),
                          row=row_idx, col=col_idx, secondary_y=False)
            current_anomalous_segment_x = []
            current_anomalous_segment_y = []
        else: # Anomaliya emas va davom etsa
            current_anomalous_segment_x = []
            current_anomalous_segment_y = []
    
    # Oxirgi anomaliya segmentini chizish (agar mavjud bo'lsa)
    if current_anomalous_segment_x:
        fig.add_trace(go.Scatter(x=current_anomalous_segment_x, y=current_anomalous_segment_y,
                                 mode='lines', line=dict(color='red', width=3),
                                 showlegend=False # Legend yo'q
                                ),
                      row=row_idx, col=col_idx, secondary_y=False)

    # Y-o'q diapazoni uchun barcha qiymatlarni qaytarish
    y_all_values = list(y_val)
    return y_all_values


# Magnitudalarni chizish funksiyasi (FAQAT Mb QIYMATI CHIZILADI)
def draw_magnitude_values(fig, result_df, row_index, col_index=1):
    """
    Seysmik Mb magnitudalarni ikkinchi Y-o'qda chizadi.
    :param fig: Plotly Figure ob'ekti.
    :param result_df: process_dataframe dan qaytgan DataFrame.
    :param row_index: Subplotning qator indeksi.
    :param col_index: Subplotning ustun indeksi.
    """
    if result_df is None or result_df.empty:
        logging.info(f"draw_magnitude_values: result_df is empty for row {row_index}")
        return [0, 1]

    # Faqat MAIN_MAGNITUDE_COLUMN (Mb) qiymatlarini filtrlaymiz
    filtered_main_mag = result_df[result_df[MAIN_MAGNITUDE_COLUMN] > 0][MAIN_MAGNITUDE_COLUMN]
    
    max_mag_for_y_axis = filtered_main_mag.max() * 1.1 if not filtered_main_mag.empty else 0.1

    fig.update_yaxes(
        range=[0, max_mag_for_y_axis],
        secondary_y=True,
        title_text="Magnituda (Mb)", # Sarlavhani Mb ga o'zgartirdik
        row=row_index,
        col=col_index
    )

    valid_mb_events = result_df[result_df[MAIN_MAGNITUDE_COLUMN] > 0].copy()
    if not valid_mb_events.empty:
        fig.add_trace(go.Scatter(
            x=valid_mb_events['datetime_combined'],
            y=valid_mb_events[MAIN_MAGNITUDE_COLUMN],
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle'),
            name=f'{MAIN_MAGNITUDE_COLUMN} Magnituda',
            hoverinfo='x+y',
            hovertemplate='Vaqt: %{x}<br>' + f'{MAIN_MAGNITUDE_COLUMN} Magnituda: ' + '%{y}<extra></extra>',
            legendgroup='magnitudes_mb', # Legend group nomi o'zgartirildi
            showlegend=True,
            yaxis='y2'
        ),
            row=row_index,
            col=col_index,
            secondary_y=True
        )

    # SECONDAAY_MAGNITUDE_COLUMN (Ml) ni chizish qismi olib tashlandi
    # valid_ml_events = result_df[result_df[SECONDARY_MAGNITUDE_COLUMN] > 0].copy()
    # if not valid_ml_events.empty:
    #     fig.add_trace(go.Scatter(
    #         x=valid_ml_events['datetime_combined'],
    #         y=valid_ml_events[SECONDARY_MAGNITUDE_COLUMN],
    #         mode='markers',
    #         marker=dict(color='hotpink', size=8, symbol='diamond'),
    #         name=f'{SECONDARY_MAGNITUDE_COLUMN} Magnituda',
    #         hoverinfo='x+y',
    #         hovertemplate='Vaqt: %{x}<br>' + f'{SECONDARY_MAGNITUDE_COLUMN} Magnituda: ' + '%{y}<extra></extra>',
    #         legendgroup='magnitudes_ml',
    #         showlegend=True,
    #         yaxis='y2'
    #     ),
    #         row=row_index,
    #         col=col_index,
    #         secondary_y=True
    #     )

    fig.update_xaxes(showgrid=True, gridwidth=0.15, gridcolor='black', griddash='dot',
                     row=row_index, col=col_index)
    fig.update_yaxes(showgrid=True, gridwidth=0.15, gridcolor='black', griddash='dot',
                     row=row_index, col=col_index, secondary_y=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.15, gridcolor='gray', griddash='dot',
                     row=row_index, col=col_index, secondary_y=True)

    return [0, max_mag_for_y_axis]


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
            for group_name, params_list in DEFAULT_ELEMENTS_GROUPS.items():
                selected_params.extend(params_list)
            selected_params = sorted(list(set(selected_params)))

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
    Generates and displays seismic analysis results and plots.
    Now plotting only Mb magnitudes.
    """
    selected_keys = request.session.get('selected_keys', [])
    selected_params = request.session.get('selected_params', [])
    min_mag = request.session.get('min_mag')
    btn_value = request.session.get('btn_value')
    min_mlgr = request.session.get('min_mlgr')
    excel_file = request.session.get('excel_file')

    if not all([selected_keys, excel_file, min_mag is not None, btn_value is not None, min_mlgr is not None]):
        logging.warning("Missing session data for results view. Redirecting to parametrs.")
        return render(request, 'results.html', {'error': 'To‘liq ma’lumotlar mavjud emas. Iltimos, oldingi qadamlarga qayting.'})

    if not selected_params:
        for group_name, params_list in DEFAULT_ELEMENTS_GROUPS.items():
            selected_params.extend(params_list)
        selected_params = sorted(list(set(selected_params)))

    try:
        file_path = os.path.join(settings.MEDIA_ROOT, excel_file)
        if not os.path.exists(file_path):
            logging.error(f"Excel file not found at {file_path}")
            return render(request, 'results.html', {'error': 'Yuklangan Excel fayl topilmadi.'})

        dfe = pd.read_excel(file_path)
        
        required_excel_cols = [DATE_COLUMN, TIME_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN, 
                               MAIN_MAGNITUDE_COLUMN, SECONDARY_MAGNITUDE_COLUMN]
        if not all(col in dfe.columns for col in required_excel_cols):
            missing_cols = [col for col in required_excel_cols if col not in dfe.columns]
            logging.error(f"Excel file is missing required columns: {missing_cols}")
            return render(request, 'results.html', {
                'error': f"Excel faylida kerakli ustunlar topilmadi: {', '.join(missing_cols)}. "
                         f"Iltimos, ustun nomlarini tekshiring (masalan: {', '.join(required_excel_cols)})."
            })

        lst_stansiya, well_coords = fetch_data()
        if not lst_stansiya or not well_coords:
             return render(request, 'results.html', {'error': 'Ma’lumotlar bazasidan ma’lumotlarni yuklashda xatolik yuz berdi.'})
        
        plot_files = []
        engine = None
        conn = None
        try:
            engine = connect_db()
            conn = engine.connect()

            for element in selected_params:
                num_subp = len(selected_keys)
                H = max(400, 300 * num_subp)

                subplot_titles = [f"<b>{key}</b> uchun <b>{element}</b> (Anomaliya: {btn_value}σ | M>={min_mag} | M/lgR>={min_mlgr})" for key in selected_keys]

                fig = make_subplots(rows=num_subp, cols=1, subplot_titles=subplot_titles,
                                    vertical_spacing=0.08, specs=[[{"secondary_y": True}] for _ in selected_keys])
                
                fig.update_layout(height=H, width=1600,
                                  title_text=f"<b>{element}</b> parametrlari va seysmik voqealar",
                                  showlegend=True,
                                  plot_bgcolor='gainsboro', 
                                  hovermode='x unified')

                current_colors = COLOR_PALETTE.copy() 

                for idx, key in enumerate(selected_keys):
                    stansiya, skvajina = key.split(' | ')
                    ssdi_id = lst_stansiya.get(key, {}).get(element)
                    
                    if not ssdi_id:
                        logging.warning(f"No ssdi_id found for Stansiya: {stansiya}, Skvajina: {skvajina}, Element: {element}")
                        fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5,
                            text=f"Ma'lumotlar mavjud emas: {key} - {element}",
                            showarrow=False, font=dict(size=16, color="red"), row=idx+1, col=1)
                        continue
                    
                    try:
                        query = text(f"SELECT date, `{ssdi_id}` FROM alldata WHERE `{ssdi_id}` IS NOT NULL")
                        data = conn.execute(query).fetchall()

                        if not data:
                            logging.info(f"No data found for ssdi_id: {ssdi_id} ({key}, {element})")
                            fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5,
                                text=f"Ma'lumotlar mavjud emas: {element}",
                                showarrow=False, font=dict(size=16, color="gray"), row=idx+1, col=1)
                            continue
                        
                        x_val = [item[0] for item in data]
                        y_val = [item[1] for item in data]

                        x_val_dt = pd.to_datetime(x_val)

                        mean = np.mean(np.array(y_val))
                        sigma = np.std(np.array(y_val))

                        current_trace_color = random.choice(current_colors)
                        current_colors.remove(current_trace_color)

                        y_all_for_range = plot_data_with_anomalies(fig, x_val_dt, y_val, mean, sigma, btn_value, idx + 1, 1, current_trace_color, element, key)
                        
                        fig.update_yaxes(title_text=f"{element} Qiymati", 
                                         range=[min(y_all_for_range) * 0.9, max(y_all_for_range) * 1.1],
                                         row=idx+1, col=1, secondary_y=False)

                        lat, lon = well_coords.get(skvajina, (0, 0))
                        
                        result_df = process_dataframe(dfe, min_mag, min_mlgr, lat, lon,
                                                      DATE_COLUMN, TIME_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN, 
                                                      MAIN_MAGNITUDE_COLUMN, SECONDARY_MAGNITUDE_COLUMN)
                        
                        draw_magnitude_values(fig, result_df, idx + 1)
                        
                        fig.update_xaxes(
                            tickformat="%Y",
                            showgrid=True,
                            griddash='dot',
                            dtick="M12",
                            row=idx+1, col=1
                        )

                    except exc.SQLAlchemyError as db_err:
                        logging.error(f"Database operation error for {key}: {db_err}")
                        fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5,
                            text=f"DB xatosi: {element}", showarrow=False, font=dict(size=16, color="orange"), row=idx+1, col=1)
                    except KeyError as ke:
                        logging.error(f"Data processing key error for {key}, element {element}: {ke}")
                        fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5,
                            text=f"Ma'lumot formatida xato: {element}", showarrow=False, font=dict(size=16, color="orange"), row=idx+1, col=1)
                    except Exception as e:
                        logging.error(f"Graph draw error for {key}, element {element}: {e}")
                        fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5,
                            text=f"Chizmada xato: {element}", showarrow=False, font=dict(size=16, color="red"), row=idx+1, col=1)
                
                safe_element_name = re.sub(r'[\/:*?"<>|]', '_', element)
                if len(selected_keys) == 1:
                    safe_key_name = re.sub(r'[\/:*?"<>|]', '_', selected_keys[0])
                    file_name = f"{safe_key_name}_{safe_element_name}.html"
                else:
                    file_name = f"multi_well_{safe_element_name}.html"
                
                file_path = os.path.join(settings.MEDIA_ROOT, file_name)
                fig.write_html(file_path, include_plotlyjs='cdn')
                plot_files.append({'element': element, 'file_url': f'{settings.MEDIA_URL}{file_name}'})

        except Exception as e:
            logging.error(f"An error occurred during plot generation: {e}")
            return render(request, 'results.html', {'error': f'Grafiklarni yaratishda kutilmagan xato: {e}'})
        finally:
            if conn:
                conn.close()
            if engine:
                engine.dispose()

        return render(request, 'results.html', {'plot_files': plot_files})
    except pd.errors.EmptyDataError:
        logging.error(f"Excel file '{excel_file}' is empty.")
        return render(request, 'results.html', {'error': 'Yuklangan Excel fayli bo‘sh.'})
    except FileNotFoundError:
        logging.error(f"Excel file '{excel_file}' not found during processing.")
        return render(request, 'results.html', {'error': 'Yuklangan Excel fayl topilmadi yoki o‘chirildi.'})
    except Exception as e:
        logging.error(f"Results view main error: {e}")
        return render(request, 'results.html', {'error': f'Natijalarni qayta ishlashda umumiy xato yuz berdi: {e}'})