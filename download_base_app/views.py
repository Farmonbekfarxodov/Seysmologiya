import requests
import json
import datetime
import time

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from .models import HydrogenSeismology

# Stansiya va quduqlar ro'yxati (API_code va DB_name)
STATIONS_AND_WELLS = {
    "SMRM": {
        "name": "SMRM",
        "wells": [
            {"api_well": "A.Yassaviy", "db_well": "A.Yassaviy"},
            {"api_well": "Fozilova", "db_well": "Fozilova"},
            {"api_well": "Nazarbek", "db_well": "Nazarbek"},
            {"api_well": "Sabzavotchilik instituti", "db_well": "Sabzavotchilik instituti"},
            {"api_well": "Tekstil", "db_well": "Tekstil"},
            {"api_well": "Zam-zam", "db_well": "Zam-zam"},
            {"api_well": "Yangi yo'l", "db_well": "Yangi yo'l"}
        ]
    },
    "CHRT": {
        "name": "Chortoq",
        "wells": [
            {"api_well": "Chortoq 2", "db_well": "Chortoq 2"},
            {"api_well": "Chortoq 3", "db_well": "Chortoq 3"},
            {"api_well": "Chortoq 6", "db_well": "Chortoq 6"}
        ]
    },
    "Chimyon KPS": {
        "name": "Chimyon KPS",
        "wells": [
            {"api_well": "Chimyon 1-YU", "db_well": "Chimyon 1-Yu"},
            {"api_well": "Chimyon NP-1", "db_well": "Chimyon NP-1"},
            {"api_well": "Chimyon P-1", "db_well": "Chimyon P-1"}
        ]
    },
    "Namangan KPS": {
        "name": "Namangan KPS",
        "wells": [
            {"api_well": "Namangan 1", "db_well": "Namangan 1"},
            {"api_well": "Namangan 2", "db_well": "Namangan 2"},
            {"api_well": "Namangan 3", "db_well": "Namangan 3"},
            {"api_well": "Ming bulog'", "db_well": "Ming bulog'"}
        ]
    },
    "JMB": {
        "name": "Jumabozor",
        "wells": [
            {"api_well": "Jumabozor 1", "db_well": "Jumabozor 1"},
            {"api_well": "Jumabozor 2", "db_well": "Jumabozor 2"},
            {"api_well": "Navro'z", "db_well": "Navro'z"},
            {"api_well": "Ibn Sino", "db_well": "Ibn Sino"}
        ]
    },
    "DJAN": {
        "name": "Jongeldi",
        "wells": [
            {"api_well": "Tuyaq-R", "db_well": "Tuyaqochar"},
            {"api_well": "meteo-ya", "db_well": "Meteostansiya"},
            {"api_well": "Gumbaz", "db_well": "Gumbaz"}
        ]
    },
    "Ozodboshi": {
        "name": "Ozodboshi",
        "wells": [
            {"api_well": "Minora", "db_well": "Minora"},
            {"api_well": "GAI", "db_well": "GAI"},
            {"api_well": "Chotqol", "db_well": "Chotqol"},
            {"api_well": "Ozodbosh bulog'i", "db_well": "Ozodbosh bulog'i"}
        ]
    },
    "Sho'rchi": {
        "name": "Sho'rchi",
        "wells": [
            {"api_well": "Shorchi 5", "db_well": "Sho'rchi 5"},
            {"api_well": "Shorchi 7", "db_well": "Sho'rchi 7"},
            {"api_well": "Sho'rchi 8", "db_well": "Sho'rchi 8"},
            {"api_well": "Sho'rchi 295", "db_well": "Sho'rchi 295"},
            {"api_well": "Sho'rchi 293", "db_well": "Sho'rchi 293"},
            {"api_well": "Sho'rchi 292", "db_well": "Sho'rchi 292"}
        ]
    },
    "XRB": {
        "name": "Xarabek",
        "wells": [
            {"api_well": "Xaрабек", "db_well": "Xarabek"},
            {"api_well": "Bobur bulog'i", "db_well": "Bobur bulog'i"}
        ]
    },
    "Buxoro": {
        "name": "Yangi Buxoro",
        "wells": [
            {"api_well": "Buxoro", "db_well": "Buxoro"},
            {"api_well": "Jo'yzar", "db_well": "Jo'yzar"},
            {"api_well": "Setorai moxixosa", "db_well": "Setorai moxixosa"},
            {"api_well": "Xarbiy bo'lim", "db_well": "Xarbiy bo'lim"},
            {"api_well": "Yangi obod", "db_well": "Yangi obod"}
        ]
    },
    "Guliston": {
        "name": "Guliston",
        "wells": [
            {"api_well": "Guliston", "db_well": "Guliston"}
        ]
    },
}

LOGIN_URL = "https://api.geofizik.uz/api/login"
DATA_URL = "https://api.geofizik.uz/api/hydrogen-seismologies"
USERNAME = "Rasulov Alisher"
PASSWORD = "rasulovalisher"


def get_auth_token():
    try:
        payload = {"username": USERNAME, "password": PASSWORD}
        response = requests.post(LOGIN_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json().get("result", {}).get("token")
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Login xatosi: {e}")
        return None


def fetch_data_from_api(params, token):
    headers = {"Authorization": f"Bearer {token}"}
    all_data = []
    url = DATA_URL
    
    try:
        while url:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            response_data = response.json()

            if response_data and response_data.get('result'):
                data = response_data['result'].get('data', [])
                all_data.extend(data)
                url = response_data['result'].get('next_page_url')
                params = {}
            else:
                break
        return all_data
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"API dan ma'lumot olishda xato: {e}")
        return None


def save_data_to_db(data_list):
    if not data_list:
        return 0
    
    rows_to_create = []
    for item in data_list:
        try:
            formatted_date_naive = datetime.datetime.strptime(item.get('date'), '%d.%m.%Y')
            formatted_date = timezone.make_aware(formatted_date_naive, timezone.get_current_timezone())
        except (ValueError, TypeError):
            formatted_date = None

        api_station_code = item.get('station_code')
        api_well_code = item.get('well_code')

        db_station_name = None
        db_well_name = None

        # 🔑 API -> DB mapping
        station_info = STATIONS_AND_WELLS.get(api_station_code)
        if station_info:
            db_station_name = station_info["name"]
            for well in station_info["wells"]:
                if well["api_well"] == api_well_code:
                    db_well_name = well["db_well"]
                    break

        if not db_station_name or not db_well_name:
            continue  
        
        new_record = HydrogenSeismology(
            station_code=db_station_name,  
            well_code=db_well_name,        
            date=formatted_date,
            he=item.get('he'),
            h2=item.get('h2'),
            o2=item.get('o2'),
            n2=item.get('n2'),
            ch4=item.get('ch4'),
            co2=item.get('co2'),
            c2h6=item.get('c2h6'),
            ph=item.get('ph'),
            eh=item.get('eh'),
            hco3=item.get('hco3'),
            ci2=item.get('ci2'),
            sio2=item.get('sio2'),
            f=item.get('f'),
            i=item.get('i'),
            b2o3=item.get('b2o3'),
            dis_rn=item.get('dis_rn'),
            nep_rn=item.get('nep_rn'),
            he2=item.get('he2'),
            t0=item.get('t0'),
            q=item.get('q'),
            p=item.get('p'),
            eocc=item.get('eocc'),
            nep_t0=item.get('nep_t0')
        )
        rows_to_create.append(new_record)
        
    HydrogenSeismology.objects.bulk_create(rows_to_create)
    return len(rows_to_create)


def index(request):
    return render(request, 'download_base_app/index.html', {'stations': STATIONS_AND_WELLS})


@csrf_exempt
def upload(request):
    if request.method == 'GET':
        return render(request, 'download_base_app/index.html', {'stations': STATIONS_AND_WELLS})

    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            station_code = data.get('station')
            well_code = data.get('well')
            start_date = data.get('start_date')
            end_date = data.get('end_date')

            if not all([station_code, well_code, start_date, end_date]):
                return JsonResponse({"success": False, "message": "Ma'lumotlar to'liq emas."}, status=400)

            auth_token = get_auth_token()
            if not auth_token:
                return JsonResponse({"success": False, "message": "Login muvaffaqiyatsiz yoki serverga ulanishda xato."}, status=401)

            all_data = []
            api_stations_to_fetch = []
            
            if station_code == "all":
                for st_code, station_info in STATIONS_AND_WELLS.items():
                    for well in station_info["wells"]:
                        api_stations_to_fetch.append({"station_code": st_code, "well_code": well["api_well"]})
            else:
                station_info = STATIONS_AND_WELLS.get(station_code)
                if not station_info:
                    return JsonResponse({"success": False, "message": f"Noto'g'ri stansiya kodi: {station_code}"}, status=400)
                
                if well_code == "all_wells":
                    for well in station_info["wells"]:
                        api_stations_to_fetch.append({"station_code": station_code, "well_code": well["api_well"]})
                else:
                    api_stations_to_fetch.append({"station_code": station_code, "well_code": well_code})

            for fetch_item in api_stations_to_fetch:
                params = {
                    'station_code': fetch_item["station_code"], 
                    'well_code': fetch_item["well_code"], 
                    'date_start': start_date, 
                    'date_end': end_date
                }
                
                fetched_data = fetch_data_from_api(params, auth_token)
                
                if fetched_data is None:
                    return JsonResponse({"success": False, "message": "API'dan ma'lumot olishda xato yuz berdi. Iltimos, server loglarini tekshiring."}, status=500)
                
                all_data.extend(fetched_data)
                time.sleep(1)

            rows_inserted = save_data_to_db(all_data)
            return JsonResponse({"success": True, "message": f"{rows_inserted} ta yozuv bazaga saqlandi."})
        
        except json.JSONDecodeError:
            return JsonResponse({"success": False, "message": "Noto'g'ri JSON formatida ma'lumot yuborildi."}, status=400)
        except Exception as e:
            return JsonResponse({"success": False, "message": f"Kutilmagan xatolik: {type(e).__name__}: {str(e)}"}, status=500)
            
    return JsonResponse({"success": False, "message": "Noto'g'ri so'rov usuli."}, status=405)
