import numpy as np

def calculate_ispu_pm25(concentration):
    """
    Computes ISPU (Indeks Standar Pencemar Udara) for PM2.5 based on Permen LHK No. P.14/2020.
    """
    c = float(concentration)
    if c <= 15.5:
        # Range 0 - 50
        return (50 / 15.5) * c
    elif c <= 55.4:
        # Range 51 - 100
        return 51 + ((100 - 51) / (55.4 - 15.6)) * (c - 15.6)
    elif c <= 150.4:
        # Range 101 - 200
        return 101 + ((200 - 101) / (150.4 - 55.5)) * (c - 55.5)
    elif c <= 250.4:
        # Range 201 - 300
        return 201 + ((300 - 201) / (250.4 - 150.5)) * (c - 150.5)
    else:
        # Range > 300
        return 301 + ((500 - 301) / (500 - 250.5)) * (c - 250.5)

def calculate_ispu_pm10(concentration):
    """
    Computes ISPU for PM10 based on Permen LHK No. P.14/2020.
    """
    c = float(concentration)
    if c <= 50:
        return c
    elif c <= 150:
        return 51 + ((100 - 51) / (150 - 51)) * (c - 51)
    elif c <= 350:
        return 101 + ((200 - 101) / (350 - 151)) * (c - 151)
    elif c <= 420:
        return 201 + ((300 - 201) / (420 - 351)) * (c - 351)
    else:
        return 301 + ((500 - 301) / (600 - 421)) * (c - 421)

def get_early_warning_status(ispu_val):
    """
    Returns Early Warning Status, Color Code, and Health Recommendation.
    """
    val = round(ispu_val, 1)
    if val <= 50:
        return {
            'ispu': val,
            'category': 'BAIK',
            'level': 'GOOD',
            'color': '#2ea043',  # Green
            'badge': 'success',
            'warning': 'Kualitas udara sangat baik. Tidak ada dampak kesehatan.',
            'action': 'Aktivitas luar ruangan aman untuk semua kelompok masyarakat dan pekerja industri.'
        }
    elif val <= 100:
        return {
            'ispu': val,
            'category': 'SEDANG',
            'level': 'MODERATE',
            'color': '#dbab09',  # Yellow
            'badge': 'warning',
            'warning': 'Kualitas udara dapat diterima bagi mayoritas orang.',
            'action': 'Kelompok sensitif (penderita asma, lansia, anak-anak) disarankan mengurangi paparan luar ruangan yang lama.'
        }
    elif val <= 200:
        return {
            'ispu': val,
            'category': 'TIDAK SEHAT',
            'level': 'UNHEALTHY',
            'color': '#e36209',  # Orange
            'badge': 'danger',
            'warning': 'EARLY WARNING: Kualitas udara tidak sehat bagi manusia dan populasi berisiko.',
            'action': 'Wajib menggunakan masker N95 di area terbuka industri Cilacap. Batasi aktivitas fisik di luar.'
        }
    elif val <= 300:
        return {
            'ispu': val,
            'category': 'SANGAT TIDAK SEHAT',
            'level': 'VERY UNHEALTHY',
            'color': '#cb2431',  # Red
            'badge': 'dark-danger',
            'warning': 'CRITICAL WARNING: Kualitas udara meningkatkan risiko gangguan pernapasan serius.',
            'action': 'Pekerja lapangan industri wajib meminimalkan durasi kerja di area terbuka. Aktifkan sistem pembersih udara indoor.'
        }
    else:
        return {
            'ispu': val,
            'category': 'BERBAHAYA',
            'level': 'HAZARDOUS',
            'color': '#8b0000',  # Dark Red
            'badge': 'hazardous',
            'warning': 'EMERGENCY WARNING: Tingkat pencemaran udara berbahaya bagi seluruh populasi!',
            'action': 'Hentikan aktivitas lapangan non-darurat. Gunakan APD respiratori khusus industri.'
        }

def evaluate_forecast_warning(pm25_pred, pm10_pred):
    """
    Evaluates multi-step predictions and returns comprehensive Early Warning Report.
    """
    ispu_pm25 = calculate_ispu_pm25(pm25_pred)
    ispu_pm10 = calculate_ispu_pm10(pm10_pred)
    max_ispu = max(ispu_pm25, ispu_pm10)
    
    status = get_early_warning_status(max_ispu)
    status['pm25_concentration'] = round(float(pm25_pred), 2)
    status['pm10_concentration'] = round(float(pm10_pred), 2)
    status['ispu_pm25'] = round(ispu_pm25, 1)
    status['ispu_pm10'] = round(ispu_pm10, 1)
    return status

if __name__ == "__main__":
    sample_warn = evaluate_forecast_warning(pm25_pred=45.2, pm10_pred=78.5)
    print("Early Warning System Test:")
    for k, v in sample_warn.items():
        print(f"  {k}: {v}")
