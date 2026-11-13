# Paket Tugas Besar Analitik dan Visualisasi Data

**Judul:** Prediksi Risiko Eutrofikasi Danau Menggunakan Data Science  
**Nama:** Septian Dwi Saputra  
**NIM:** 411232056  
**Mata kuliah:** Analitik dan Visualisasi Data

## Cara menjalankan di macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python analisis_eutrofikasi.py
```

Untuk membuka notebook:

```bash
python -m pip install -r requirements.txt
jupyter notebook Notebook_Analisis_Eutrofikasi.ipynb
```

```python
import joblib
import pandas as pd

bundle = joblib.load("outputs/model_klasifikasi_eutrofikasi.joblib")
model = bundle["model"]

observasi_baru = pd.DataFrame([{
    "total_nitrogen_mg_l": 0.45,
    "water_temperature_c": 27.0,
    "dissolved_oxygen_mg_l": 7.5,
    "annual_rainfall_mm": 1800,
    "agricultural_land_use_pct": 35,
    "urban_land_use_pct": 12,
    "mean_lake_depth_m": 6.0,
    "ph": 7.2,
}])

print("Prediksi status trofik:", model.predict(observasi_baru)[0])
print("Probabilitas per kelas:", model.predict_proba(observasi_baru)[0])
```

Contoh angka di atas hanya ilustrasi pemanggilan kode, bukan hasil uji atau rekomendasi kualitas air.

- Data.gov: https://catalog.data.gov/dataset/estimates-of-lake-nitrogen-phosphorus-and-chlorophyll-a-concentrations-to-characterize-har
- DOI dataset: https://doi.org/10.23719/1529835
- Water Quality Portal: https://www.waterqualitydata.us/

## Referensi utama

- Carlson, R. E. (1977). A trophic state index for lakes. _Limnology and Oceanography, 22_(2), 361–369. https://doi.org/10.4319/lo.1977.22.2.0361
- U.S. EPA. (2000). _Nutrient Criteria Technical Guidance Manual: Lakes and Reservoirs_. EPA 822-B-00-001.
- U.S. EPA. (2023). _Estimates of lake nitrogen, phosphorus, and chlorophyll-a concentrations to characterize harmful algal bloom risk across the United States_. https://doi.org/10.23719/1529835
