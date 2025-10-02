# Deforestation Detection with Open Landsat Imagery

**Author:** Avner Gomes  
**Client:** Kunal Sachdeva  
**Project:** Land Cover Change Detection with NDVI Analysis

---

## 📋 Project Overview

This project provides a Python-based workflow for detecting land cover changes
and deforestation using freely available Landsat imagery. The workflow talks
directly to the public Landsat Collection 2 Level 2 archive published on AWS via
the Element84 STAC API and computes NDVI locally using `rasterio`.

### Key Features

- ✅ **Flexible Input**: Accept CSV files with latitude/longitude coordinates.
- ✅ **NDVI Time Series**: Extracts and analyzes vegetation index data over
  multiple years.
- ✅ **Deforestation Detection**: Automatically flags significant vegetation
  loss using NDVI trends.
- ✅ **Interactive Visualizations**: Generates time series plots and interactive
  maps for quick exploration.
- ✅ **Streamlit Dashboard**: Runs without any private credentials and is ready
  for deployment on Streamlit Community Cloud.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9 or higher
- Internet connection (each analysis downloads Landsat COG tiles from AWS)

The Landsat archive hosted on AWS requires the `AWS_REQUEST_PAYER=requester`
header. The detector automatically sets this header for you when fetching
imagery, so no additional configuration is necessary for typical usage.

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

The requirements file includes `rasterio`, `pystac-client`, and Streamlit along
with the scientific Python stack needed for the NDVI workflow.

### Step 2: Verify the detector locally

```bash
python - <<'PY'
from deforestation_detector import DeforestationDetector
detector = DeforestationDetector(start_year=2018, end_year=2024)
print("✓ Landsat STAC client is ready")
PY
```

If the script prints the confirmation message, the STAC client was initialised
successfully and network access is working.

---

## 📂 Project Structure

```
deforestation-detection-gee/
│
├── deforestation_detector.py    # Core Landsat NDVI workflow
├── locations.csv                # Sample locations for quick testing
├── README.md                    # Documentation (this file)
├── requirements.txt             # Python dependencies
├── streamlit_app.py             # Streamlit dashboard
└── simple_example.py            # Minimal script usage example
```

---

## 💻 Usage

### Command-line workflow

1. **Prepare your CSV file** with the following columns:

   ```csv
   location_name,latitude,longitude,description
   Location 1,44.2145,-122.1567,Description of location 1
   Location 2,42.9446,-122.1090,Description of location 2
   ```

2. **Run your custom analysis**:

   ```python
   from deforestation_detector import DeforestationDetector
   import pandas as pd

   detector = DeforestationDetector(start_year=2015, end_year=2024)
   locations = pd.read_csv("locations.csv")

   results = {}
   for _, row in locations.iterrows():
       geom = detector.create_buffer_polygon(row.latitude, row.longitude, buffer_km=5)
       results[row.location_name] = detector.extract_ndvi_time_series(geom, row.location_name)

   detector.plot_ndvi_time_series(results)
   ```

3. **Interpret the output**: Review the console messages, the generated Matplotlib
   figure, and the optional folium map (if you call `create_interactive_map`).

### Streamlit Dashboard

Run the interactive dashboard locally or deploy it to Streamlit Community Cloud
without any extra secrets:

```bash
streamlit run streamlit_app.py
```

Upload a CSV (or use the bundled `locations.csv`) and click **Run analysis**.
The app downloads the required Landsat scenes, calculates NDVI per observation,
and displays charts, maps, and summary statistics. Use the **Max cloud cover**
slider in the sidebar if you need to relax or tighten the quality filter applied
to the STAC search (lower values exclude cloudy scenes).

---

## 📊 Understanding the Outputs

- **NDVI Time Series Plot** – Shows individual observations, a smoothed trend,
  and annual averages.
- **Summary Table** – Lists statistics per location, highlighting areas with
  significant NDVI decline.
- **Interactive Map** – Uses folium to colour-code locations based on the
  detected trend.

Deforestation is flagged when the mean NDVI in the final years of the series is
more than 15 % lower than the mean of the initial years. You can adapt this
threshold by modifying `analyze_deforestation` in `deforestation_detector.py`.

---

## 🙋‍♀️ FAQ

**Why is the first run slow?**  
Scenes are downloaded on demand from AWS. Subsequent runs that reuse the same
locations and time range benefit from HTTP caching handled by GDAL.

**Can I use Sentinel-2 or another dataset?**  
Yes. Update `_search_landsat_scenes` to point to a different STAC collection and
adjust the band names and scaling factors accordingly.

## 🛠️ Troubleshooting

- **`RasterioIOError` when downloading assets** – Verify that outbound HTTPS
  traffic is allowed and retry; intermittent network hiccups can happen when
  streaming large rasters.
- **Empty NDVI results** – Check that the buffer size covers your area of
  interest and that Landsat acquired cloud-free observations in the selected
  period.
- **Streamlit session hits memory limits** – Reduce the buffer radius or the
  number of locations so each analysis downloads a smaller cutout.

---

## 📄 License

This project is distributed under the terms of the MIT License. See
[`LICENSE`](LICENSE) for details.

