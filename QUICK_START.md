# Quick Start Guide

Get up and running with the Landsat-based deforestation detector in minutes.

## 🚀 Fast Track Installation

### Step 1: Install Python packages (2 min)

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The detector relies on scientific Python packages (`numpy`, `pandas`,
`matplotlib`) plus geospatial tooling (`rasterio`, `pyproj`, `shapely`,
`pystac-client`) to query the Landsat STAC catalog and process imagery locally.

### Step 2: Run the analysis script (1 min)

Execute the simple example to confirm everything works end-to-end:

```bash
python simple_example.py
```

The script downloads a few Landsat scenes from the Element84 STAC API, computes
NDVI for two sample locations in Oregon, and saves a plot named
`simple_results.png`.

### Step 3: Launch the Streamlit app (optional)

```bash
streamlit run streamlit_app.py
```

Upload your own CSV of locations—or use the bundled `locations.csv`—and click
**Run analysis** to generate time-series charts, an interactive folium map, and a
summary table.

---

## 📁 Files You Need

Make sure these files are in the same folder:

1. **deforestation_detector.py** – Core Landsat NDVI workflow
2. **locations.csv** – Sample coordinates for testing
3. **streamlit_app.py** – Streamlit dashboard
4. **simple_example.py** – Minimal command-line example

---

## 💡 New to Python?

### Install Python

- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python3`
- **Linux**: `sudo apt-get install python3-pip`

### Run Commands

- **Windows**: Use Command Prompt or PowerShell
- **macOS/Linux**: Use Terminal

---

## 🗺️ Sample Locations (Oregon, USA)

The provided `locations.csv` contains two areas for validation:

1. **Willamette National Forest** – Logging activity
2. **Crater Lake National Park** – Protected area

---

## 📊 Expected Results

Running the example or Streamlit app produces:

1. **NDVI plot (`simple_results.png`)** – Visualises vegetation trends
2. **Interactive map** – Colour-coded markers by deforestation status
3. **Console summary** – Change percentages and detection status

---

## ❓ Troubleshooting

### "ModuleNotFoundError"

**Fix**: Install dependencies with `pip install -r requirements.txt`.

### "No NDVI observations"

**Fix**: Increase the buffer radius or widen the start/end years to capture more
Landsat scenes. You can also raise the **Max cloud cover** slider in the
Streamlit sidebar to accept scenes with partial cloud contamination.

### Slow downloads

**Fix**: Run the analysis with fewer locations or a smaller buffer to reduce the
amount of imagery requested from AWS.

---

## ✅ Success Checklist

- [ ] Python installed
- [ ] Dependencies installed
- [ ] `simple_example.py` runs without errors
- [ ] `simple_results.png` generated
- [ ] Streamlit app loads (optional)
- [ ] Results match expectations

Estimated total time: **5–10 minutes**

You're ready to monitor deforestation with open Landsat data! 🌲📊
