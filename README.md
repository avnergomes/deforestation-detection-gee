# Deforestation Detection using Google Earth Engine

**Author:** Avner Gomes  
**Client:** Kunal Sachdeva  
**Project:** Land Cover Change Detection with NDVI Analysis  

---

## 📋 Project Overview

This project provides a Python-based solution for detecting land cover changes and deforestation using Google Earth Engine (GEE) and NDVI (Normalized Difference Vegetation Index) analysis. The system analyzes satellite imagery from Landsat missions to identify vegetation changes over time.

### Key Features

- ✅ **Flexible Input**: Accepts CSV files with latitude/longitude coordinates
- ✅ **NDVI Time Series**: Extracts and analyzes vegetation index data over multiple years
- ✅ **Deforestation Detection**: Automatically identifies significant vegetation loss
- ✅ **Interactive Visualizations**: Generates time series plots and interactive maps
- ✅ **Comprehensive Reports**: Provides detailed analysis with quantitative metrics

---

## 🎯 Deliverables for Milestone 1

### ✓ Baseline Code
- Complete Python script with Google Earth Engine integration
- Object-oriented design for easy maintenance and extension
- Well-documented code with inline comments

### ✓ Two Test Locations in Oregon, USA
1. **Willamette National Forest - Logging Area**
   - Latitude: 44.2145°N
   - Longitude: 122.1567°W
   - Expected: Deforestation detected (logging activity)

2. **Crater Lake National Park - Protected Area**
   - Latitude: 42.9446°N
   - Longitude: 122.1090°W
   - Expected: Stable forest cover (protected area)

### ✓ NDVI Data Processing
- Automatic download of Landsat imagery (2015-2024)
- Cloud filtering (< 20% cloud cover)
- NDVI calculation and aggregation

### ✓ Visualizations
- Time series plots showing NDVI changes over time
- Annual averages with trend lines
- Interactive map with location markers and status indicators

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Google Earth Engine account
- Internet connection for satellite data access

### Step 1: Install Required Packages

```bash
pip install earthengine-api
pip install geemap
pip install pandas
pip install matplotlib
pip install folium
pip install numpy
```

Or install all at once:

```bash
pip install earthengine-api geemap pandas matplotlib folium numpy
```

### Step 2: Authenticate Google Earth Engine

**First time only:**

```python
import ee
ee.Authenticate()
```

This will open a browser window for you to authenticate with your Google account and authorize Earth Engine access. Follow the instructions and paste the authorization code back into the terminal.

After authentication is complete, you can initialize Earth Engine in your code:

```python
ee.Initialize()
```

### Step 3: Verify Installation

```python
import ee
ee.Initialize()
print("✓ Google Earth Engine is ready!")
```

---

## 📂 Project Structure

```
deforestation-detection/
│
├── deforestation_detector.py    # Main Python script
├── locations.csv                # Input CSV with coordinates
├── README.md                    # This file
│
├── outputs/                     # Generated outputs (created automatically)
│   ├── ndvi_time_series.png    # Time series plots
│   └── deforestation_map.html  # Interactive map
│
└── requirements.txt             # Python dependencies
```

---

## 💻 Usage

### Basic Usage

1. **Prepare your CSV file** with the following format:

```csv
location_name,latitude,longitude,description
Location 1,44.2145,-122.1567,Description of location 1
Location 2,42.9446,-122.1090,Description of location 2
```

2. **Run the script:**

```bash
python deforestation_detector.py
```

3. **View the outputs:**
   - `ndvi_time_series.png` - Visual plots of NDVI trends
   - `deforestation_map.html` - Open in browser for interactive map
   - Console output with detailed analysis report

### Advanced Usage

#### Customize Analysis Period

```python
from deforestation_detector import DeforestationDetector

# Analyze different time period
detector = DeforestationDetector(start_year=2010, end_year=2023)
```

#### Adjust Buffer Size

```python
# Create 10km buffer instead of default 5km
geometry = detector.create_buffer_polygon(latitude, longitude, buffer_km=10)
```

#### Process Single Location

```python
detector = DeforestationDetector()

# Define location
latitude = 44.2145
longitude = -122.1567
location_name = "My Forest Area"

# Create geometry
geometry = detector.create_buffer_polygon(latitude, longitude)

# Extract NDVI data
ndvi_df = detector.extract_ndvi_time_series(geometry, location_name)

# Analyze
analysis = detector.analyze_deforestation(ndvi_df)
print(analysis)
```

---

## 📊 Understanding the Outputs

### 1. Time Series Plots

The generated plots show:
- **Green dots**: Individual NDVI measurements from satellite images
- **Dark green line**: Smoothed trend (5-point moving average)
- **Red markers**: Annual average NDVI values
- **Title**: Overall change percentage and deforestation status

#### Interpreting NDVI Values:
- **0.8 - 1.0**: Dense, healthy vegetation
- **0.6 - 0.8**: Moderate vegetation
- **0.4 - 0.6**: Sparse vegetation
- **< 0.4**: Little to no vegetation

### 2. Interactive Map

Open `deforestation_map.html` in any web browser to see:
- **Green markers**: Stable or improving vegetation
- **Red markers**: Deforestation detected
- **Circles**: 5km buffer zones around each location
- **Popups**: Click markers for detailed statistics

### 3. Analysis Report

Console output includes:
- Number of satellite images analyzed
- Initial and final NDVI values
- Percentage change over the analysis period
- Deforestation detection status

---

## 🔧 Technical Details

### Data Sources

- **Landsat 8** (2013-present): Primary data source
- **Landsat 7** (1999-present): Additional temporal coverage
- **Collection 2 Level-2**: Surface reflectance products with atmospheric correction

### NDVI Calculation

```
NDVI = (NIR - RED) / (NIR + RED)
```

Where:
- NIR = Near-Infrared band (Landsat Band 5)
- RED = Red band (Landsat Band 4)

### Deforestation Detection Algorithm

1. Calculate annual average NDVI for each year
2. Compare first 3 years average vs. last 3 years average
3. Calculate percentage change
4. **Threshold**: > 15% decrease indicates deforestation

### Processing Workflow

```
1. Load CSV coordinates
2. Create 5km buffer around each point
3. Filter Landsat images (2015-2024, < 20% clouds)
4. Calculate NDVI for each image
5. Extract mean NDVI for each location
6. Aggregate into time series
7. Analyze trends and detect changes
8. Generate visualizations and report
```

---

## 📈 Interpreting Results

### Deforestation Indicators

| Change | Status | Interpretation |
|--------|--------|----------------|
| < -15% | ⚠ DEFORESTATION | Significant vegetation loss detected |
| -15% to -5% | ⚡ DECLINING | Moderate vegetation decrease |
| -5% to +5% | ✓ STABLE | No significant change |
| > +5% | ✓ IMPROVING | Vegetation increase (regrowth/reforestation) |

### Example Interpretation

```
Location: Willamette National Forest - Logging Area
Initial NDVI: 0.756
Final NDVI: 0.612
Change: -19.0%
Status: ⚠ DEFORESTATION DETECTED
```

**Interpretation**: This area has experienced significant vegetation loss (~19%), consistent with logging activities. The NDVI dropped from dense forest (0.756) to moderate vegetation (0.612).

---

## 🔍 Troubleshooting

### Common Issues

**1. Authentication Error**

```
Error: Please authenticate Earth Engine
```

**Solution:**
```python
import ee
ee.Authenticate()  # Run this once
```

---

**2. No Images Found**

```
Found 0 suitable images
```

**Possible causes:**
- Location is too cloudy (>20% cloud cover year-round)
- Coordinates are in ocean or extreme latitudes
- Date range too narrow

**Solution:**
- Adjust cloud cover threshold
- Verify coordinates are correct
- Expand date range

---

**3. Memory Error**

```
Computation timed out or exceeded memory limits
```

**Solution:**
- Reduce buffer size (use 3km instead of 5km)
- Shorten date range
- Process fewer locations at once

---

**4. Module Import Error**

```
ModuleNotFoundError: No module named 'ee'
```

**Solution:**
```bash
pip install earthengine-api
```

---

## 🔄 Customization Options

### Modify Deforestation Threshold

In the `analyze_deforestation()` method, change:

```python
# Current threshold: 15% decrease
deforestation_detected = change_percentage < -15

# More sensitive (10% decrease)
deforestation_detected = change_percentage < -10

# Less sensitive (20% decrease)
deforestation_detected = change_percentage < -20
```

### Add Different Vegetation Indices

Besides NDVI, you can calculate:

**EVI (Enhanced Vegetation Index)**:
```python
def calculate_evi(self, image):
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': image.select('B5'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }
    ).rename('EVI')
    return image.addBands(evi)
```

**SAVI (Soil-Adjusted Vegetation Index)**:
```python
def calculate_savi(self, image):
    L = 0.5  # Soil brightness correction factor
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + L)) * (1 + L)',
        {
            'NIR': image.select('B5'),
            'RED': image.select('B4'),
            'L': L
        }
    ).rename('SAVI')
    return image.addBands(savi)
```

---

## 📝 CSV Format Specifications

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| location_name | string | Name of the location | "Forest Area 1" |
| latitude | float | Latitude in decimal degrees | 44.2145 |
| longitude | float | Longitude in decimal degrees | -122.1567 |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| description | string | Additional information about the location |
| expected_status | string | Expected outcome (for validation) |
| buffer_km | float | Custom buffer size for this location |

### Example CSV

```csv
location_name,latitude,longitude,description,buffer_km
Primary Forest,45.5231,-122.6765,Old growth forest area,5
Logged Area,45.4234,-122.5432,Commercial logging zone,7
Regrowth Zone,45.3876,-122.4987,Replanted area from 2015,5
```

---

## 🎓 Background & Methodology

### Why NDVI?

NDVI is the most widely used vegetation index because:
1. **Robust**: Works across different ecosystems and conditions
2. **Simple**: Easy to calculate and interpret
3. **Proven**: Decades of scientific validation
4. **Sensitive**: Detects subtle vegetation changes

### Limitations

- **Cloud cover**: Can limit available imagery
- **Seasonal variation**: Natural NDVI fluctuation throughout the year
- **Spatial resolution**: 30m pixels may miss small clearings
- **Atmospheric effects**: Partially corrected in Level-2 products

---

## 📞 Support & Contact

**Developer**: Avner Gomes  
**Expertise**: Forestry Engineering + Data Science  
**Specialties**: Remote Sensing, GEE, Vegetation Indices, Change Detection

For questions about:
- Code modifications
- Additional features
- Different vegetation indices (EVI, SAVI)
- Custom analysis regions
- Carbon credit feasibility studies

---

## 📜 License & Usage

This code is provided for the client's use. Feel free to:
- ✅ Modify the code for your needs
- ✅ Use it for research or commercial projects
- ✅ Share with your team

---

## 🚀 Next Steps (Milestone 2 Preparation)

Potential enhancements for the next phase:
1. Compare NDVI vs. EVI vs. SAVI performance
2. Add seasonal decomposition analysis
3. Export results to CSV/Excel
4. Generate PDF reports with maps and charts
5. Add more sophisticated change detection algorithms
6. Integration with GIS software (QGIS, ArcGIS)
7. Automated alert system for significant changes

---

## ✅ Milestone 1 Checklist

- [x] Python script with GEE integration
- [x] CSV input functionality
- [x] Two locations identified (Oregon, same state)
- [x] NDVI data extraction working
- [x] Time series plots generated
- [x] Interactive map created
- [x] Analysis report with metrics
- [x] Complete documentation
- [x] Code ready for modifications (Milestone 2)

---

**Status**: Ready for review and testing  
**Last Updated**: October 2025  
**Version**: 1.0
