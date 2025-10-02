# Quick Start Guide

Get up and running with deforestation detection in 5 minutes!

## 🚀 Fast Track Installation

### Step 1: Install Python packages (2 min)

```bash
pip install earthengine-api geemap pandas matplotlib folium numpy
```

### Step 2: Authenticate Google Earth Engine (2 min)

Open Python and run:

```python
import ee
ee.Authenticate()
```

This opens a browser for Google login. Follow the prompts and you're done!

### Step 3: Run the analysis (1 min)

```bash
python deforestation_detector.py
```

That's it! 🎉

---

## 📁 Files You Need

Download these 3 files to the same folder:

1. **deforestation_detector.py** - Main script
2. **locations.csv** - Your coordinates (already includes 2 Oregon locations)
3. **requirements.txt** - Package list (optional)

---

## 💡 First Time Using Python?

### Install Python

**Windows**: Download from [python.org](https://www.python.org/downloads/)  
**Mac**: `brew install python3`  
**Linux**: `sudo apt-get install python3-pip`

### Run Commands

**Windows**: Use Command Prompt or PowerShell  
**Mac/Linux**: Use Terminal

---

## 🗺️ Your Two Test Locations (Oregon, USA)

The included `locations.csv` has:

1. **Willamette National Forest** - Logging area (should show deforestation)
2. **Crater Lake National Park** - Protected area (should be stable)

---

## 📊 Expected Results

After ~5 minutes of processing, you'll get:

1. **ndvi_time_series.png**
   - Shows NDVI trends from 2015-2024
   - Red alert if deforestation detected

2. **deforestation_map.html**
   - Interactive map (open in browser)
   - Red markers = deforestation
   - Green markers = stable forest

3. **Console Report**
   - Statistics and change percentages
   - Clear detection status

---

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'ee'"

**Fix**: Run `pip install earthengine-api`

---

### "Please authenticate Earth Engine"

**Fix**: 
```python
import ee
ee.Authenticate()
```

---

### "No images found"

**Fix**: Location might be too cloudy. Try different coordinates or expand date range in code:

```python
detector = DeforestationDetector(start_year=2013, end_year=2024)
```

---

## 🎯 Test with Your Own Locations

Edit `locations.csv`:

```csv
location_name,latitude,longitude
My Forest,45.5231,-122.6765
My Field,45.4234,-122.5432
```

Then run: `python deforestation_detector.py`

---

## 📞 Need Help?

**Common fixes**:
- Update packages: `pip install --upgrade earthengine-api`
- Re-authenticate: `ee.Authenticate()`
- Check coordinates are decimal degrees (not DMS)
- Ensure internet connection is stable

**Still stuck?** Check the full README.md for detailed documentation.

---

## ✅ Success Checklist

- [ ] Python installed
- [ ] Packages installed
- [ ] Earth Engine authenticated
- [ ] locations.csv in same folder as script
- [ ] Script runs without errors
- [ ] PNG and HTML files generated
- [ ] Results make sense

---

**Estimated Total Time**: 5-10 minutes  
**Processing Time**: ~3-5 minutes per location  
**Output Size**: ~2-5 MB

Now you're ready to detect deforestation! 🌲📊
