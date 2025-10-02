"""
Simple Example - Minimal Code to Detect Deforestation
Perfect for quick tests and learning the workflow
"""

import ee
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# SETUP
# ============================================================================

# Initialize Earth Engine
ee.Initialize()
print("✓ Earth Engine initialized\n")

# ============================================================================
# DEFINE YOUR LOCATIONS
# ============================================================================

# Option 1: Define locations directly in code (no CSV needed)
locations = [
    {
        'name': 'Logging Area',
        'lat': 44.2145,
        'lon': -122.1567
    },
    {
        'name': 'Protected Forest',
        'lat': 42.9446,
        'lon': -122.1090
    }
]

# Option 2: Load from CSV (comment out Option 1 if using this)
# df = pd.read_csv('locations.csv')
# locations = df.to_dict('records')

# ============================================================================
# PROCESS EACH LOCATION
# ============================================================================

results = {}

for loc in locations:
    print(f"Processing: {loc['name']}...")
    
    # Create 5km buffer around point
    point = ee.Geometry.Point([loc['lon'], loc['lat']])
    area = point.buffer(5000)
    
    # Get Landsat images (2015-2024)
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(area) \
        .filterDate('2015-01-01', '2024-12-31') \
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
    
    print(f"  Found {collection.size().getInfo()} images")
    
    # Calculate NDVI and extract time series
    def get_ndvi(image):
        # Apply scale factors
        optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        image = image.addBands(optical, None, True)
        
        # Calculate NDVI
        ndvi = image.normalizedDifference(['B5', 'B4'])
        
        # Get mean NDVI for area
        stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area,
            scale=30
        )
        
        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'ndvi': stats.get('nd')
        })
    
    features = collection.map(get_ndvi).getInfo()['features']
    
    # Convert to pandas DataFrame
    data = []
    for f in features:
        if f['properties']['ndvi'] is not None:
            data.append({
                'date': f['properties']['date'],
                'ndvi': f['properties']['ndvi']
            })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    results[loc['name']] = df
    print(f"  ✓ Extracted {len(df)} measurements\n")

# ============================================================================
# ANALYZE & VISUALIZE
# ============================================================================

fig, axes = plt.subplots(len(results), 1, figsize=(12, 5*len(results)))

if len(results) == 1:
    axes = [axes]

for idx, (name, df) in enumerate(results.items()):
    ax = axes[idx]
    
    # Calculate change
    first_year = df[df['date'].dt.year <= 2017]['ndvi'].mean()
    last_year = df[df['date'].dt.year >= 2022]['ndvi'].mean()
    change = ((last_year - first_year) / first_year) * 100
    
    # Determine status
    if change < -15:
        status = "⚠ DEFORESTATION DETECTED"
        color = 'red'
    elif change < -5:
        status = "⚡ DECLINING"
        color = 'orange'
    else:
        status = "✓ STABLE"
        color = 'green'
    
    # Plot
    ax.scatter(df['date'], df['ndvi'], alpha=0.5, s=20)
    ax.plot(df['date'], df['ndvi'].rolling(5).mean(), linewidth=2, color=color)
    
    ax.set_title(f'{name}\nChange: {change:.1f}% | {status}', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('NDVI')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Print summary
    print(f"{name}:")
    print(f"  Initial NDVI: {first_year:.3f}")
    print(f"  Final NDVI: {last_year:.3f}")
    print(f"  Change: {change:+.1f}%")
    print(f"  Status: {status}\n")

plt.tight_layout()
plt.savefig('simple_results.png', dpi=200, bbox_inches='tight')
print("✅ Results saved to: simple_results.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nAnalyzed {len(results)} location(s)")
print(f"Time period: 2015-2024")
print(f"Output: simple_results.png")
print("\n" + "="*60)
