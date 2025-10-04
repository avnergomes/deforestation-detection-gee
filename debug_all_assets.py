"""Debug script to see ALL assets in Landsat scenes."""

import sys
from pystac_client import Client
from shapely.geometry import Polygon, mapping
import pyproj
import numpy as np

def inspect_all_assets():
    """Inspect ALL assets in Landsat scenes to find the correct band names."""
    
    print("=" * 70)
    print("🔍 FULL ASSET INSPECTION")
    print("=" * 70)
    
    # Create polygon for Mount Hood
    geod = pyproj.Geod(ellps="WGS84")
    latitude, longitude = 45.3726, -121.6959
    buffer_km = 5
    
    azimuths_full = np.linspace(0, 360, num=73)
    lon_arr_full = np.full_like(azimuths_full, longitude, dtype=float)
    lat_arr_full = np.full_like(azimuths_full, latitude, dtype=float)
    distances_full = np.full_like(azimuths_full, buffer_km * 1000.0, dtype=float)
    lon_points_full, lat_points_full, _ = geod.fwd(lon_arr_full, lat_arr_full, azimuths_full, distances_full)
    ring = np.column_stack((lon_points_full, lat_points_full))
    polygon = Polygon(ring)
    
    # Search for scenes
    client = Client.open("https://earth-search.aws.element84.com/v1")
    
    search = client.search(
        collections=["landsat-c2-l2"],
        datetime="2020-01-01/2024-12-31",
        query={"eo:cloud_cover": {"lt": 70}},
        intersects=mapping(polygon),
    )
    
    items = list(search.get_items())
    print(f"\n✅ Found {len(items)} total scenes")
    
    if not items:
        print("❌ No scenes found!")
        return
    
    # Inspect first scene in detail
    print("\n" + "=" * 70)
    print("📋 DETAILED INSPECTION OF FIRST SCENE:")
    print("=" * 70)
    
    item = items[0]
    print(f"\nScene ID: {item.id}")
    print(f"Date: {item.properties.get('datetime', 'Unknown')}")
    print(f"Cloud Cover: {item.properties.get('eo:cloud_cover', 'Unknown')}%")
    
    print(f"\n📦 ALL ASSETS ({len(item.assets)} total):")
    print("-" * 70)
    
    for asset_name, asset in sorted(item.assets.items()):
        print(f"\n  Asset: {asset_name}")
        print(f"    Type: {asset.media_type if hasattr(asset, 'media_type') else 'Unknown'}")
        print(f"    Title: {asset.title if hasattr(asset, 'title') else 'N/A'}")
        if hasattr(asset, 'extra_fields'):
            eo_bands = asset.extra_fields.get('eo:bands', [])
            if eo_bands:
                for band in eo_bands:
                    print(f"    Band info: {band}")
    
    # Look for common band patterns
    print("\n" + "=" * 70)
    print("🔍 SEARCHING FOR BAND PATTERNS:")
    print("=" * 70)
    
    asset_keys = list(item.assets.keys())
    
    print("\nAll asset keys:")
    for key in sorted(asset_keys):
        print(f"  - {key}")
    
    # Check for various band naming patterns
    patterns = {
        "SR_B*": [k for k in asset_keys if k.startswith('SR_B')],
        "B*": [k for k in asset_keys if k.startswith('B') and len(k) <= 3],
        "*band*": [k for k in asset_keys if 'band' in k.lower()],
        "*red*": [k for k in asset_keys if 'red' in k.lower()],
        "*nir*": [k for k in asset_keys if 'nir' in k.lower()],
        "*blue*": [k for k in asset_keys if 'blue' in k.lower()],
        "*green*": [k for k in asset_keys if 'green' in k.lower()],
        "coastal*": [k for k in asset_keys if 'coastal' in k.lower()],
        "swir*": [k for k in asset_keys if 'swir' in k.lower()],
    }
    
    print("\nPattern matches:")
    for pattern, matches in patterns.items():
        if matches:
            print(f"  {pattern}: {matches}")
    
    # Check multiple scenes to see consistency
    print("\n" + "=" * 70)
    print("📊 CHECKING FIRST 10 SCENES FOR CONSISTENCY:")
    print("=" * 70)
    
    from collections import Counter
    all_asset_names = []
    
    for idx, item in enumerate(items[:10]):
        asset_names = list(item.assets.keys())
        all_asset_names.extend(asset_names)
        print(f"\nScene {idx+1} ({item.id}):")
        print(f"  Asset count: {len(asset_names)}")
        
        # Show band-like assets
        band_like = [k for k in asset_names if 
                     k.startswith('SR_') or k.startswith('B') or 
                     'nir' in k.lower() or 'red' in k.lower() or 
                     'blue' in k.lower() or 'green' in k.lower()]
        if band_like:
            print(f"  Band-like assets: {sorted(band_like)}")
    
    print("\n" + "=" * 70)
    print("📈 MOST COMMON ASSET NAMES (across first 10 scenes):")
    print("=" * 70)
    
    asset_counter = Counter(all_asset_names)
    for asset_name, count in asset_counter.most_common(30):
        print(f"  {asset_name}: {count}/10 scenes")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    inspect_all_assets()
