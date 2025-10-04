"""Debug script to inspect actual Landsat scene band names."""

import sys
from pystac_client import Client
from shapely.geometry import Polygon, mapping
import pyproj

def inspect_landsat_bands():
    """Inspect what bands are actually in the Landsat scenes."""
    
    print("=" * 70)
    print("🔍 LANDSAT BAND INSPECTION")
    print("=" * 70)
    
    # Create polygon for Mount Hood
    geod = pyproj.Geod(ellps="WGS84")
    latitude, longitude = 45.3726, -121.6959
    buffer_km = 5
    
    azimuths = [0, 90, 180, 270]
    lon_arr = [longitude] * 4
    lat_arr = [latitude] * 4
    distances = [buffer_km * 1000.0] * 4
    lon_points, lat_points, _ = geod.fwd(lon_arr, lat_arr, azimuths, distances)
    
    import numpy as np
    azimuths_full = np.linspace(0, 360, num=73)
    lon_arr_full = np.full_like(azimuths_full, longitude, dtype=float)
    lat_arr_full = np.full_like(azimuths_full, latitude, dtype=float)
    distances_full = np.full_like(azimuths_full, buffer_km * 1000.0, dtype=float)
    lon_points_full, lat_points_full, _ = geod.fwd(lon_arr_full, lat_arr_full, azimuths_full, distances_full)
    ring = np.column_stack((lon_points_full, lat_points_full))
    polygon = Polygon(ring)
    
    print(f"\n📍 Location: Mount Hood")
    print(f"   Coordinates: {latitude}, {longitude}")
    print(f"   Buffer: {buffer_km} km")
    
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
    
    # Analyze first 5 scenes to see what bands they have
    print("\n" + "=" * 70)
    print("🔬 INSPECTING FIRST 5 SCENES:")
    print("=" * 70)
    
    for idx, item in enumerate(items[:5]):
        print(f"\n--- Scene {idx + 1}: {item.id} ---")
        print(f"Date: {item.properties.get('datetime', 'Unknown')}")
        print(f"Cloud Cover: {item.properties.get('eo:cloud_cover', 'Unknown')}%")
        
        # Get all asset keys
        asset_keys = list(item.assets.keys())
        print(f"Total assets: {len(asset_keys)}")
        
        # Filter for SR (Surface Reflectance) bands
        sr_bands = [key for key in asset_keys if key.startswith('SR_B')]
        print(f"SR Bands available: {sorted(sr_bands)}")
        
        # Check what we're looking for
        l89_bands = ["SR_B5", "SR_B4", "SR_B3", "SR_B2"]
        l457_bands = ["SR_B4", "SR_B3", "SR_B2", "SR_B1"]
        
        has_l89 = all(band in asset_keys for band in l89_bands)
        has_l457 = all(band in asset_keys for band in l457_bands)
        
        print(f"Has L8/9 bands {l89_bands}: {has_l89}")
        print(f"Has L4/5/7 bands {l457_bands}: {has_l457}")
        
        if not has_l89 and not has_l457:
            print("⚠️  PROBLEM: Scene has neither L8/9 nor L4/5/7 complete band set!")
            missing_l89 = [b for b in l89_bands if b not in asset_keys]
            missing_l457 = [b for b in l457_bands if b not in asset_keys]
            print(f"   Missing for L8/9: {missing_l89}")
            print(f"   Missing for L4/5/7: {missing_l457}")
    
    # Summary analysis
    print("\n" + "=" * 70)
    print("📊 SUMMARY ANALYSIS:")
    print("=" * 70)
    
    l89_count = 0
    l457_count = 0
    neither_count = 0
    
    for item in items:
        asset_keys = list(item.assets.keys())
        has_l89 = all(band in asset_keys for band in ["SR_B5", "SR_B4", "SR_B3", "SR_B2"])
        has_l457 = all(band in asset_keys for band in ["SR_B4", "SR_B3", "SR_B2", "SR_B1"])
        
        if has_l89:
            l89_count += 1
        elif has_l457:
            l457_count += 1
        else:
            neither_count += 1
    
    print(f"Scenes with L8/9 bands (SR_B5, SR_B4, SR_B3, SR_B2): {l89_count}")
    print(f"Scenes with L4/5/7 bands (SR_B4, SR_B3, SR_B2, SR_B1): {l457_count}")
    print(f"Scenes with neither complete set: {neither_count}")
    print(f"Total scenes: {len(items)}")
    
    if l89_count + l457_count == 0:
        print("\n❌ CRITICAL: NO scenes have the expected band combinations!")
        print("   This explains why 0 scenes are being prepared.")
        print("\n🔍 Let's check what bands are most common:")
        
        from collections import Counter
        all_bands = []
        for item in items[:20]:  # Check first 20
            all_bands.extend([k for k in item.assets.keys() if k.startswith('SR_B')])
        
        band_counts = Counter(all_bands)
        print("\nMost common SR bands (from first 20 scenes):")
        for band, count in band_counts.most_common(10):
            print(f"  {band}: appears in {count} scenes")
    else:
        print(f"\n✅ Found {l89_count + l457_count} usable scenes!")
        print("   The multi-satellite support should be working.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    inspect_landsat_bands()
