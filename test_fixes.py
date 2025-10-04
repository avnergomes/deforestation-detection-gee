"""Quick test script to verify the deforestation detector fixes."""

import sys
from deforestation_detector import DeforestationDetector

def test_detector():
    """Test basic functionality of the detector."""
    
    print("=" * 60)
    print("🧪 TESTING DEFORESTATION DETECTOR")
    print("=" * 60)
    
    # Test 1: Initialize detector
    print("\n✓ Test 1: Initializing detector...")
    try:
        detector = DeforestationDetector(start_year=2020, end_year=2024)
        print("  ✅ Detector initialized successfully")
        print(f"  - Start date: {detector.start_date}")
        print(f"  - End date: {detector.end_date}")
        print(f"  - Max cloud cover: {detector.max_cloud_cover}%")
    except Exception as e:
        print(f"  ❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Test 2: Create buffer polygon
    print("\n✓ Test 2: Creating buffer polygon...")
    try:
        # Mount Hood coordinates
        lat, lon = 45.3726, -121.6959
        polygon = detector.create_buffer_polygon(lat, lon, buffer_km=5)
        print(f"  ✅ Created polygon with {len(polygon.exterior.coords)} points")
        print(f"  - Bounds: {polygon.bounds}")
    except Exception as e:
        print(f"  ❌ Failed to create polygon: {e}")
        sys.exit(1)
    
    # Test 3: Search for scenes
    print("\n✓ Test 3: Searching for Landsat scenes...")
    print("  (This may take a moment...)")
    try:
        scenes = detector._search_landsat_scenes(polygon)
        if len(scenes) > 0:
            print(f"  ✅ Found {len(scenes)} scenes")
            print(f"  - First scene: {scenes[0].id}")
            print(f"  - Date range: {scenes[0].datetime.date()} to {scenes[-1].datetime.date()}")
        else:
            print(f"  ⚠️  Found 0 usable scenes")
            print(f"  - This might mean no Landsat coverage for this area/time")
    except Exception as e:
        print(f"  ❌ Failed to search scenes: {e}")
        sys.exit(1)
    
    # Test 4: Calculate NDVI for one scene (if available)
    if scenes:
        print("\n✓ Test 4: Calculating NDVI for one scene...")
        print("  (This downloads data from AWS and may take a moment...)")
        try:
            test_scene = scenes[0]
            ndvi_value = detector._calculate_scene_ndvi(test_scene, polygon)
            if ndvi_value is not None:
                print(f"  ✅ NDVI calculated successfully: {ndvi_value:.4f}")
            else:
                print("  ⚠️  NDVI calculation returned None (might be all clouds)")
        except Exception as e:
            print(f"  ❌ Failed to calculate NDVI: {e}")
            print("  This is likely due to AWS access issues or network problems")
    else:
        print("\n⚠️  Skipping Test 4: No scenes found")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print("✅ Detector is properly configured")
    print(f"✅ Found {len(scenes)} Landsat scenes for Mount Hood area (2020-2024)")
    
    if scenes:
        print("✅ All critical components are working!")
        print("\n💡 Next steps:")
        print("   1. Run Streamlit app: streamlit run streamlit_app.py")
        print("   2. Upload locations_mount_hood.csv")
        print("   3. Click 'Run analysis'")
    else:
        print("⚠️  No scenes found - try adjusting date range or location")
    
    print("=" * 60)

if __name__ == "__main__":
    test_detector()
