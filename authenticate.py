"""
Quick Authentication Script for Google Earth Engine
Run this script once to authenticate your Google Earth Engine account
"""

import ee

print("="*70)
print("GOOGLE EARTH ENGINE AUTHENTICATION")
print("="*70 + "\n")

print("This script will open a browser window for you to authenticate.")
print("Please follow these steps:\n")
print("1. Sign in with your Google account")
print("2. Authorize Earth Engine access")
print("3. Copy the authorization code")
print("4. Paste it back here\n")

print("Starting authentication process...\n")

try:
    ee.Authenticate()
    print("\n" + "="*70)
    print("✅ AUTHENTICATION SUCCESSFUL!")
    print("="*70 + "\n")
    
    print("Testing connection...")
    ee.Initialize()
    print("✓ Earth Engine initialized successfully\n")
    
    # Quick test
    image = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_044034_20140318')
    info = image.getInfo()
    print("✓ Test query successful\n")
    
    print("="*70)
    print("You're all set! You can now run:")
    print("  python deforestation_detector.py")
    print("="*70 + "\n")
    
except Exception as e:
    print("\n" + "="*70)
    print("⚠ AUTHENTICATION FAILED")
    print("="*70 + "\n")
    print(f"Error: {e}\n")
    print("Please try again or check your internet connection.\n")
