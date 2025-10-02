"""
Setup and Environment Testing Script
Verifies that all required packages are installed and Earth Engine is properly configured
"""

import sys
import subprocess

def check_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("="*70)
    print("DEFORESTATION DETECTION - SETUP & ENVIRONMENT CHECK")
    print("="*70 + "\n")
    
    # Required packages
    required_packages = {
        'ee': 'earthengine-api',
        'geemap': 'geemap',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'folium': 'folium'
    }
    
    print("📦 Checking required packages...\n")
    
    missing_packages = []
    installed_packages = []
    
    for import_name, package_name in required_packages.items():
        if check_package(import_name):
            print(f"  ✓ {package_name}")
            installed_packages.append(package_name)
        else:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            missing_packages.append(package_name)
    
    print()
    
    # Install missing packages
    if missing_packages:
        print(f"⚠  Found {len(missing_packages)} missing package(s)\n")
        response = input("Would you like to install missing packages now? (yes/no): ").lower()
        
        if response in ['yes', 'y']:
            print("\n📥 Installing missing packages...\n")
            for package in missing_packages:
                print(f"  Installing {package}...", end=" ")
                if install_package(package):
                    print("✓")
                else:
                    print("✗ FAILED")
            print("\n✅ Installation complete!")
        else:
            print("\nℹ  To install manually, run:")
            print(f"   pip install {' '.join(missing_packages)}")
    else:
        print("✅ All required packages are installed!\n")
    
    # Test Earth Engine authentication
    print("="*70)
    print("GOOGLE EARTH ENGINE AUTHENTICATION")
    print("="*70 + "\n")
    
    try:
        import ee
        
        try:
            ee.Initialize()
            print("✅ Google Earth Engine is already authenticated and initialized!")
            print("\nTesting Earth Engine access...")
            
            # Simple test query
            image = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_044034_20140318')
            info = image.getInfo()
            
            print("✓ Successfully connected to Google Earth Engine")
            print(f"✓ Test query successful (retrieved Landsat image metadata)")
            
        except Exception as e:
            print("⚠  Earth Engine is not authenticated yet\n")
            print("To authenticate, please run the following commands:\n")
            print("  import ee")
            print("  ee.Authenticate()\n")
            print("This will open a browser window for you to sign in with your Google account.")
            print("After authentication, run this script again to verify.\n")
            
    except ImportError:
        print("✗ Could not import Earth Engine. Please install it first:")
        print("  pip install earthengine-api\n")
    
    # Check for locations.csv
    print("="*70)
    print("INPUT FILES CHECK")
    print("="*70 + "\n")
    
    import os
    
    if os.path.exists('locations.csv'):
        print("✓ locations.csv found")
        
        # Verify CSV format
        try:
            import pandas as pd
            df = pd.read_csv('locations.csv')
            required_cols = ['location_name', 'latitude', 'longitude']
            
            if all(col in df.columns for col in required_cols):
                print(f"✓ CSV format is correct ({len(df)} location(s) found)")
                print("\nLocations to be analyzed:")
                for idx, row in df.iterrows():
                    print(f"  {idx+1}. {row['location_name']}")
            else:
                print("⚠  CSV is missing required columns")
                print(f"   Required: {required_cols}")
                print(f"   Found: {list(df.columns)}")
        except Exception as e:
            print(f"⚠  Error reading CSV: {e}")
    else:
        print("⚠  locations.csv not found")
        print("\nCreating sample locations.csv...")
        
        sample_csv = """location_name,latitude,longitude,description
Willamette National Forest - Logging Area,44.2145,-122.1567,Area with documented logging activity in Oregon
Crater Lake National Park - Protected Area,42.9446,-122.1090,Protected forest area in Oregon with minimal change"""
        
        with open('locations.csv', 'w') as f:
            f.write(sample_csv)
        
        print("✓ Sample locations.csv created with two Oregon locations")
    
    # Final summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70 + "\n")
    
    all_ready = len(missing_packages) == 0
    
    if all_ready:
        try:
            import ee
            ee.Initialize()
            print("✅ Your environment is ready!")
            print("\nYou can now run the main script:")
            print("   python deforestation_detector.py\n")
        except:
            print("⚠  Almost ready! Just need to authenticate Earth Engine")
            print("\nRun these commands:")
            print("   import ee")
            print("   ee.Authenticate()")
            print("\nThen run: python deforestation_detector.py\n")
    else:
        print("⚠  Setup incomplete. Please install missing packages first.\n")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
