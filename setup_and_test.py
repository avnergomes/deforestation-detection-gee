"""Setup and environment testing script for the Landsat workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict

STAC_API_URL = "https://earth-search.aws.element84.com/v1"


REQUIRED_PACKAGES: Dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "rasterio": "rasterio",
    "pyproj": "pyproj",
    "shapely": "shapely",
    "pystac_client": "pystac-client",
    "folium": "folium",
    "streamlit": "streamlit",
    "streamlit_folium": "streamlit-folium",
}


def check_package(package_name: str) -> bool:
    """Return True if a package can be imported."""

    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name: str) -> bool:
    """Attempt to install a package via pip."""

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except subprocess.CalledProcessError:
        return False
    return True


def verify_stac_connection() -> bool:
    """Ensure the Landsat STAC API is reachable."""

    try:
        from pystac_client import Client

        client = Client.open(STAC_API_URL)
        _ = client.get_collections()
    except Exception as exc:  # pragma: no cover - depends on network
        print("✗ Unable to reach the Landsat STAC API")
        print(f"  {exc}\n")
        return False

    print("✓ Landsat STAC API connection successful\n")
    return True


def check_locations_file() -> None:
    """Confirm that locations.csv exists and has the right columns."""

    if os.path.exists("locations.csv"):
        print("✓ locations.csv found")
        try:
            import pandas as pd

            df = pd.read_csv("locations.csv")
            required_cols = {"location_name", "latitude", "longitude"}
            if required_cols.issubset(df.columns):
                print(f"✓ CSV format is correct ({len(df)} location(s) found)")
                for idx, row in df.iterrows():
                    print(f"  {idx + 1}. {row['location_name']}")
            else:
                print("⚠ locations.csv is missing required columns")
                print(f"  Required: {sorted(required_cols)}")
                print(f"  Found: {list(df.columns)}")
        except Exception as exc:  # pragma: no cover - user environment
            print(f"⚠ Error reading CSV: {exc}")
    else:
        print("⚠ locations.csv not found")
        sample_csv = (
            "location_name,latitude,longitude,description\n"
            "Willamette National Forest - Logging Area,44.2145,-122.1567,"
            "Area with documented logging activity in Oregon\n"
            "Crater Lake National Park - Protected Area,42.9446,-122.1090,"
            "Protected forest area in Oregon with minimal change"
        )
        with open("locations.csv", "w", encoding="utf-8") as file:
            file.write(sample_csv)
        print("✓ Sample locations.csv created with two Oregon locations")


def main() -> None:
    print("=" * 70)
    print("DEFORESTATION DETECTION - SETUP & ENVIRONMENT CHECK")
    print("=" * 70 + "\n")

    print("📦 Checking required packages...\n")
    missing_packages = []
    for import_name, package_name in REQUIRED_PACKAGES.items():
        if check_package(import_name):
            print(f"  ✓ {package_name}")
        else:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            missing_packages.append(package_name)

    print()

    if missing_packages:
        print(f"⚠ Found {len(missing_packages)} missing package(s)\n")
        response = input("Install missing packages now? (yes/no): ").strip().lower()
        if response in {"yes", "y"}:
            print("\n📥 Installing missing packages...\n")
            for package in missing_packages:
                print(f"  Installing {package}...", end=" ")
                if install_package(package):
                    print("✓")
                else:
                    print("✗ FAILED")
            print("\n✅ Installation attempt complete!\n")
        else:
            print("\nℹ To install manually, run:")
            print(f"   pip install {' '.join(missing_packages)}\n")
    else:
        print("✅ All required packages are installed!\n")

    print("=" * 70)
    print("LANDSAT DATA ACCESS")
    print("=" * 70 + "\n")
    stac_ok = verify_stac_connection()

    print("=" * 70)
    print("INPUT FILES CHECK")
    print("=" * 70 + "\n")
    check_locations_file()

    print("\n" + "=" * 70)
    print("SETUP SUMMARY")
    print("=" * 70 + "\n")

    if not missing_packages and stac_ok:
        print("✅ Your environment is ready! Run:")
        print("   python simple_example.py\n")
    else:
        print("⚠ Setup incomplete. See notes above and retry once resolved.\n")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
