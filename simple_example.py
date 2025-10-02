"""
Simple example showing how to run the Landsat-based detector without Streamlit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from deforestation_detector import DeforestationDetector


def main() -> None:
    detector = DeforestationDetector(start_year=2018, end_year=2024)

    locations = [
        {"name": "Logging Area", "lat": 44.2145, "lon": -122.1567},
        {"name": "Protected Forest", "lat": 42.9446, "lon": -122.1090},
    ]

    ndvi_results = {}
    for loc in locations:
        geom = detector.create_buffer_polygon(loc["lat"], loc["lon"], buffer_km=5)
        ndvi_results[loc["name"]] = detector.extract_ndvi_time_series(geom, loc["name"])

    fig = detector.plot_ndvi_time_series(ndvi_results, show=False)
    output_path = Path("simple_results.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✅ Results saved to: {output_path.resolve()}")

    for name, df in ndvi_results.items():
        analysis = detector.analyze_deforestation(df)
        print(
            f"{name}: change {analysis['change_percentage']:.1f}% | "
            f"status: {'deforestation' if analysis['deforestation_detected'] else 'stable'}"
        )


if __name__ == "__main__":
    main()
