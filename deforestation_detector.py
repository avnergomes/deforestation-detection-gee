"""Deforestation detection workflow built on open Landsat imagery.

This module replaces the original Google Earth Engine based approach with a
purely open-data solution so the project can run on Streamlit Community Cloud
without requiring Earth Engine credentials. Landsat Collection 2 Level 2 scenes
are retrieved from the public STAC API operated by Element84
(`https://earth-search.aws.element84.com`). Each scene is clipped to the area of
interest, converted to surface reflectance, and NDVI statistics are derived to
detect vegetation loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.mask import mask
from shapely.geometry import Polygon, mapping
from pystac_client import Client
import folium


STAC_API_URL = "https://earth-search.aws.element84.com/v1"
LANDSAT_COLLECTION = "landsat-c2-l2"
NIR_BAND = "SR_B5"
RED_BAND = "SR_B4"
L2_SCALE = 0.0000275
L2_OFFSET = -0.2


@dataclass
class LandsatScene:
    """Simple container for Landsat scene metadata used in the workflow."""

    id: str
    datetime: datetime
    nir_href: str
    red_href: str


class DeforestationDetector:
    """
    A class to detect and analyze deforestation using Google Earth Engine
    and NDVI time series analysis.
    """
    
    def __init__(self, start_year=2015, end_year=2024, credentials=None):
        """
        Initialize the DeforestationDetector.
        
        Parameters:
        -----------
        start_year : int
            Starting year for analysis
        end_year : int
            Ending year for analysis
        """
        self.start_year = start_year
        self.end_year = end_year
        self.start_date = f"{start_year}-01-01"
        self.end_date = f"{end_year}-12-31"
        self.max_cloud_cover = max_cloud_cover
        self.geod = pyproj.Geod(ellps="WGS84")
        self.client = stac_client or Client.open(stac_url)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def create_buffer_polygon(self, latitude: float, longitude: float, buffer_km: float = 5) -> Polygon:
        """Create a geodesic buffer polygon around a lat/lon point."""

        azimuths = np.linspace(0, 360, num=73)
        lon_arr = np.full_like(azimuths, longitude, dtype=float)
        lat_arr = np.full_like(azimuths, latitude, dtype=float)
        distances = np.full_like(azimuths, buffer_km * 1000.0, dtype=float)
        lon_points, lat_points, _ = self.geod.fwd(lon_arr, lat_arr, azimuths, distances)
        ring = np.column_stack((lon_points, lat_points))
        return Polygon(ring)

    # ------------------------------------------------------------------
    # Landsat retrieval and processing
    # ------------------------------------------------------------------
    def _search_landsat_scenes(self, polygon: Polygon) -> List[LandsatScene]:
        """Query the STAC API for Landsat scenes covering the polygon."""

        search = self.client.search(
            collections=[LANDSAT_COLLECTION],
            datetime=f"{self.start_date}/{self.end_date}",
            query={"eo:cloud_cover": {"lt": self.max_cloud_cover}},
            intersects=mapping(polygon),
        )

        items = list(search.get_items())
        scenes: List[LandsatScene] = []

        for item in items:
            assets = item.assets
            if NIR_BAND not in assets or RED_BAND not in assets:
                continue

            scenes.append(
                LandsatScene(
                    id=item.id,
                    datetime=datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00")),
                    nir_href=assets[NIR_BAND].href,
                    red_href=assets[RED_BAND].href,
                )
            )

        scenes.sort(key=lambda scene: scene.datetime)
        return scenes

    def _read_band_array(self, href: str, polygon: Polygon) -> np.ndarray:
        """Read a band as a masked array clipped to the polygon."""

        try:
            if credentials is not None:
                ee.Initialize(credentials)
            else:
                ee.Initialize()
            print("✓ Google Earth Engine initialized successfully")
        except Exception as e:
            print("✗ Error initializing Earth Engine. Please authenticate:")
            print("  Run: ee.Authenticate()")
            raise e
    
    def load_locations_from_csv(self, csv_path):
        """
        Load location coordinates from a CSV file.
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file with columns: location_name, latitude, longitude
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with location information
        """
        try:
            nir = self._read_band_array(scene.nir_href, polygon)
            red = self._read_band_array(scene.red_href, polygon)
        except RuntimeError:
            return None

        nir_reflectance = nir * L2_SCALE + L2_OFFSET
        red_reflectance = red * L2_SCALE + L2_OFFSET

        denominator = nir_reflectance + red_reflectance
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir_reflectance - red_reflectance) / denominator

        ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)
        if np.isnan(ndvi).all():
            return None

        return float(np.nanmean(ndvi))

    def extract_ndvi_time_series(self, geometry: Polygon, location_name: str) -> pd.DataFrame:
        """Extract NDVI statistics for all Landsat scenes over the polygon."""

        print(f"\n📊 Processing {location_name}...")

        scenes = self._search_landsat_scenes(geometry)
        print(f"  Found {len(scenes)} candidate scenes")

        records: List[Dict[str, object]] = []
        for scene in scenes:
            ndvi_mean = self._calculate_scene_ndvi(scene, geometry)
            if ndvi_mean is None:
                continue
            records.append(
                {
                    "date": scene.datetime,
                    "ndvi_mean": ndvi_mean,
                    "location": location_name,
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            print(f"  ✓ Extracted {len(df)} NDVI measurements")
        else:
            print(f"  ⚠ No valid NDVI observations for {location_name}")

        return df

    # ------------------------------------------------------------------
    # Analysis and visualisation utilities
    # ------------------------------------------------------------------
    def analyze_deforestation(self, df: pd.DataFrame) -> Dict[str, object]:
        """Analyze NDVI trends to detect deforestation."""

        if df.empty:
            return {
                "trend": None,
                "change_percentage": 0,
                "mean_ndvi_start": 0,
                "mean_ndvi_end": 0,
                "deforestation_detected": False,
            }

        df = df.copy()
        df["year"] = df["date"].dt.year
        annual_means = df.groupby("year")["ndvi_mean"].mean()

        first_period = df[df["year"] <= df["year"].min() + 2]["ndvi_mean"].mean()
        last_period = df[df["year"] >= df["year"].max() - 2]["ndvi_mean"].mean()

        change_percentage = ((last_period - first_period) / first_period) * 100 if first_period != 0 else 0
        deforestation_detected = change_percentage < -15

        return {
            "trend": "decreasing" if change_percentage < 0 else "increasing",
            "change_percentage": change_percentage,
            "mean_ndvi_start": first_period,
            "mean_ndvi_end": last_period,
            "deforestation_detected": deforestation_detected,
            "annual_means": annual_means,
        }
        
        return results
    
    def plot_ndvi_time_series(self, df_dict, save_path=None, show=True):
        """
        Create visualization of NDVI time series for multiple locations.
        
        Parameters:
        -----------
        df_dict : dict
            Dictionary with location names as keys and DataFrames as values
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(len(df_dict), 1, figsize=(14, 6*len(df_dict)))
        
        if len(df_dict) == 1:
            axes = [axes]

        for idx, (location_name, df) in enumerate(df_dict.items()):
            ax = axes[idx]

            if df.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data available for {location_name}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(f"{location_name} - No Data")
                continue

            df = df.copy()
            ax.scatter(
                df["date"],
                df["ndvi_mean"],
                alpha=0.4,
                s=30,
                label="NDVI measurements",
                color="#2ecc71",
            )

            df["ndvi_smooth"] = df["ndvi_mean"].rolling(window=5, center=True).mean()
            ax.plot(
                df["date"],
                df["ndvi_smooth"],
                linewidth=2.5,
                label="Smoothed trend",
                color="#27ae60",
            )

            df["year"] = df["date"].dt.year
            annual_means = df.groupby("year").agg({"ndvi_mean": "mean", "date": "mean"})
            ax.plot(
                annual_means["date"],
                annual_means["ndvi_mean"],
                "o-",
                linewidth=2,
                markersize=8,
                label="Annual average",
                color="#e74c3c",
            )

            analysis = self.analyze_deforestation(df)
            status = "⚠ DEFORESTATION DETECTED" if analysis["deforestation_detected"] else "✓ Stable/Improving"

            ax.set_xlabel("Date", fontsize=12, fontweight="bold")
            ax.set_ylabel("NDVI", fontsize=12, fontweight="bold")
            ax.set_title(
                f"{location_name} - NDVI Time Series\n"
                f"Change: {analysis['change_percentage']:.1f}% | Status: {status}",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(loc="best", fontsize=10)
            ax.set_ylim([0, 1])

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Plot saved to: {save_path}")

        if show:
            plt.show()

        return fig
    
    def create_interactive_map(self, locations_df, ndvi_data_dict):
        """
        Create an interactive map showing the locations and their status.
        
        Parameters:
        -----------
        locations_df : pandas.DataFrame
            DataFrame with location information
        ndvi_data_dict : dict
            Dictionary with NDVI time series data
            
        Returns:
        --------
        folium.Map
            Interactive map object
        """
        # Calculate center point
        center_lat = locations_df['latitude'].mean()
        center_lon = locations_df['longitude'].mean()
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )
        
        # Add markers for each location
        for idx, row in locations_df.iterrows():
            location_name = row['location_name']
            
            if location_name in ndvi_data_dict and not ndvi_data_dict[location_name].empty:
                analysis = self.analyze_deforestation(ndvi_data_dict[location_name])
                color = "red" if analysis["deforestation_detected"] else "green"
                icon = "exclamation-triangle" if analysis["deforestation_detected"] else "leaf"
                popup_html = f"""
                <div style=\"font-family: Arial; min-width: 200px;\">
                    <h4>{location_name}</h4>
                    <p><b>Status:</b> {'⚠ Deforestation Detected' if analysis['deforestation_detected'] else '✓ Stable'}</p>
                    <p><b>NDVI Change:</b> {analysis['change_percentage']:.1f}%</p>
                    <p><b>Start NDVI:</b> {analysis['mean_ndvi_start']:.3f}</p>
                    <p><b>End NDVI:</b> {analysis['mean_ndvi_end']:.3f}</p>
                </div>
                """
            else:
                color = "gray"
                icon = "question"
                popup_html = f"""
                <div style=\"font-family: Arial; min-width: 200px;\">
                    <h4>{location_name}</h4>
                    <p>No NDVI observations available.</p>
                </div>
                """

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=color, icon=icon, prefix="fa"),
            ).add_to(m)

        return m

