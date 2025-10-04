"""Utilities for downloading and analysing Landsat NDVI time-series data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from pystac_client import Client
from pystac_client.exceptions import APIError
from rasterio.errors import RasterioIOError
from rasterio.mask import mask
from rasterio.env import Env
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shapely_transform

try:  # Pillow 9.1+ exposes resampling filters via Image.Resampling
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError:  # pragma: no cover - optional dependency at runtime
    Image = ImageDraw = ImageFont = ImageOps = None  # type: ignore
    _RESAMPLING_BILINEAR = None
else:  # pragma: no cover - import-time branch
    _RESAMPLING_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

STAC_API_URL = "https://earth-search.aws.element84.com/v1"
LANDSAT_COLLECTION = "landsat-c2-l2"

# Element84 STAC uses descriptive band names for Landsat Collection 2
NIR_BAND = "nir08"
RED_BAND = "red"
GREEN_BAND = "green"
BLUE_BAND = "blue"

L2_SCALE = 0.0000275
L2_OFFSET = -0.2


@dataclass
class LandsatScene:
    """Container for the assets required to compute NDVI for a scene."""

    id: str
    datetime: datetime
    nir_href: str
    red_href: str
    green_href: str
    blue_href: str


class DeforestationDetector:
    """Download Landsat L2 scenes and derive NDVI statistics for polygons."""

    def __init__(
        self,
        start_year: int = 2015,
        end_year: int = 2024,
        *,
        max_cloud_cover: int = 70,
        stac_url: str = STAC_API_URL,
        stac_client: Optional[Client] = None,
    ) -> None:
        if end_year < start_year:
            raise ValueError("end_year must be greater or equal to start_year")

        self.start_year = start_year
        self.end_year = end_year
        self.start_date = f"{start_year}-01-01"
        self.end_date = f"{end_year}-12-31"
        self.max_cloud_cover = max_cloud_cover
        self.geod = pyproj.Geod(ellps="WGS84")
        self.client = stac_client or Client.open(stac_url)
        
        # Set AWS request payer environment variable for Landsat access
        import os
        os.environ['AWS_REQUEST_PAYER'] = 'requester'

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def create_buffer_polygon(
        self, latitude: float, longitude: float, buffer_km: float = 5
    ) -> Polygon:
        """Return a polygon approximating a geodesic buffer around a point."""

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
    def _select_asset_href(self, asset) -> str:
        """Return an HTTP-accessible href for a STAC asset."""

        href = getattr(asset, "href", "") or ""

        alternates = getattr(asset, "extra_fields", {}).get("alternate", [])
        if isinstance(alternates, dict):
            alternates = [alternates]
        for alternate in alternates:
            alt_href = alternate.get("href")
            if isinstance(alt_href, str) and alt_href.startswith("http"):
                return alt_href

        s3_prefix_map = {
            "s3://usgs-landsat/": "https://landsatlook.usgs.gov/data/",
            "s3://landsat-c2/": "https://landsatlook.usgs.gov/data/",
            "s3://landsat-pds/": "https://landsat-pds.s3.amazonaws.com/",
        }
        for prefix, replacement in s3_prefix_map.items():
            if href.startswith(prefix):
                return href.replace(prefix, replacement)

        return href

    def _search_landsat_scenes(self, polygon: Polygon) -> List[LandsatScene]:
        """Query the STAC API for Landsat scenes covering ``polygon``."""

        try:
            search = self.client.search(
                collections=[LANDSAT_COLLECTION],
                datetime=f"{self.start_date}/{self.end_date}",
                query={"eo:cloud_cover": {"lt": self.max_cloud_cover}},
                intersects=mapping(polygon),
            )
        except APIError as exc:
            raise RuntimeError(
                "Unable to query the Landsat STAC endpoint."
            ) from exc

        items = list(search.get_items())
        print(f"Found {len(items)} Landsat scenes matching the criteria")
        
        scenes: List[LandsatScene] = []
        required_bands = [NIR_BAND, RED_BAND, GREEN_BAND, BLUE_BAND]

        for item in items:
            assets = item.assets
            
            # Check if all required bands are present
            if not all(band in assets for band in required_bands):
                continue

            nir_asset = assets[NIR_BAND]
            red_asset = assets[RED_BAND]
            green_asset = assets[GREEN_BAND]
            blue_asset = assets[BLUE_BAND]

            scenes.append(
                LandsatScene(
                    id=item.id,
                    datetime=datetime.fromisoformat(
                        item.properties["datetime"].replace("Z", "+00:00")
                    ),
                    nir_href=self._select_asset_href(nir_asset),
                    red_href=self._select_asset_href(red_asset),
                    green_href=self._select_asset_href(green_asset),
                    blue_href=self._select_asset_href(blue_asset),
                )
            )

        scenes.sort(key=lambda scene: scene.datetime)
        print(f"Prepared {len(scenes)} scenes for NDVI calculation")
        return scenes

    def _read_band_array(self, href: str, polygon: Polygon) -> np.ndarray:
        """Read a single Landsat band clipped to ``polygon`` as a float array."""

        # Set up rasterio environment with AWS request payer
        env_options = {
            'AWS_REQUEST_PAYER': 'requester',
            'GDAL_HTTP_MULTIRANGE': 'YES',
            'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES',
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
        }

        try:
            with Env(**env_options):
                with rasterio.open(href) as src:
                    nodata = src.nodata
                    if src.crs is None:
                        raise RuntimeError(f"Raster {href} lacks CRS information")
                    transformer = pyproj.Transformer.from_crs(
                        "EPSG:4326", src.crs, always_xy=True
                    )
                    projected_polygon = shapely_transform(transformer.transform, polygon)
                    data, _ = mask(src, [mapping(projected_polygon)], crop=True)
        except RasterioIOError as exc:  # pragma: no cover - network/local IO
            raise RuntimeError(f"Unable to read raster {href}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Unexpected error reading {href}: {exc}") from exc

        band = data[0].astype("float32")
        if nodata is not None:
            band[band == nodata] = np.nan
        band = np.where(band == -9999, np.nan, band)
        band = np.where(band == 0, np.nan, band)
        return band

    def _calculate_scene_ndvi(self, scene: LandsatScene, polygon: Polygon) -> Optional[float]:
        """Return the mean NDVI for ``scene`` over ``polygon``."""

        try:
            nir = self._read_band_array(scene.nir_href, polygon)
            red = self._read_band_array(scene.red_href, polygon)
        except RuntimeError as e:
            print(f"Warning: Failed to read bands for scene {scene.id}: {e}")
            return None

        if nir.shape != red.shape:
            min_rows = min(nir.shape[0], red.shape[0])
            min_cols = min(nir.shape[1], red.shape[1])
            nir = nir[:min_rows, :min_cols]
            red = red[:min_rows, :min_cols]

        nir_reflectance = nir * L2_SCALE + L2_OFFSET
        red_reflectance = red * L2_SCALE + L2_OFFSET

        denominator = nir_reflectance + red_reflectance
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir_reflectance - red_reflectance) / denominator

        ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)
        
        # Count valid pixels
        valid_pixels = np.sum(~np.isnan(ndvi))
        total_pixels = ndvi.size
        
        if valid_pixels == 0:
            print(f"Warning: No valid pixels for scene {scene.id}")
            return None
        
        if valid_pixels < total_pixels * 0.1:  # Less than 10% valid data
            print(f"Warning: Scene {scene.id} has only {valid_pixels}/{total_pixels} valid pixels ({100*valid_pixels/total_pixels:.1f}%)")

        return float(np.nanmean(ndvi))

    def extract_ndvi_time_series(
        self, geometry: Polygon, location_name: str
    ) -> pd.DataFrame:
        """Extract NDVI measurements for every Landsat scene that intersects."""

        df, _ = self.extract_ndvi_time_series_and_scenes(geometry, location_name)
        return df

    def extract_ndvi_time_series_and_scenes(
        self, geometry: Polygon, location_name: str
    ) -> Tuple[pd.DataFrame, List[LandsatScene]]:
        """Return the NDVI dataframe and associated Landsat scenes."""

        print(f"\n=== Processing {location_name} ===")
        scenes = self._search_landsat_scenes(geometry)
        
        if not scenes:
            print(f"No scenes found for {location_name}")
            return pd.DataFrame(columns=["date", "ndvi_mean", "location"]), []

        records: List[Dict[str, object]] = []
        successful_scenes = 0
        
        for idx, scene in enumerate(scenes):
            print(f"Processing scene {idx+1}/{len(scenes)}: {scene.id} ({scene.datetime.strftime('%Y-%m-%d')})")
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
            successful_scenes += 1

        print(f"Successfully calculated NDVI for {successful_scenes}/{len(scenes)} scenes")
        
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
        else:
            print(f"WARNING: No valid NDVI data extracted for {location_name}")
            
        return df, scenes

    # ------------------------------------------------------------------
    # Analysis and visualisation utilities
    # ------------------------------------------------------------------
    def analyze_deforestation(self, df: pd.DataFrame) -> Dict[str, object]:
        """Analyse NDVI trends to detect possible deforestation."""

        if df.empty:
            return {
                "trend": None,
                "change_percentage": 0.0,
                "mean_ndvi_start": 0.0,
                "mean_ndvi_end": 0.0,
                "deforestation_detected": False,
                "annual_means": pd.Series(dtype=float),
            }

        df = df.copy()
        df["year"] = df["date"].dt.year
        annual_means = df.groupby("year")["ndvi_mean"].mean()

        first_period = (
            df[df["year"] <= df["year"].min() + 2]["ndvi_mean"].mean()
        )
        last_period = (
            df[df["year"] >= df["year"].max() - 2]["ndvi_mean"].mean()
        )

        change_percentage = (
            ((last_period - first_period) / first_period) * 100 if first_period else 0.0
        )
        deforestation_detected = change_percentage < -15

        return {
            "trend": "decreasing" if change_percentage < 0 else "increasing",
            "change_percentage": change_percentage,
            "mean_ndvi_start": first_period,
            "mean_ndvi_end": last_period,
            "deforestation_detected": deforestation_detected,
            "annual_means": annual_means,
        }

    def plot_ndvi_time_series(
        self,
        df_dict: Dict[str, pd.DataFrame],
        *,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Create a multi-panel NDVI plot from the extracted time-series."""

        if not df_dict:
            raise ValueError("df_dict must contain at least one location")

        fig, axes = plt.subplots(len(df_dict), 1, figsize=(14, 6 * len(df_dict)))
        if len(df_dict) == 1:
            axes = [axes]

        for ax, (location_name, df) in zip(axes, df_dict.items()):
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
                ax.set_axis_off()
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
            status = (
                "⚠ DEFORESTATION DETECTED"
                if analysis["deforestation_detected"]
                else "✓ Stable/Improving"
            )

            ax.set_xlabel("Date", fontsize=12, fontweight="bold")
            ax.set_ylabel("NDVI", fontsize=12, fontweight="bold")
            ax.set_title(
                (
                    f"{location_name} - NDVI Time Series\n"
                    f"Change: {analysis['change_percentage']:.1f}% | Status: {status}"
                ),
                fontsize=13,
                fontweight="bold",
                pad=15,
            )

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(loc="best", fontsize=10)
            ax.set_ylim([0, 1])

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def _prepare_true_color_frame(
        self,
        red: np.ndarray,
        green: np.ndarray,
        blue: np.ndarray,
        *,
        frame_size: int,
    ) -> Optional[Image.Image]:
        """Create a display-ready RGB image from reflectance bands."""

        if red.shape != green.shape or red.shape != blue.shape:
            min_rows = min(red.shape[0], green.shape[0], blue.shape[0])
            min_cols = min(red.shape[1], green.shape[1], blue.shape[1])
            red = red[:min_rows, :min_cols]
            green = green[:min_rows, :min_cols]
            blue = blue[:min_rows, :min_cols]

        stack = np.stack([red, green, blue], axis=-1).astype("float32")
        stack = stack * L2_SCALE + L2_OFFSET
        if not np.isfinite(stack).any():
            return None

        finite_values = stack[np.isfinite(stack)]
        if finite_values.size == 0:
            return None

        lower, upper = np.nanpercentile(finite_values, (2, 98))
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower, upper = 0.0, 0.3

        scaled = (stack - lower) / (upper - lower if upper > lower else 1.0)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled = np.nan_to_num(scaled, nan=0.0)

        rgb_uint8 = (scaled * 255).astype(np.uint8)
        image = Image.fromarray(rgb_uint8, mode="RGB")
        image = ImageOps.fit(image, (frame_size, frame_size), method=_RESAMPLING_BILINEAR)
        return image

    def _annotate_frame(self, image: Image.Image, timestamp: datetime) -> None:
        """Overlay the acquisition year on the GIF frame."""

        draw = ImageDraw.Draw(image)
        text = timestamp.strftime("%Y")
        font = ImageFont.load_default()
        
        # Use textbbox instead of deprecated textsize
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        padding = 10
        rect = (
            padding,
            padding,
            padding + text_width + 12,
            padding + text_height + 12,
        )
        draw.rectangle(rect, fill=(0, 0, 0))
        draw.text(
            (rect[0] + 6, rect[1] + 6),
            text,
            font=font,
            fill=(255, 255, 255),
        )

    def create_time_lapse_gif(
        self,
        polygon: Polygon,
        scenes: List[LandsatScene],
        *,
        frame_size: int = 512,
        frame_duration_ms: int = 800,
    ) -> bytes:
        """Generate a true-colour Landsat GIF for the supplied scenes."""

        if Image is None or ImageOps is None or _RESAMPLING_BILINEAR is None:
            raise ImportError("Pillow is required to generate Landsat time-lapse GIFs.")

        if not scenes:
            raise ValueError("scenes must contain at least one LandsatScene")

        frames: List[Image.Image] = []

        for scene in scenes:
            try:
                red = self._read_band_array(scene.red_href, polygon)
                green = self._read_band_array(scene.green_href, polygon)
                blue = self._read_band_array(scene.blue_href, polygon)
            except RuntimeError:
                continue

            frame = self._prepare_true_color_frame(
                red,
                green,
                blue,
                frame_size=frame_size,
            )
            if frame is None:
                continue

            self._annotate_frame(frame, scene.datetime)
            frames.append(frame)

        if not frames:
            raise RuntimeError("Unable to generate GIF frames from the available scenes")

        buffer = BytesIO()
        frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )
        buffer.seek(0)
        return buffer.getvalue()

    def create_interactive_map(
        self, locations_df: pd.DataFrame, ndvi_data_dict: Dict[str, pd.DataFrame]
    ) -> folium.Map:
        """Render an interactive map with NDVI status indicators."""

        if locations_df.empty:
            raise ValueError("locations_df must contain at least one location")

        center_lat = locations_df["latitude"].mean()
        center_lon = locations_df["longitude"].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="OpenStreetMap")

        for _, row in locations_df.iterrows():
            location_name = row["location_name"]
            analysis = None
            if location_name in ndvi_data_dict and not ndvi_data_dict[location_name].empty:
                analysis = self.analyze_deforestation(ndvi_data_dict[location_name])

            if analysis:
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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def load_locations_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load a CSV file containing ``location_name``, ``latitude`` and ``longitude``."""

        df = pd.read_csv(csv_path)
        required = {"location_name", "latitude", "longitude"}
        missing = required.difference(df.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required column(s): {missing_cols}")
        return df
