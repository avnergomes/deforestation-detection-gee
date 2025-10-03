"""Streamlit interface for the deforestation detection workflow."""

from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from deforestation_detector import DeforestationDetector
from deforestation_detector import LandsatScene
from shapely.geometry import Polygon

REQUIRED_COLUMNS = ["location_name", "latitude", "longitude"]


def _load_locations(upload) -> pd.DataFrame:
    """Return a dataframe of locations either from the upload or sample CSV."""
    if upload is not None:
        df = pd.read_csv(upload)
    else:
        df = pd.read_csv("locations.csv")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(missing)}"
        )
    return df[REQUIRED_COLUMNS + [col for col in df.columns if col not in REQUIRED_COLUMNS]]


def _run_analysis(
    detector: DeforestationDetector, locations: pd.DataFrame, buffer_km: float
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, List[LandsatScene]],
    Dict[str, Polygon],
]:
    """Execute the NDVI extraction workflow for each location."""

    ndvi_data: Dict[str, pd.DataFrame] = {}
    scene_data: Dict[str, List[LandsatScene]] = {}
    geometries: Dict[str, Polygon] = {}

    if locations.empty:
        st.warning("No locations available for analysis.")
        return ndvi_data, scene_data, geometries

    progress_text = st.empty()
    progress_bar = st.progress(0)

    total_locations = len(locations)

    for idx, (_, row) in enumerate(locations.iterrows(), start=1):
        location_name = row["location_name"]
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])

        progress_text.markdown(f"**Processing:** {location_name}")
        geometry = detector.create_buffer_polygon(latitude, longitude, buffer_km=buffer_km)
        geometries[location_name] = geometry
        try:
            ndvi_df, scenes = detector.extract_ndvi_time_series_and_scenes(
                geometry, location_name
            )
        except RuntimeError as exc:
            st.warning(
                f"{location_name}: Failed to retrieve Landsat data ({exc})."
            )
            ndvi_df = pd.DataFrame(columns=["date", "ndvi_mean", "location"])
            scenes = []
        ndvi_data[location_name] = ndvi_df
        scene_data[location_name] = scenes
        progress_bar.progress(idx / total_locations)

    progress_text.markdown("✅ Processing complete")
    progress_bar.empty()

    return ndvi_data, scene_data, geometries


def _display_summary(detector: DeforestationDetector, ndvi_data: Dict[str, pd.DataFrame]):
    """Render a tabular summary of analysis results in Streamlit."""
    summary_rows = []

    for name, df in ndvi_data.items():
        analysis = detector.analyze_deforestation(df)
        summary_rows.append(
            {
                "Location": name,
                "Measurements": len(df),
                "Mean NDVI (start)": analysis["mean_ndvi_start"],
                "Mean NDVI (end)": analysis["mean_ndvi_end"],
                "Change (%)": analysis["change_percentage"],
                "Trend": analysis["trend"],
                "Deforestation Detected": "Yes" if analysis["deforestation_detected"] else "No",
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df.style.format(
                {
                    "Mean NDVI (start)": "{:.3f}",
                    "Mean NDVI (end)": "{:.3f}",
                    "Change (%)": "{:+.1f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("No NDVI measurements were extracted for the selected locations.")


st.set_page_config(
    page_title="Deforestation Detection Dashboard",
    page_icon="🌲",
    layout="wide",
)

st.title("🌲 Deforestation Detection Dashboard")
st.write(
    "Analyze vegetation change with Landsat NDVI data. Upload your CSV with `location_name`, "
    "`latitude`, and `longitude`, then launch the analysis."
)

with st.sidebar:
    st.header("Configuration")
    start_year = st.number_input("Start year", min_value=1984, max_value=2024, value=2015)
    end_year = st.number_input("End year", min_value=start_year, max_value=2024, value=2024)
    buffer_km = st.slider("Buffer radius (km)", min_value=1, max_value=20, value=5)
    uploaded_csv = st.file_uploader("Locations CSV", type=["csv"], accept_multiple_files=False)
    run_button = st.button("Run analysis", type="primary")

if run_button:
    try:
        locations_df = _load_locations(uploaded_csv)
    except Exception as exc:  # pragma: no cover - user input dependent
        st.error(f"Unable to read locations CSV: {exc}")
        st.stop()

    try:
        detector = DeforestationDetector(
            start_year=int(start_year),
            end_year=int(end_year),
        )
    except Exception as exc:  # pragma: no cover - depends on runtime auth
        st.error("Failed to prepare the Landsat client. Please try again later.")
        st.exception(exc)
        st.stop()

    ndvi_results, scene_results, geometries = _run_analysis(
        detector, locations_df, buffer_km=float(buffer_km)
    )

    if ndvi_results:
        st.subheader("NDVI Time Series")
        fig = detector.plot_ndvi_time_series(ndvi_results, save_path=None, show=False)
        st.pyplot(fig)

        st.subheader("Landsat Time-lapse")
        gifs_rendered = False
        for idx, location_name in enumerate(locations_df["location_name"]):
            location_label = str(location_name)
            scenes = scene_results.get(location_name, [])
            geometry = geometries.get(location_name)
            if not scenes or geometry is None:
                st.write(
                    f"*{location_label}:* No imagery available for time-lapse generation."
                )
                continue

            try:
                gif_bytes = detector.create_time_lapse_gif(geometry, scenes)
            except RuntimeError as exc:
                st.warning(f"{location_label}: Unable to create GIF ({exc}).")
                continue

            gifs_rendered = True
            st.markdown(f"**{location_label}** ({len(scenes)} scenes)")
            st.image(gif_bytes, caption="Landsat true colour time-lapse", use_column_width=True)
            st.download_button(
                label="Download GIF",
                data=gif_bytes,
                file_name=f"{location_label.replace(' ', '_').lower()}_timelapse.gif",
                mime="image/gif",
                key=f"download-gif-{idx}-{location_label}",
            )

        if not gifs_rendered:
            st.info("No Landsat imagery was available to generate time-lapse animations.")

        st.subheader("Interactive Map")
        map_object = detector.create_interactive_map(locations_df, ndvi_results)
        st_folium(map_object, width=None, height=500)

        st.subheader("Summary")
        _display_summary(detector, ndvi_results)
    else:
        st.info("Analysis completed, but no NDVI data was retrieved for the selected locations.")

else:
    st.info(
        "Configure your analysis in the sidebar and click **Run analysis** to start. "
        "If no file is uploaded, the default `locations.csv` will be used."
    )
