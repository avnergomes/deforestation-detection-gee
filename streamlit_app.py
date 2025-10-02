"""Streamlit interface for the deforestation detection workflow."""

from typing import Dict
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from deforestation_detector import DeforestationDetector

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
def _run_analysis(detector: DeforestationDetector, locations: pd.DataFrame, buffer_km: float) -> Dict[str, pd.DataFrame]:
    """Execute the NDVI extraction workflow for each location."""
    ndvi_data: Dict[str, pd.DataFrame] = {}

    if locations.empty:
        st.warning("No locations available for analysis.")
        return ndvi_data

    progress_text = st.empty()
    progress_bar = st.progress(0)

    for idx, row in locations.iterrows():
        location_name = row["location_name"]
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])

        progress_text.markdown(f"**Processing:** {location_name}")
        geometry = detector.create_buffer_polygon(latitude, longitude, buffer_km=buffer_km)
        ndvi_df = detector.extract_ndvi_time_series(geometry, location_name)
        ndvi_data[location_name] = ndvi_df
        progress_bar.progress((idx + 1) / len(locations))

    progress_text.markdown("✅ Processing complete")
    progress_bar.empty()

    return ndvi_data


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
    "Analyze vegetation change with Landsat NDVI data sourced from the public "
    "Element84 STAC API. Upload your CSV with `location_name`, `latitude`, and "
    "`longitude`, then launch the analysis."
)

with st.sidebar:
    st.header("Configuration")
    start_year = st.number_input("Start year", min_value=1984, max_value=2024, value=2015)
    end_year = st.number_input("End year", min_value=start_year, max_value=2024, value=2024)
    buffer_km = st.slider("Buffer radius (km)", min_value=1, max_value=20, value=5)
    max_cloud_cover = st.slider(
        "Max cloud cover (%)",
        min_value=0,
        max_value=100,
        value=20,
        help="Scenes with higher reported cloud cover are ignored to improve NDVI quality.",
    )
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
            max_cloud_cover=int(max_cloud_cover),
        )
    except Exception as exc:  # pragma: no cover - depends on remote service
        st.error(
            "Unable to initialise the Landsat STAC client. "
            "Check your network connection and try again."
        )
        st.exception(exc)
        st.stop()

    try:
        with st.spinner("Downloading Landsat scenes and computing NDVI..."):
            ndvi_results = _run_analysis(detector, locations_df, buffer_km=float(buffer_km))
    except Exception as exc:  # pragma: no cover - depends on remote service
        st.error("NDVI analysis failed. Please retry with a smaller area or later.")
        st.exception(exc)
        st.stop()

    if ndvi_results:
        st.subheader("NDVI Time Series")
        fig = detector.plot_ndvi_time_series(ndvi_results, save_path=None, show=False)
        st.pyplot(fig)

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
