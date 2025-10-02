"""
Deforestation Detection using Google Earth Engine
Author: Avner Gomes
Project: Land Cover Change Detection with NDVI Analysis
"""

import ee
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import folium
from folium import plugins
import geemap
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class DeforestationDetector:
    """
    A class to detect and analyze deforestation using Google Earth Engine
    and NDVI time series analysis.
    """
    
    def __init__(self, start_year=2015, end_year=2024):
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
        self.start_date = f'{start_year}-01-01'
        self.end_date = f'{end_year}-12-31'
        
        # Initialize Earth Engine
        try:
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
            df = pd.read_csv(csv_path)
            required_columns = ['location_name', 'latitude', 'longitude']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            print(f"✓ Loaded {len(df)} locations from {csv_path}")
            return df
        except Exception as e:
            print(f"✗ Error loading CSV: {e}")
            raise e
    
    def create_buffer_polygon(self, latitude, longitude, buffer_km=5):
        """
        Create a buffer polygon around a point.
        
        Parameters:
        -----------
        latitude : float
            Latitude coordinate
        longitude : float
            Longitude coordinate
        buffer_km : float
            Buffer radius in kilometers
            
        Returns:
        --------
        ee.Geometry
            Buffer polygon geometry
        """
        point = ee.Geometry.Point([longitude, latitude])
        # Convert km to meters
        buffer_meters = buffer_km * 1000
        return point.buffer(buffer_meters)
    
    def calculate_ndvi(self, image):
        """
        Calculate NDVI for a Landsat image.
        
        Parameters:
        -----------
        image : ee.Image
            Landsat image
            
        Returns:
        --------
        ee.Image
            Image with NDVI band added
        """
        ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    def get_landsat_collection(self, geometry):
        """
        Get Landsat image collection for the specified geometry and date range.
        
        Parameters:
        -----------
        geometry : ee.Geometry
            Area of interest
            
        Returns:
        --------
        ee.ImageCollection
            Filtered and processed Landsat collection
        """
        # Landsat 8 (2013-present)
        landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(geometry) \
            .filterDate(self.start_date, self.end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 20))
        
        # Landsat 7 (1999-present) - for better temporal coverage
        landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
            .filterBounds(geometry) \
            .filterDate(self.start_date, self.end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 20))
        
        # Merge collections
        collection = landsat8.merge(landsat7)
        
        # Apply scaling factors for Landsat Collection 2
        def apply_scale_factors(image):
            optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
            return image.addBands(optical_bands, None, True)
        
        collection = collection.map(apply_scale_factors)
        
        # Calculate NDVI
        collection = collection.map(self.calculate_ndvi)
        
        return collection
    
    def extract_ndvi_time_series(self, geometry, location_name):
        """
        Extract NDVI time series for a given geometry.
        
        Parameters:
        -----------
        geometry : ee.Geometry
            Area of interest
        location_name : str
            Name of the location
            
        Returns:
        --------
        pandas.DataFrame
            Time series data with dates and NDVI values
        """
        print(f"\n📊 Processing {location_name}...")
        
        collection = self.get_landsat_collection(geometry)
        
        # Get the number of images
        count = collection.size().getInfo()
        print(f"  Found {count} suitable images")
        
        if count == 0:
            print(f"  ⚠ No images found for {location_name}")
            return pd.DataFrame()
        
        # Extract NDVI statistics
        def extract_stats(image):
            stats = image.select('NDVI').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=30,
                maxPixels=1e9
            )
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'ndvi_mean': stats.get('NDVI'),
                'timestamp': image.date().millis()
            })
        
        features = collection.map(extract_stats)
        feature_list = features.getInfo()['features']
        
        # Convert to DataFrame
        data = []
        for feature in feature_list:
            props = feature['properties']
            if props.get('ndvi_mean') is not None:
                data.append({
                    'date': props['date'],
                    'ndvi_mean': props['ndvi_mean'],
                    'location': location_name
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            print(f"  ✓ Extracted {len(df)} NDVI measurements")
        
        return df
    
    def analyze_deforestation(self, df):
        """
        Analyze NDVI trends to detect deforestation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Time series data
            
        Returns:
        --------
        dict
            Analysis results including trends and statistics
        """
        if df.empty:
            return {
                'trend': None,
                'change_percentage': 0,
                'mean_ndvi_start': 0,
                'mean_ndvi_end': 0,
                'deforestation_detected': False
            }
        
        # Calculate annual averages
        df['year'] = df['date'].dt.year
        annual_means = df.groupby('year')['ndvi_mean'].mean()
        
        # Compare first and last periods
        first_period = df[df['year'] <= df['year'].min() + 2]['ndvi_mean'].mean()
        last_period = df[df['year'] >= df['year'].max() - 2]['ndvi_mean'].mean()
        
        change_percentage = ((last_period - first_period) / first_period) * 100
        
        # Deforestation threshold: >15% decrease in NDVI
        deforestation_detected = change_percentage < -15
        
        results = {
            'trend': 'decreasing' if change_percentage < 0 else 'increasing',
            'change_percentage': change_percentage,
            'mean_ndvi_start': first_period,
            'mean_ndvi_end': last_period,
            'deforestation_detected': deforestation_detected,
            'annual_means': annual_means
        }
        
        return results
    
    def plot_ndvi_time_series(self, df_dict, save_path=None):
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
                ax.text(0.5, 0.5, f'No data available for {location_name}', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{location_name} - No Data')
                continue
            
            # Plot raw data
            ax.scatter(df['date'], df['ndvi_mean'], alpha=0.4, s=30, 
                      label='NDVI measurements', color='#2ecc71')
            
            # Calculate and plot moving average
            df['ndvi_smooth'] = df['ndvi_mean'].rolling(window=5, center=True).mean()
            ax.plot(df['date'], df['ndvi_smooth'], linewidth=2.5, 
                   label='Smoothed trend', color='#27ae60')
            
            # Add annual averages
            df['year'] = df['date'].dt.year
            annual_means = df.groupby('year').agg({
                'ndvi_mean': 'mean',
                'date': 'mean'
            })
            ax.plot(annual_means['date'], annual_means['ndvi_mean'], 
                   'o-', linewidth=2, markersize=8, 
                   label='Annual average', color='#e74c3c')
            
            # Analyze trends
            analysis = self.analyze_deforestation(df)
            
            # Formatting
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('NDVI', fontsize=12, fontweight='bold')
            ax.set_title(f'{location_name} - NDVI Time Series\n'
                        f'Change: {analysis["change_percentage"]:.1f}% | '
                        f'Status: {"⚠ DEFORESTATION DETECTED" if analysis["deforestation_detected"] else "✓ Stable/Improving"}',
                        fontsize=13, fontweight='bold', pad=15)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=10)
            ax.set_ylim([0, 1])
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Plot saved to: {save_path}")
        
        plt.show()
    
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
                
                # Determine marker color based on deforestation status
                color = 'red' if analysis['deforestation_detected'] else 'green'
                icon = 'exclamation-triangle' if analysis['deforestation_detected'] else 'leaf'
                
                popup_html = f"""
                <div style="font-family: Arial; min-width: 200px;">
                    <h4>{location_name}</h4>
                    <p><b>Status:</b> {'⚠ Deforestation Detected' if analysis['deforestation_detected'] else '✓ Stable'}</p>
                    <p><b>NDVI Change:</b> {analysis['change_percentage']:.1f}%</p>
                    <p><b>Start NDVI:</b> {analysis['mean_ndvi_start']:.3f}</p>
                    <p><b>End NDVI:</b> {analysis['mean_ndvi_end']:.3f}</p>
                </div>
                """
            else:
                color = 'gray'
                icon = 'question'
                popup_html = f"<b>{location_name}</b><br>No data available"
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=location_name,
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(m)
            
            # Add circle to show buffer area
            folium.Circle(
                location=[row['latitude'], row['longitude']],
                radius=5000,  # 5km buffer
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.2,
                popup=f"{location_name} - 5km buffer"
            ).add_to(m)
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(m)
        
        return m
    
    def generate_report(self, locations_df, ndvi_data_dict):
        """
        Generate a summary report of the analysis.
        
        Parameters:
        -----------
        locations_df : pandas.DataFrame
            DataFrame with location information
        ndvi_data_dict : dict
            Dictionary with NDVI time series data
        """
        print("\n" + "="*70)
        print("DEFORESTATION DETECTION REPORT")
        print("="*70)
        print(f"Analysis Period: {self.start_year} - {self.end_year}")
        print(f"Number of Locations: {len(locations_df)}")
        print("="*70 + "\n")
        
        for location_name in locations_df['location_name']:
            if location_name in ndvi_data_dict and not ndvi_data_dict[location_name].empty:
                analysis = self.analyze_deforestation(ndvi_data_dict[location_name])
                
                print(f"📍 {location_name}")
                print("-" * 70)
                print(f"  Measurements collected: {len(ndvi_data_dict[location_name])}")
                print(f"  Initial NDVI (avg): {analysis['mean_ndvi_start']:.3f}")
                print(f"  Final NDVI (avg): {analysis['mean_ndvi_end']:.3f}")
                print(f"  Change: {analysis['change_percentage']:+.1f}%")
                print(f"  Trend: {analysis['trend'].upper()}")
                print(f"  Deforestation Status: {'⚠ DETECTED' if analysis['deforestation_detected'] else '✓ NOT DETECTED'}")
                print()
            else:
                print(f"📍 {location_name}")
                print("-" * 70)
                print(f"  ⚠ No data available")
                print()
        
        print("="*70)
        print("End of Report")
        print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for the deforestation detection pipeline.
    """
    print("\n" + "="*70)
    print("DEFORESTATION DETECTION SYSTEM")
    print("Powered by Google Earth Engine")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = DeforestationDetector(start_year=2015, end_year=2024)
    
    # Load locations from CSV
    csv_path = 'locations.csv'
    locations_df = detector.load_locations_from_csv(csv_path)
    
    # Process each location
    ndvi_data_dict = {}
    
    for idx, row in locations_df.iterrows():
        location_name = row['location_name']
        latitude = row['latitude']
        longitude = row['longitude']
        
        print(f"\n{'='*70}")
        print(f"Processing: {location_name}")
        print(f"Coordinates: {latitude}°N, {longitude}°W")
        print(f"{'='*70}")
        
        # Create buffer geometry
        geometry = detector.create_buffer_polygon(latitude, longitude, buffer_km=5)
        
        # Extract NDVI time series
        ndvi_df = detector.extract_ndvi_time_series(geometry, location_name)
        ndvi_data_dict[location_name] = ndvi_df
    
    # Generate visualizations
    print("\n📊 Creating visualizations...")
    detector.plot_ndvi_time_series(ndvi_data_dict, save_path='ndvi_time_series.png')
    
    # Create interactive map
    print("🗺  Creating interactive map...")
    interactive_map = detector.create_interactive_map(locations_df, ndvi_data_dict)
    interactive_map.save('deforestation_map.html')
    print("💾 Interactive map saved to: deforestation_map.html")
    
    # Generate report
    detector.generate_report(locations_df, ndvi_data_dict)
    
    print("\n✅ Analysis complete!")
    print("\nOutputs generated:")
    print("  1. ndvi_time_series.png - Time series plots")
    print("  2. deforestation_map.html - Interactive map")


if __name__ == "__main__":
    # Note: Before running, authenticate with Google Earth Engine
    # Run this once: ee.Authenticate()
    main()
