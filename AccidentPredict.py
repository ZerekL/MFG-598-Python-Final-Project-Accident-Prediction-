# ZL
# REFERENCES
# https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
# Weather Forecast:
# https://api.weather.gov
# Department of Transportation for percentages for specific weather causes
# Department of Transportation claim that 21% of crashes are due to weather
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from datetime import datetime
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from geopy.geocoders import Nominatim
import os

# Buttons for GUI
class AccidentPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Accident Predictor")
        self.root.geometry("800x600")
        self.StateLabel = tk.Label(root, text="Select a state:")
        self.StateLabel.pack()
        self.state_var = tk.StringVar()
        self.StateDropdown = ttk.Combobox(root, textvariable=self.state_var)
        self.StateDropdown["values"] = ["Arizona", "California", "Nevada"]
        self.StateDropdown.pack()
        self.Loadbutton = tk.Button(root, text="Load Data", command=self.load_data)
        self.Loadbutton.pack()
        self.Date_label = tk.Label(root, text="Select a date option:")
        self.Date_label.pack()
        self.DateOption_var = tk.StringVar()
        self.Datedropdown = ttk.Combobox(root, textvariable=self.DateOption_var)
        self.Datedropdown["values"] = ["Today", "Input Date"]
        self.Datedropdown.current(0)
        self.Datedropdown.pack()
        self.Date_label = tk.Label(root, text="Enter a date (MM-DD):")
        self.DateEntry = tk.Entry(root)
        self.Date_label.pack_forget()
        self.DateEntry.pack_forget()
        self.Datedropdown.bind("<<ComboboxSelected>>", self.ToggleDate)
        self.Filterbutton = tk.Button(root, text="Filter Data", command=self.filter_data)
        self.Filterbutton.pack()
        self.Weatherbutton = tk.Button(root, text="Forecast", command=self.Forecast)
        self.Weatherbutton.pack()
        self.Predictbutton = tk.Button(root, text="Run Prediction", command=self.PredictionResults)
        self.Predictbutton.pack()
        self.Mapbutton = tk.Button(root, text="Generate Map", command=self.GenerateMap)
        self.Mapbutton.pack()
        self.tree = ttk.Treeview(root, columns=("City", "Weather_Condition", "Accident_Count", "Total_Accidents"), show="headings")
        self.tree.heading("City", text="City")
        self.tree.heading("Weather_Condition", text="Weather Condition")
        self.tree.heading("Accident_Count", text="Accident Count")
        self.tree.heading("Total_Accidents", text="Total Accidents")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.accidents_data = None
        self.filtered_data = None

# If the (.csv)s are not made it will create a csv for each state. Limited to AZ, CA, NV
    def SplitAccidentData():
        try:
            Maindata_path = "US_Accidents_March23.csv"
            states = {"CA": "Accidents_CA.csv", "AZ": "Accidents_AZ.csv", "NV": "Accidents_NV.csv"}
            # Check if all output files already exist
            if all(os.path.exists(output_file) for output_file in states.values()):
                messagebox.showinfo("Info", "State files already exist. Skipping split operation.")
                return
            data = pd.read_csv(Maindata_path)
            for state, output_file in states.items():
                state_data = data[data['State'] == state]
                state_data.to_csv(output_file, index=False)
            messagebox.showinfo("Success", "Data has been split into separate state files.")
        except FileNotFoundError:
            messagebox.showerror("Error", f"{Maindata_path} not found.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while splitting data: {e}")
    SplitAccidentData()

# GUI changes to make buttons disappear/appear depending on the decision for Date
    def ToggleDate(self, event):
        if self.DateOption_var.get() == "Input Date":
            self.Date_label.pack()
            self.DateEntry.pack()
            self.Weatherbutton.pack_forget()
            self.Predictbutton.pack_forget()
        else:
            self.Date_label.pack_forget()
            self.DateEntry.pack_forget()
            self.Weatherbutton.pack()
            self.Predictbutton.pack()

# Loads the correct csv data respectively for user input
    def load_data(self):
        state = self.state_var.get().lower()
        state_files = {
            'arizona': 'Accidents_AZ.csv',
            'california': 'Accidents_CA.csv',
            'nevada': 'Accidents_NV.csv'
        }
        
        if state not in state_files:
            messagebox.showerror("Error", "Please select a valid state.")
            return
        state_file = state_files[state]
        try:
            self.accidents_data = pd.read_csv(state_file)
            if 'Start_Time' in self.accidents_data.columns:
                self.accidents_data['Start_Time'] = pd.to_datetime(self.accidents_data['Start_Time'], errors='coerce')
            messagebox.showinfo("Success", f"Loaded data for {state.title()}.")
        except FileNotFoundError:
            messagebox.showerror("Error", f"File {state_file} not found.")

# Filtering the data for the top ten cities and setting up structure for future calculations
    def filter_data(self):
        if self.accidents_data is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        if self.DateOption_var.get() == "Today":
            today = datetime.today()
            month, day = today.month, today.day
        else:
            user_input = self.DateEntry.get().strip().lower()
            try:
                month, day = map(int, user_input.split('-'))
                if not (1 <= month <= 12 and 1 <= day <= 31):
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Invalid date format. Please use MM-DD.")
                return
        self.filtered_data = self.accidents_data[
            self.accidents_data['Start_Time'].apply(
                lambda x: (x.month, x.day) == (month, day) if pd.notnull(x) else False
            )
        ]

        if self.filtered_data.empty:
            messagebox.showinfo("No Data", f"No accidents found for {month:02d}-{day:02d}.")
            return
        self.filtered_data = self.filtered_data.dropna(subset=['City', 'Weather_Condition', 'Severity'])
        AccidentsbyCityWeather = (
            self.filtered_data.groupby(['City', 'Weather_Condition'])
            .size()
            .reset_index(name='Accident_Count')
        )
        
        TotalAccidentsbyCity = (
            AccidentsbyCityWeather.groupby('City')['Accident_Count'].sum().reset_index(name='Total_Accidents')
        )
        
        final_data = pd.merge(AccidentsbyCityWeather, TotalAccidentsbyCity, on='City')
        top_cities = (
            TotalAccidentsbyCity.sort_values(by='Total_Accidents', ascending=False).head(10)['City']
        )
        
        TCData = final_data[final_data['City'].isin(top_cities)].sort_values(
            by=['Total_Accidents', 'City'], ascending=[False, True]
        )
        
        for row in self.tree.get_children():
            self.tree.delete(row)
        for _, row in TCData.iterrows():
            self.tree.insert("", tk.END, values=(row['City'], row['Weather_Condition'], row['Accident_Count'], row['Total_Accidents']))
        # Due to accident having a 21% higher chance likelyhood. I wanted to nutrealize the accidents that
        # did happen in nonclear weather to be weighted down to being an accident likelyhood if it was clear
        # weather instead to create an evenized sum of all the accidents from past years. To predict future.
        # Next line of code will be used with a + for the prediction with the forecast
        temp_file = "temp.csv"
        TCData.to_csv(temp_file, index=False)
        self.AWeatherWeights(temp_file)

# Calc Cleaner for the Accident data. Neutralizes the data to help with prediction.
    def AWeatherWeights(self, temp_file):
        Weight = 0.21
        Fog_Weight = 0.04
        Wetpavement_Weight = 0.42
        Rain_Weight = 0.28
        Snow_Weight = 0.09
        Icypavement_Weight = 0.07
        Snowpavement_Weight = 0.08
        weather_weights = {
            "Light Rain": Rain_Weight,
            "Heavy Rain": Rain_Weight + Wetpavement_Weight,
            "Rain": Rain_Weight + Wetpavement_Weight,
            "Fog": Fog_Weight + Rain_Weight + Wetpavement_Weight,
            "Fog_Cold": Fog_Weight + Rain_Weight + Wetpavement_Weight + Icypavement_Weight,
            "Snow": Snow_Weight + Wetpavement_Weight + Icypavement_Weight + Snowpavement_Weight,
            "Clear": 0
        }
        
        data = pd.read_csv(temp_file)
        data['Weight'] = data['Weather_Condition'].map(weather_weights).fillna(0)
        data['Accidents_From_Weather'] = data['Accident_Count'] * Weight
        data['Accidents_Not_From_Weather'] = data['Accident_Count'] - data['Accidents_From_Weather']
        data['Accidents_Weighted'] = data['Accidents_From_Weather'] * (1 + data['Weight'])
        data['Total_Weighted_Accident'] = data['Accidents_Not_From_Weather'] + data['Accidents_Weighted']
        weighted_totals = (
            data.groupby('City')['Total_Weighted_Accident']
            .sum()
            .reset_index(name='Sum_Weighted_Accidents')
        )
        
        final_data = pd.merge(data, weighted_totals, on='City')
        # 6.75 precious years of data for accidents
        final_data['Yearly_Accidents'] = final_data['Sum_Weighted_Accidents'] / 6.75
        output_file = "tempweight.csv"
        final_data.to_csv(output_file, index=False)
        messagebox.showinfo("Success", f"Weighted data saved to {output_file}.")

# To predict the number of accidents the forecast for the weather must be obtained.
# Only pulls the corresponding cities of their state depending on user input. 
# Cleans the data for future use.
    def Forecast(self):
        try:
            multi_cities = {
                # AZ
                "Phoenix": (33.4484, -112.0740),
                "Tucson": (32.2226, -110.9747),
                "Mesa": (33.4152, -111.8315),
                "Chandler": (33.3062, -111.8413),
                "Scottsdale": (33.4942, -111.9261),
                "Glendale": (33.5387, -112.1860),
                "Tempe": (33.4255, -111.9400),
                "Peoria": (33.5806, -112.2374),
                "Surprise": (33.6292, -112.3679),
                "Yuma": (32.6927, -114.6277),
                # NV
                "Las Vegas": (36.1699, -115.1398),
                "Henderson": (36.0395, -114.9817),
                "Reno": (39.5296, -119.8138),
                "North Las Vegas": (36.1989, -115.1175),
                "Sparks": (39.5349, -119.7527),
                "Carson City": (39.1638, -119.7674),
                "Elko": (40.8324, -115.7631),
                "Fernley": (39.6088, -119.2518),
                "Mesquite": (36.8055, -114.0672),
                "Boulder City": (35.9789, -114.8316),
                # CA
                "Los Angeles": (34.0522, -118.2437),
                "San Diego": (32.7157, -117.1611),
                "San Jose": (37.3382, -121.8863),
                "San Francisco": (37.7749, -122.4194),
                "Fresno": (36.7378, -119.7871),
                "Sacramento": (38.5816, -121.4944),
                "Long Beach": (33.7701, -118.1937),
                "Oakland": (37.8044, -122.2711),
                "Bakersfield": (35.3733, -119.0187),
                "Anaheim": (33.8366, -117.9143),
                "Santa Ana": (33.7455, -117.8677),
                "Riverside": (33.9806, -117.3755),
                "Stockton": (37.9577, -121.2908),
                "Chula Vista": (32.6401, -117.0842),
                "Irvine": (33.6846, -117.8265),
                "Fremont": (37.5483, -121.9886),
                "San Bernardino": (34.1083, -117.2898),
                "Modesto": (37.6391, -120.9969),
                "Fontana": (34.0922, -117.4350),
                "Oxnard": (34.1975, -119.1771),
                "Moreno Valley": (33.9425, -117.2297)
            }
            
            forecast_data_list = []
            for city, (lat, lon) in multi_cities.items():
                base_url = f"https://api.weather.gov/points/{lat},{lon}"
                headers = {
                    "User-Agent": "TrafficDataAnalysis (zleyva@asu.edu)" # Input email
                }
                response = requests.get(base_url, headers=headers)
                response.raise_for_status()
                forecast_url = response.json()["properties"]["forecast"]
                forecast_response = requests.get(forecast_url, headers=headers)
                forecast_response.raise_for_status()
                forecasts = forecast_response.json()["properties"]["periods"][:7]
                for forecast in forecasts:
                    forecast_data_list.append({
                        "City": city,
                        "Date": forecast["startTime"].split("T")[0],
                        "Time": forecast["startTime"].split("T")[1].split("+")[0],
                        "Temperature": forecast["temperature"],
                        "TemperatureUnit": forecast["temperatureUnit"],
                        "WindSpeed": forecast["windSpeed"],
                        "WindDirection": forecast["windDirection"],
                        "DetailedForecast": forecast["detailedForecast"],
                    })
                    
            forecast_df = pd.DataFrame(forecast_data_list)
            Forecastfile = "forecast_weather.csv"
            forecast_df.to_csv(Forecastfile, index=False)
            data = pd.read_csv(Forecastfile)
            
            # Further cleans forecast for prediction calculation
            def ExtractSpecific(text):
                if pd.isnull(text):
                    return None
                text = text.lower()
                keywords = ["clear", "rain", "fog", "snow"]
                for keyword in keywords:
                    if keyword in text:
                        return keyword.capitalize()
                return None
            data['DetailedForecast'] = data['DetailedForecast'].apply(ExtractSpecific)
            data['DetailedForecast'].fillna('Clear', inplace=True)
            # Cleans Forecast windspeed for prediction
            def CleanWS(speed):
                try:
                    nv = int(speed.split()[0])
                    return 'Windy' if nv > 30 else 'Notwindy'
                except (ValueError, AttributeError):
                    return 'Notwindy'
            data['WindSpeed'] = data['WindSpeed'].apply(CleanWS)
            Fog_Weight = 0.04
            Wetpavement_Weight = 0.42
            Rain_Weight = 0.28
            Snow_Weight = 0.09
            Icypavement_Weight = 0.07
            Snowpavement_Weight = 0.08
            weather_weights = {
                "Light Rain": Rain_Weight,
                "Heavy Rain": Rain_Weight + Wetpavement_Weight,
                "Rain": Rain_Weight + Wetpavement_Weight,
                "Fog": Fog_Weight + Rain_Weight + Wetpavement_Weight,
                "Fog_Cold": Fog_Weight + Rain_Weight + Wetpavement_Weight + Icypavement_Weight,
                "Snow": Snow_Weight + Wetpavement_Weight + Icypavement_Weight + Snowpavement_Weight,
                "Clear": 0
            }
            
            data['Weight'] = data['DetailedForecast'].map(weather_weights).fillna(0)
            Forecastcf = "forecast_temp.csv"
            data.to_csv(Forecastcf, index=False)
            messagebox.showinfo("Success", f"Weather forecast data fetched and cleaned. Saved to {Forecastfile} and {Forecastcf}.")
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Error fetching weather data: {e}")
        except FileNotFoundError:
            messagebox.showerror("Error", "Error cleaning forecast data file not found.")
    
    # Prediction Calculation. Pulls from cleaned data and cleaned forecast
    def PredictionResults(self):
        try:
            for row in self.tree.get_children():
                self.tree.delete(row)
            Forecastfp = 'forecast_temp.csv'
            Accidentfp = 'tempweight.csv'
            forecast_data = pd.read_csv(Forecastfp)
            Accidentdata = pd.read_csv(Accidentfp)
            if forecast_data.empty or Accidentdata.empty:
                raise ValueError("Forecast or accident data is empty.")
            cf = forecast_data.groupby(['City']).first().reset_index()
            merged = pd.merge(
                Accidentdata[['City', 'Yearly_Accidents']],
                cf,
                on='City',
                how='left'
            )
            
            selected_state = self.state_var.get().capitalize()
            defc = "Phoenix" if selected_state == "Arizona" else "Los Angeles" if selected_state == "California" else "Las Vegas"
            default_forecast = merged.loc[merged['City'] == defc]
            if not default_forecast.empty:
                default_values = {
                    'DetailedForecast': default_forecast['DetailedForecast'].iloc[0],
                    'WindSpeed': default_forecast['WindSpeed'].iloc[0],
                    'Date': default_forecast['Date'].iloc[0],
                    'Temperature': default_forecast['Temperature'].iloc[0],
                    'Weight': default_forecast['Weight'].iloc[0]
                }
                
                for column, value in default_values.items():
                    merged[column] = merged[column].fillna(value)
            merged = merged.dropna(subset=['Yearly_Accidents'])
            # Calculations
            final_data = merged.drop_duplicates(subset=['City'], keep='first')
            final_data['WeatherBasedCrash'] = final_data['Yearly_Accidents'] * 0.21
            final_data['NonWeatherBasedCrash'] = final_data['Yearly_Accidents'] - final_data['WeatherBasedCrash']
            final_data['WeatherBasedCrashWeighted'] = final_data['WeatherBasedCrash'] * (1 + final_data['Weight'])
            final_data['Estimate'] = final_data['WeatherBasedCrashWeighted'] + final_data['NonWeatherBasedCrash']
            final_data['Prediction'] = final_data['Estimate'].round().apply(lambda x: max(1, x))
            final_data = final_data.sort_values(by='Yearly_Accidents', ascending=False)
            # Removes cities with only one accident probability. One is not enough to set a focus on the area.
            final_data = final_data[final_data['Prediction'] > 1]
            p_file_path = 'Prediction.csv'
            final_data.to_csv(p_file_path, index=False)
            prediction_data = pd.read_csv(p_file_path)
            for _, row in prediction_data.iterrows():
                self.tree.insert(
                    "", tk.END, 
                    values=(row['City'], row['DetailedForecast'], "", row['Prediction'])
                )
                
            print(f"----Prediction Processed to {p_file_path}----")
            print(final_data[['City', 'Prediction']].to_string(index=False))
            print("---Prediction 21% Margin---")
            messagebox.showinfo("Success", f"Predictions saved to {p_file_path}. Check console for details.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")

# Map generation to show Hotspots(Many crashes) and Coldspots(few crashes)
    def GenerateMap(self):
        try:
            prediction_data = pd.read_csv("Prediction.csv")
            selected_state = self.state_var.get().capitalize()
            MapDetails = {
                "Arizona": {
                    "map_path": "Arizonamap.PNG",
                    "extent": [-114.8, -109.1, 30.2, 37.5]  # Arizona map alignment
                },
                "Nevada": {
                    "map_path": "Nevadamap.PNG",
                    "extent": [-120.35, -114.0, 35.0, 42.0]  # Nevada map alignment
                },
                "California": {
                    "map_path": "Californiamap.PNG",
                    "extent": [-125.5, -113.0, 32.5, 42.0]  # California map alignment
                }
            }

            if selected_state not in MapDetails:
                raise ValueError("Map details for the selected state are not defined.")
            map_path = MapDetails[selected_state]["map_path"]
            extent = MapDetails[selected_state]["extent"]
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"Map image file not found for {selected_state}.")
            map_img = mpimg.imread(map_path)
            geolocator = Nominatim(user_agent="geoapi")
            city_coords = []
            for city in prediction_data['City']:
                location = geolocator.geocode(f"{city}, {selected_state}, USA")
                if location:
                    city_coords.append((city, location.latitude, location.longitude))
                else:
                    print(f"Coordinates not found for {city}")
            prediction_data['Prediction'] = prediction_data['Prediction']
            city_colors = ['red' if row['Prediction'] >= 10 else 'purple' for _, row in prediction_data.iterrows()]
            coords_df = pd.DataFrame(city_coords, columns=['City', 'Latitude', 'Longitude'])
            coords_df['Color'] = city_colors
            plt.figure(figsize=(10, 8))
            plt.imshow(map_img, extent=extent)
            plt.title(f"Cities in {selected_state} with Predictions")
            for i, row in coords_df.iterrows():
                plt.scatter(row['Longitude'], row['Latitude'], color=row['Color'], marker='o', s=50, alpha=0.6)
            hotspot_patch = mpatches.Patch(color='red', label='Hotspot')
            coldspot_patch = mpatches.Patch(color='purple', label='Coldspot')
            plt.legend(handles=[hotspot_patch, coldspot_patch], loc='upper right')
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            output_file_path = f"{selected_state}_Accident_Prediction_Map.png"
            plt.savefig(output_file_path, dpi=300)
            plt.show()
            os.startfile(output_file_path)
            messagebox.showinfo("Success", f"Map saved to {output_file_path} and opened.")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating map: {e}")

# Runs tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = AccidentPredictorApp(root)
    root.mainloop()