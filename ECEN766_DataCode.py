# this program reads in 3 separate csv files (2 for hourly energy data and 1 for daily weather data) and creates a new csv file with data from the loaded in csvs but all the data is matched daily

import pandas as pd

def merge_data(energy_file1, energy_file2, weather_file, output_file):
    """
    Reads hourly energy data from two CSV files and daily weather data from another CSV file.
    Merges the data based on date, aggregating hourly energy data to daily totals,
    and saves the combined data to a new CSV file.

    Args:
        energy_file1 (str): Path to the first hourly energy data CSV file.
        energy_file2 (str): Path to the second hourly energy data CSV file.
        weather_file (str): Path to the daily weather data CSV file.
        output_file (str): Path to save the merged and aggregated data.
    """

    try:
        # Read CSV files into pandas DataFrames
        energy1_df = pd.read_csv(energy_file1)
        energy2_df = pd.read_csv(energy_file2)
        weather_df = pd.read_csv(weather_file)


        # Ensure date columns are datetime objects
        for df in [energy1_df, energy2_df, weather_df]:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])  # Replace 'Date' with actual column name

        # Aggregate hourly energy data to daily totals
        energy1_daily = energy1_df.groupby('Date')['Energy'].sum().reset_index()
        energy2_daily = energy2_df.groupby('Date')['Energy'].sum().reset_index()

        # Rename columns to avoid confusion after merging
        energy1_daily.rename(columns={'Energy': 'Energy1'}, inplace=True)
        energy2_daily.rename(columns={'Energy': 'Energy2'}, inplace=True)


        # Merge energy dataframes
        merged_energy = pd.merge(energy1_daily, energy2_daily, on='Date', how='outer')

        # Merge with weather data
        merged_df = pd.merge(merged_energy, weather_df, on='Date', how='outer')


        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv(output_file, index=False)

        print(f"Merged data saved to {output_file}")

    except FileNotFoundError:
        print("One or more input files not found.")
    except KeyError as e:
        print(f"Error: Column '{e}' not found in one of the input files.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
energy_file1_path = 'Native_Load_2024.csv'
energy_file2_path = 'Native_Load_2025.csv'
weather_file_path = 'DailyWeatherData_Houston_01_2024_to_03_2025.csv'
output_file_path = 'merged_data.csv'

merge_data(energy_file1_path, energy_file2_path, weather_file_path, output_file_path)

     
