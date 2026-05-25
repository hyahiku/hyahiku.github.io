# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 09:24:17 2025

@author: haley
"""

import pandas as pd, random, matplotlib.pyplot as plt, numpy as np

# Import the csv files and assign them accordingly
Africa_Q = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Africa_Quantity.csv")  
Africa_V = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Africa_Value.csv")
Americas_Q = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Americas_Quantity.csv")
Americas_V = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Americas_Value.csv")
Asia_Q = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Asia_Quantity.csv")
Asia_V = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Asia_Value.csv")
Europe_Q = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Europe_Quantity.csv")
Europe_V = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Europe_Value.csv")
Global_Q = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Global_Quantity.csv")
Global_V = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Global_Value.csv")
Oceania_Q = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Oceania_Quantity.csv")
Oceania_V = pd.read_csv(r"C:\Users\haley\OneDrive\Desktop\School\Spring '24\DA 621\Project\Oceania_Value.csv")

# ------ CLEAN THE DATA ------
def process_data(csv_files):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    # Concatenate all DataFrames in the list along rows
    merged_df = pd.concat(csv_files, ignore_index=True)

    # Remove columns that start with capital 'S'
    merged_df = merged_df.loc[:, ~merged_df.columns.str.startswith('S')]

    # Rename the 'Land Area' column to 'Country'
    merged_df = merged_df.rename(columns={'Land Area': 'Country'})

    # Remove rows where the value in the 'Commodity' column is 'Crustaceans'
    merged_df = merged_df[merged_df['Commodity'] != 'Crustaceans']
    
    # Remove rows where columns beginning with '2' are empty or 0
    relevant_columns = [col for col in merged_df.columns if col.startswith('2')]
    merged_df = merged_df[(merged_df[relevant_columns] != 0).all(axis=1)]
    
    # Convert columns starting with '2' to integer types
    for col in relevant_columns:
        merged_df[col] = merged_df[col].astype(int)

    return merged_df

# Put the DataFrame variables into a list
csv_files_Q = [Africa_Q, Americas_Q, Asia_Q, Europe_Q, Global_Q, Oceania_Q]
csv_files_V = [Africa_V, Americas_V, Asia_V, Europe_V, Global_V, Oceania_V]

# Process the data for quanitity and value 
Q_data = process_data(csv_files_Q)
V_data = process_data(csv_files_V)

# Select columns starting with '2'
V_years = V_data.loc[:, V_data.columns.str.startswith('2')]


def import_sum(country, Q_data):
    # Initialize the import sum
    import_sum = 0

    # Check if the country exists in the 'Country' column of Q_data
    if country in Q_data['Country'].values:
        # Filter rows where 'Country' matches the given country and 'Trade flow' is 'Import'
        filtered_rows = Q_data[(Q_data['Country'] == country) & (Q_data['Trade flow'] == 'Import')]
        
        # Iterate through each row in filtered rows
        for index, row in filtered_rows.iterrows():
            # Iterate through each column in Q_data
            for column in Q_data.columns:
                # Check if the column name starts with '2'
                if column.startswith('2'):
                    # Add the value in the column to the import_sum
                    import_sum += row[column]
    
    # Format the import sum with commas every three digits from the right
    formatted_import_sum = "{:,.0f}".format(import_sum)

    return formatted_import_sum



def export_sum(country, Q_data):
    # Initialize the export sum
    export_sum = 0

    # Check if the country exists in the 'Country' column of Q_data
    if country in Q_data['Country'].values:
        # Filter rows where 'Country' matches the given country and 'Trade flow' is 'Export'
        filtered_rows = Q_data[(Q_data['Country'] == country) & (Q_data['Trade flow'] == 'Export')]
        
        # Iterate through each row in filtered rows
        for index, row in filtered_rows.iterrows():
            # Iterate through each column in Q_data
            for column in Q_data.columns:
                # Check if the column name starts with '2'
                if column.startswith('2'):
                    # Add the value in the column to the export_sum
                    export_sum += row[column]
      
    # Format the export sum with commas every three digits from the right
    formatted_export_sum = "{:,.0f}".format(export_sum)

    return formatted_export_sum


      
def import_avg(country, Q_data):
    # Initialize the import sum
    import_sum = 0

    # Check if the country exists in the 'Country' column of Q_data
    if country in Q_data['Country'].values:
        # Filter rows where 'Country' matches the given country and 'Trade flow' is 'Import'
        filtered_rows = Q_data[(Q_data['Country'] == country) & (Q_data['Trade flow'] == 'Import')]
        
        # Iterate through each row in filtered rows
        for index, row in filtered_rows.iterrows():
            # Iterate through each column in Q_data
            for column in Q_data.columns:
                # Check if the column name starts with '2'
                if column.startswith('2'):
                    # Add the value in the column to the import_sum
                    import_sum += row[column]
                    import_sum = round(import_sum/15)
    
    # Format the import sum with commas every three digits from the right
    import_avg = "{:,.0f}".format(import_sum)

    return import_avg

    
# sum divided by the number of units
def export_avg(country, Q_data):
    # Initialize the export sum
    export_sum = 0

    # Check if the country exists in the 'Country' column of Q_data
    if country in Q_data['Country'].values:
        # Filter rows where 'Country' matches the given country and 'Trade flow' is 'Export'
        filtered_rows = Q_data[(Q_data['Country'] == country) & (Q_data['Trade flow'] == 'Export')]
        
        # Iterate through each row in filtered rows
        for index, row in filtered_rows.iterrows():
            # Iterate through each column in Q_data
            for column in Q_data.columns:
                # Check if the column name starts with '2'
                if column.startswith('2'):
                    # Add the value in the column to the export_sum
                    export_sum += row[column]
                    export_sum = round(export_sum/15)
      
    # Format the export sum with commas every three digits from the right
    export_avg = "{:,.0f}".format(export_sum)

    return export_avg


# sum of all values for a particular country. Exports & imports included
def value_sum(country, V_data): 
    # Initialize the sum
    total_sum = 0
    
    # Check if the country exists in the 'Country' column of V_data
    if country in V_data['Country'].values:
        # Iterate through each column in V_data
        for column in V_data.columns:
            # Check if the column name starts with '2'
            if column.startswith('2'):
                # Add the sum of values in the column to the total_sum
                total_sum += V_data[V_data['Country'] == country][column].sum()
    
        # Format the total sum with commas every three digits from the right
        formatted_total_sum = "{:,.0f}".format(total_sum)

        return formatted_total_sum
    

# Initialize the sum of values in columns starting with '2'
V_total_sum = 0
num_rows = len(V_data)

# Iterate through each row in V_data
for index, row in V_data.iterrows():
    # Sum the values in columns starting with '2' for the current row
    row_sum = sum(row[column] for column in V_data.columns if column.startswith('2'))
     
    # Add the sum of the current row to the total sum
    V_total_sum += row_sum
     
    # Calculate the average
    V_average = round(V_total_sum / num_rows)  


def money_rank(country, V_data, V_average):
    # Initialize the sum
    total_sum = 0
        
    # Check if the country exists in the 'Country' column of V_data
    if country in V_data['Country'].values:
        # Iterate through each column in V_data
        for column in V_data.columns:
            # Check if the column name starts with '2'
            if column.startswith('2'):
                # Add the sum of values in the column to the total_sum
                total_sum += V_data[V_data['Country'] == country][column].sum()

    # Compare total_sum with V_average
    if total_sum == int(V_average):
        return "about"
    elif total_sum < int(V_average):
        return "less than"
    else:
        return "more than"

    # Call the inner function and return its result
    return value_sum(country, V_data)


def lowest_year(country, V_data):
    # Initialize variables to track the minimum value and its corresponding column name
    min_value = float('inf')  # Initialize with positive infinity to ensure any value will be smaller
    min_column = None

    # Check if the country exists in the 'Country' column of V_data
    if country in V_data['Country'].values:
        # Iterate through each column in V_data that starts with '2'
        for column in V_data.columns:
            if column.startswith('2'):
                # Get the minimum value in the current column for the chosen country
                country_min = V_data[V_data['Country'] == country][column].min()
                
                # Update the minimum value and its corresponding column name if needed
                if country_min < min_value:
                    min_value = country_min
                    min_column = column

    return min_column  # Return the column name with the lowest value


def highest_year(country, V_data):
    # Initialize variables to track the minimum value and its corresponding column name
    max_value = float('inf')  # Initialize with positive infinity to ensure any value will be smaller
    max_column = None
    
    # Check if the country exists in the 'Country' column of V_data
    if country in V_data['Country'].values:
        # Iterate through each column in V_data that starts with '2'
        for column in V_data.columns:
            if column.startswith('2'):
                # Get the minimum value in the current column for the chosen country
                country_max = V_data[V_data['Country'] == country][column].max()

                # Update the minimum value and its corresponding column name if needed
                if country_max < max_value:
                    max_value = country_max
                    max_column = column

    return max_column  # Return the column name with the lowest value

def forecast_growth(country, V_data):
    # Get the year columns (e.g., '2000', ..., '2015')
    year_cols = [col for col in V_data.columns if col.startswith('2')]
    years = [int(col) for col in year_cols]

    # Get the total value per year for the given country
    values = []
    for col in year_cols:
        yearly_total = V_data[V_data['Country'] == country][col].sum()
        values.append(yearly_total)

    # Check if we have enough data
    if len(values) < 2:
        return 0

    # Perform linear regression: y = m*x + b
    x = np.arange(len(years))  # 0, 1, 2, ..., len-1
    y = np.array(values)
    m, b = np.polyfit(x, y, 1)

    # Forecast the next 5 years beyond available data
    future_indices = np.arange(len(years), len(years) + 5)
    forecast = np.sum(m * future_indices + b)

    return round(forecast)


def visualize_growth(country, V_data):

    # Get year columns and numeric years
    year_cols = [col for col in V_data.columns if col.startswith('2')]
    years = [int(col) for col in year_cols]

    # Get the total trade value per year
    values = []
    for col in year_cols:
        val = V_data[V_data['Country'] == country][col].sum()
        values.append(val)

    if len(values) < 2:
        print("Not enough data to forecast.")
        return

    # Linear regression on historical data
    x = np.arange(len(values))
    y = np.array(values)
    m, b = np.polyfit(x, y, 1)

    # Forecast next 5 years
    forecast_x = np.arange(len(values), len(values) + 5)
    forecast_y = m * forecast_x + b

    # Combine years and values
    all_years = years + [years[-1] + i + 1 for i in range(5)]
    all_values = values + list(forecast_y)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(all_years, all_values, marker='o', label="Total Trade Value")
    plt.axvline(x=years[-1], color='gray', linestyle='--', label="Forecast Starts")
    plt.title(f"{country} – Historical and Forecasted Trade Value")
    plt.xlabel("Year")
    plt.ylabel("Value (USD)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




while True:
    try:
        # Ask the user what country they live in
        country = str(input("What country do you live in? ")).capitalize()

        # Check if the country exists in the data
        if country in Q_data['Country'].values or country in V_data['Country'].values:
            
            growth = "{:,.0f}".format(forecast_growth(country, V_data))
            
            print(
               f"""
               {country} has imported a total of {import_sum(country, Q_data)}, and exported a total of \n
               {export_sum(country, Q_data)} fish between the years 2000 and 2015. On average, {country} imports \n
               {import_avg(country, Q_data)} fish per year while exporting about {export_avg(country, Q_data)}. \n
               All of this fish movement values at ${value_sum(country, V_data)} over 15 years. This is \n
               {money_rank(country, V_data, V_average)} average compared to other countries in the same time span. \n
               {lowest_year(country, V_data)} made the least revenue while {highest_year(country, V_data)} made \n
               the most. We can expect to see an growth of ${growth} over the next 5 years \n""")

            visualize_growth(country, V_data)

            break
        else:
            print("Invalid country! Please enter a valid country.")
    except ValueError:
        print("Whoa there")

    # If input is invalid, provide choices
    print("Sorry! Your country's not on the list. Please choose one of the following:")
    print("1. Enter a different country")
    print("2. Give me information about a random country!")
    print("3. Exit")
    user_choice = input("Enter your choice (1, 2, or 3): ")

    if user_choice == '1':
        try:
            country = str(input("Which country do you want to learn more about? \n")).capitalize()
            
            growth = "{:,.0f}".format(forecast_growth(country, V_data))
            
            print(
               f"""
               {country} has imported a total of {import_sum(country, Q_data)}, and exported a total of \n
               {export_sum(country, Q_data)} fish between the years 2000 and 2015. On average, {country} imports \n
               {import_avg(country, Q_data)} fish per year while exporting about {export_avg(country, Q_data)}. \n
               All of this fish movement values at ${value_sum(country, V_data)} over 15 years. This is \n
               {money_rank(country, V_data, V_average)} average compared to other countries in the same time span. \n
               {lowest_year(country, V_data)} made the least revenue while {highest_year(country, V_data)} made \n
               the most. We can expect to see an growth of ${growth} over the next 5 years \n""")
    
            visualize_growth(country, V_data)
        
        except ValueError:
            print("Whoa there")
        
    elif user_choice == '2':
    
        # Get unique country values from both Q_data and V_data
        unique_countries_Q = Q_data['Country'].unique()
        unique_countries_V = V_data['Country'].unique()

        # Combine unique country values from both DataFrames
        unique_countries = list(set(unique_countries_Q) | set(unique_countries_V))

        # Randomly choose a country from the combined list
        chosen_country = random.choice(unique_countries)

        country = chosen_country
        
        growth = "{:,.0f}".format(forecast_growth(country, V_data))

    
        print(
            f"""
            {country} has imported a total of {import_sum(country, Q_data)}, and exported a total of \n
            {export_sum(country, Q_data)} fish between the years 2000 and 2015. On average, {country} imports \n
            {import_avg(country, Q_data)} fish per year while exporting about {export_avg(country, Q_data)}. \n
            All of this fish movement values at ${value_sum(country, V_data)} over 15 years. This is \n
            {money_rank(country, V_data, V_average)} average compared to other countries in the same time span. \n
            {lowest_year(country, V_data)} made the least revenue while {highest_year(country, V_data)} made \n
            the most. We can expect to see an growth of ${growth} over the next 5 years \n""")
            
        visualize_growth(country, V_data)
       
    elif user_choice == '3':
    # Exit the program
        print("Bye!")
        break

    else:
        # Handle invalid choice
        print("Invalid choice! Please enter 1, 2, or 3.")
