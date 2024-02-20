import requests
import csv

# Define the Overpass API URL
overpass_url = "http://overpass-api.de/api/interpreter"

# Define the Overpass QL query
overpass_query = """
[out:json];
node["place"="city"];
out body;
"""

# Make the request
response = requests.get(overpass_url, params={'data': overpass_query})

if response.status_code == 200:
    data = response.json()
    # Open a file to write
    with open('Data/cities_info.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ID', 'Lat', 'Lon', 'Name', 'Population', 'Population Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for element in data['elements']:
            node_id = element.get('id')
            lat = element.get('lat')
            lon = element.get('lon')
            name = element['tags'].get('name', 'Unknown')
            population = element['tags'].get('population', 'Unknown')
            # Population date is not a standard tag, so it's unlikely to be present
            population_date = element['tags'].get('population:date', 'Unknown')

            # Write the city information to the CSV file
            writer.writerow({'ID': node_id, 'Lat': lat, 'Lon': lon, 'Name': name, 'Population': population, 'Population Date': population_date})
    print("Data successfully written to 'cities_info.csv'")
else:
    print("Failed to fetch data:", response.status_code)
