
#%%
import pandas as pd 
import numpy as np
import folium
from folium import plugins
from branca.element import Figure
import matplotlib.pyplot as plt

# %%
transit_data = pd.read_csv('/Users/emmaritcey/Documents/School/CISC451/Assignment1/Transit Data - October.csv')
#%%
#drop rows where either longitude or latitude equal zero
transit_data = transit_data.drop(transit_data[transit_data.Latitude == 0].index)
transit_data = transit_data.drop(transit_data[transit_data.Longitude == 0].index)

#%%
# get rid of locations with latitude and logitude outside of where Kingston bus routes run
transit_data = transit_data.drop(transit_data[transit_data.Latitude < 44.20953].index)
transit_data = transit_data.drop(transit_data[transit_data.Latitude > 44.28924].index)
transit_data = transit_data.drop(transit_data[transit_data.Longitude < -76.71167].index)
transit_data = transit_data.drop(transit_data[transit_data.Longitude > -76.39638].index)

# %%
#create numpy array of the latitude and longitude values of everyone and just students
coordArray = transit_data[['Latitude', 'Longitude']].values
students = transit_data.loc[(transit_data['Class'] == 'QUEENS') | (transit_data['Class'] == 'ST LAWRENCE')]
cA_students = students[['Latitude', 'Longitude']].values

#%%
#create two maps with the location set to Kingston
fig1=Figure(width=900,height=500)
m1=folium.Map(location=[44.2520, -76.525], zoom_start=12)
fig1.add_child(m1)
fig2=Figure(width=900,height=500)
m2=folium.Map(location=[44.2520, -76.525], zoom_start=12)
fig2.add_child(m2)
#%%
#first map is a heatmap showing where people got on the bus 
m1.add_children(plugins.HeatMap(coordArray, radius=8))
#%%
#second map is a heatmap showing where students got on the bus
m2.add_children(plugins.HeatMap(cA_students, radius=8))

#%%
#import bus stop data
#make sure there are no records with latitude and longitude values that don't make sense,
bus_stops = pd.read_csv('transit-gtfs-stops.csv', encoding='ISO-8859-1')
bus_stops = bus_stops.drop(bus_stops[bus_stops.Latitude == 0].index)
bus_stops = bus_stops.drop(bus_stops[bus_stops.Longitude == 0].index)
#put latitude and longitute values into a numpy array
bs_array = bus_stops[['Latitude', 'Longitude']].values

#%%
#create third map
fig3=Figure(width=900,height=500)
m3=folium.Map(location=[44.2520, -76.525], zoom_start=12)
fig3.add_child(m3)
#add heatmap to map of bus stops to show where the areas with high density of
#bus stops are
m3.add_children(plugins.HeatMap(bs_array, radius=14))

#%%
#read in parking data
#make sure to get rid of any records that contain errors/missing values
#we also only care about the parking within the same region as the bus stops
parking = pd.read_csv('Parking.csv')
parking = parking.dropna(0) #drop rows with mising values
parking = parking.drop(parking[parking.latitude == 0].index)
parking = parking.drop(parking[parking.longitude == 0].index)
parking = parking.drop(parking[parking.latitude < 44.20953].index)
parking = parking.drop(parking[parking.latitude > 44.28924].index)
parking = parking.drop(parking[parking.longitude < -76.71167].index)
parking = parking.drop(parking[parking.longitude > -76.39638].index)
#put latitude and longitude values into a numpy array
parkingCoords = parking[['latitude', 'longitude']].values

#%%
fig4=Figure(width=900,height=500)
m4=folium.Map(location=[44.2520, -76.525], zoom_start=12)

# put a blue circle of radius 500m around each bus stop
for index, row in bus_stops.iterrows():
    folium.Circle([row['Latitude'], row['Longitude']],
                        radius = 500, popup=row['Name'],
                        fill_color = "#3db7e4",
                        fill_opacity = 0.1,
                        color = '#3db7e4',
                        opacity = 0.1).add_to(m4)
# mark each parking area with a red circle of radius 100m
for index, row in parking.iterrows():
    folium.Circle([row['latitude'], row['longitude']],
                        radius = 100, popup=row['PARKING_AREA_ID'],
                        fill_color = "#FF5533",
                        fill_opacity = 0.2,
                        color = '#FF5533',
                        opacity = 0.2).add_to(m4)
# also explicitly mark each bus stop with small black circle
for index, row in bus_stops.iterrows():
    folium.Circle([row['Latitude'], row['Longitude']],
                        radius = 0.5, popup=row['Name'],
                        fill_color = "#000000",
                        fill_opacity = 1,
                        color = '#000000',
                        opacity = 1).add_to(m4)
fig4.add_child(m4)



#%%
driveways = pd.read_csv('/Users/emmaritcey/Documents/School/CISC451/Assignment1/driveways.csv')
#get rid of extra column
driveways = driveways.drop(columns = ['Unnamed: 3'])
#drop rows with mising values
driveways = driveways.dropna(0) 

#%%
#make sure there are no latitude or longitude values of 0
driveways = driveways.drop(driveways[driveways.Latitude == 0].index)
driveways = driveways.drop(driveways[driveways.Longitude == 0].index)
#%%

#%%
fig5=Figure(width=950,height=600)
m5=folium.Map(location=[44.2520, -76.525], zoom_start=12)

#put a black dot for each driveway
for index, row in driveways.iterrows():
    folium.Circle([row['Latitude'], row['Longitude']],
                        radius = 1,
                        fill_color = "#000000",
                        fill_opacity = 1,
                        color = '#000000',
                        opacity = 1).add_to(m5)
fig5.add_child(m5)
