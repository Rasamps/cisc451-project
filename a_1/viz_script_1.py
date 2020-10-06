import os
import itertools
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

dir = os.getcwd()

os.chdir(dir) #Move to the repo directory.

### Read in any relevant data!
td = pd.read_csv("data/Transit Data - October.csv", header = 0, index_col = False) #Read in the transit data
td = td[(td.Latitude != 0) & (td.Longitude != 0)] #Remove any recording errors.
## The following section reads in the route data such as stop locations.
with open("data/transit-gtfs-routes.json") as json_file: #Open the JSON file
    td_paths = json.load(json_file)

route_nums  = ["702","1","2","502","501","999","15","16","802","602","701",
                "COV","3","4","7","11","10","12","601","6","8","14","801"]

paths = [] #Initialize a list of dataframes containing the locations of stops for each route.
for r,num in zip(td_paths, route_nums):
    if (num == "999" or num == "COV"):
        continue
    paths.append(pd.DataFrame((r["fields"]["shape"]["coordinates"][0]), columns = [num+"_lat", num+"_lon"]))
##

def change_class(row): #Organize the various rider type labels into four over-arching groups.
    if (row in ["Youth","YOUTH","Child","CHILD"]):
        return "Youth"
    elif (row in ["QUEENS","ST LAWRENCE","Student"]):
        return "Student"
    elif (row in ["ADULT","Adult","SENIOR"]):
        return "Adult"
    else:
        return "Transpass"

def get_day(row): #Gets the day of week for a given person boarding the bus.
    day_as_num = (datetime.strptime(row,"%Y-%m-%d  %H:%M")).weekday()
    if (day_as_num == 0):
        return "Monday"
    elif (day_as_num == 1):
        return "Tuesday"
    elif (day_as_num == 2):
        return "Wednesday"
    elif (day_as_num == 3):
        return "Thursday"
    elif (day_as_num == 4):
        return "Friday"
    elif (day_as_num == 5):
        return "Saturday"
    elif (day_as_num == 6):
        return "Sunday"

def bar_plot(td): #Produces the first plot.
    td.Class = td.apply(lambda row: change_class(row.Class), axis = 1) #Apply the change_class function to get the four labels.
    td = td[(td.Route != 8) & (td.Route != 13)] #Eliminate the non-regular routes.
    riders = td.Class.unique()
    routes = td.Route.unique()
    route_count = [] #Count the number of people for each rider class that take a certain route.
    for route in routes:
        temp = []
        for rider in riders:
            temp.append(td.Class[(td.Class == rider) & (td.Route == route)].count())
        route_count.append(temp)

    fig = go.Figure() #Initialize the Plotly Figure that's produced.
    for route,i in zip(routes,range(0,len(routes))):
        fig.add_trace(go.Bar(
        name = str(route), x = riders, y = route_count[i],
        text = str(route), textposition = "outside"
        ))
    fig.update_layout(barmode='group')
    fig.show()

def line_plot(td): #Produces the second plot.
    col_to_add = td.apply(lambda row: get_day(row.Date), axis = 1)
    td.insert(len(td.columns),"day_of_week",col_to_add) #Add a column to the dataset containing the day of week a recording was made.
    td = td[(td.Route != 8) & (td.Route != 13)] #Eliminate the non-regular routes.
    routes = td.Route.unique()
    route_count = [] #Get count values of each route on a given week day.
    for route in routes:
        temp = []
        for day in td.day_of_week.unique():
            temp.append(td.day_of_week[(td.Route == route) & (td.day_of_week == day)].count())
        route_count.append(temp)
    fig = go.Figure() #Initialize Plotly figure to be generated.
    for route,i in zip(routes,range(0,len(routes))):
        fig.add_trace(go.Scatter(
            x = td.day_of_week.unique(), y = route_count[i],
            mode = "lines", name = str(route)
        ))
    annotations = [] #Create annotations denoting which line corresponds to which route.
    for route,i in zip(routes,range(0,len(routes))):
        annotations.append(dict(x = 0.05, y = route_count[i][0],
                                xanchor = 'right', yanchor = 'middle',
                                text = str(route)))
    fig.update_layout(annotations = annotations)
    fig.show()

def route_plot(paths): #Produces the third plot.
    fig = go.Figure() #Intialize the Plotly Figure to be generated.
    for path in paths: #Add each route onto the map.
        fig.add_trace(
            go.Scattermapbox(
                lon = path.iloc[:,0],
                lat = path.iloc[:,1],
                mode = "markers+lines",
            )
        )
    fig.update_layout( #Update the first to use the City of Kingston as its background map.
        mapbox = dict(
            accesstoken = "pk.eyJ1IjoicmFzYW1wcyIsImEiOiJja2F5YWV2OXgwMXR0Mzdtb2sxMHpjZjlhIn0.6a57ZXMCsxgdyvQ7hJoOIw", #Access key for plotting.
            center = go.layout.mapbox.Center(
                lat = 44, lon = -76
            )
        ),
        geo = dict( #Set geographic features of the map.
            scope = 'north america',
            projection_type = 'azimuthal equal area',
            showland = True,
            landcolor = "rgb(243, 243, 243)",
            countrycolor = "rgb(204, 204, 204)"
        )
    )
    fig.show()

def main(): #Main function which calls each plot function.
    print("Beginning plotting process.")
    bar_plot(td)
    print("Done plotting bar plot...")
    line_plot(td)
    print("Done plotting line plot...")
    route_plot(paths)
    print("Done plotting bus routes...")

main()
