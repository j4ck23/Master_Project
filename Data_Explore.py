import pandas as pd
import statistics
from tabulate import tabulate

#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed.csv")
#print(df)

#--------------------------------------------Stats-------------------------------------------

#Define lists to hold values we seek
means = []
ranges = []
StandardDivs = []
minium = []
maximum = []
data = []
names = ['Pollutant', 'Mean', 'Standard Deviation', 'Min', 'Max', 'Range']
cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']
cols_nom = ['StationId', 'AQI_Bucket', 'StationName', 'City', 'State', 'Status', 'Region', 'Day_period', 'Month', 'Year', 'Season', 'Weekday_or_weekend', 'Regular_day_or_holiday', 'AQ_Acceptability']#Nominal columns
nom_val = {}

#For every column in the list we wish to find
for i in range(len(cols)):
    means.append(df[cols[i]].mean())#Find mean
    StandardDivs.append(statistics.stdev(df[cols[i]]))#Find stdiv
    maximum.append(df[cols[i]].min())#find min
    minium.append(df[cols[i]].max())#find max

for i in range(len(minium)):
    ranges.append(minium[i] - maximum[i])#Work out range for each instance

for i in range(len(cols)):
    data.append([cols[i],means[i],StandardDivs[i], minium[i] ,maximum[i], ranges[i]])#compound to a array

for i in range(len(cols_nom)):
    nom_val[cols_nom[i]] = df[cols_nom[i]].value_counts()#found mode for nominal values



print(tabulate(data, headers=names, tablefmt="fancy_grid"))#print numeric stats

print(nom_val)#print mode