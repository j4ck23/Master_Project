import pandas as pd
import statistics
from tabulate import tabulate
import matplotlib.pyplot as plt

#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed.csv")
print(df)

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
#year = []

#For every column in the list we wish to find
for i in range(len(cols)):
    means.append(df[cols[i]].mean())#Find mean
    StandardDivs.append(statistics.stdev(df[cols[i]]))#Find stdiv
    minium.append(df[cols[i]].min())#find min
    maximum.append(df[cols[i]].max())#find max

for i in range(len(minium)):
    ranges.append(minium[i] - maximum[i])#Work out range for each instance

for i in range(len(cols)):
    data.append([cols[i],means[i],StandardDivs[i], minium[i] ,maximum[i], ranges[i]])#compound to a array

for i in range(len(cols_nom)):
    nom_val[cols_nom[i]] = df[cols_nom[i]].value_counts()#found mode for nominal values

seasons = df["Season"].value_counts()
year = df["Year"].value_counts()
years = list(year)
weekday = df["Weekday_or_weekend"].value_counts()
holiday = df["Regular_day_or_holiday"].value_counts()
acceptable = df["AQ_Acceptability"].value_counts()
period = df["Day_period"].value_counts()

print(tabulate(data, headers=names, tablefmt="fancy_grid"))#print numeric stats

print(nom_val)#print mode

plt.figure()
plt.title("Seasons Histogram")
plt.xlabel("Seasons")
plt.ylabel("Instances")
plt.bar("Monsoon", seasons[0])
plt.bar("Summer", seasons[1])
plt.bar("Post-Monsoon", seasons[2])
plt.bar("Winter", seasons[3])
plt.show()

plt.figure()
plt.title("Year Histogram")
plt.xlabel("Year")
plt.ylabel("Instances")
plt.bar("2015", years[5])
plt.bar("2016", years[4])
plt.bar("2017", years[3])
plt.bar("2018", years[1])
plt.bar("2019", years[0])
plt.bar("2020", years[2])
plt.show()

plt.figure()
plt.title("Day Histogram")
plt.xlabel("Day")
plt.ylabel("Instances")
plt.bar("Weekday", weekday[0])
plt.bar("Weekend", weekday[1])
plt.show()

plt.figure()
plt.title("Holiday Histogram")
plt.xlabel("Type of day")
plt.ylabel("Instances")
plt.bar("Holiday", holiday[1])
plt.bar("Regular", holiday[0])
plt.show()

plt.figure()
plt.title("Air Quality Histogram")
plt.xlabel("Air Quality Acceptable")
plt.ylabel("Instances")
plt.bar("Acceptable", acceptable[1])
plt.bar("Unacceptable", acceptable[0])
plt.show()

plt.figure()
plt.title("Day Period Histogram")
plt.xlabel("Period of day")
plt.ylabel("Instances")
plt.bar("Morning", period[1])
plt.bar("Afternoon", period[2])
plt.bar("Evening", period[3])
plt.bar("Night", period[0])
plt.show()