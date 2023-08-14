import pandas as pd
import statistics
from tabulate import tabulate

#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed.csv")
#print(df)

#--------------------------------------------Stats-------------------------------------------

means = []
ranges = []
StandardDivs = []
minium = []
maximum = []
data = []
names = ['Pollutant', 'Mean', 'Standard Deviation', 'Min', 'Max', 'Range']

cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']

for i in range(len(cols)):
    means.append(df[cols[i]].mean())
    StandardDivs.append(statistics.stdev(df[cols[i]]))
    maximum.append(df[cols[i]].min())
    minium.append(df[cols[i]].max())

for i in range(len(minium)):
    ranges.append(minium[i] - maximum[i])

for i in range(len(cols)):
    data.append([cols[i],means[i],StandardDivs[i], minium[i] ,maximum[i], ranges[i]])


print(tabulate(data, headers=names, tablefmt="fancy_grid"))