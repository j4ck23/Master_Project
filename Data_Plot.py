import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import f_oneway


#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed.csv")

#------------------------------------------Plotting------------------------------------------
cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
years = ['2015', '2016', '2017', '2018', '2019', '2020']
Weekday = ['Weekday', 'Weekend']
holiday = ['Holiday', 'Regular day']
means = []
means_year = {}
mean_week = {}
mean_typeday = {}
mean_Season = {}
mean_time = {}
mean_weekday = []
mean_weekend = []
mean_regday = []
mean_holiday = []
mean_winter = []
mean_summer = []
mean_monsoon = []
mean_post_monsoon = []
mean_2015 = []
mean_2016 = []
mean_2017 = []
mean_2018 = []
mean_2019 = []
mean_2020 = []

for i in range(len(cols)):
    means.append(df[cols[i]].mean())#Find mean
    means_year[i] = df.groupby("Year")[cols[i]].mean()
    mean_week[i] = df.groupby("Weekday_or_weekend")[cols[i]].mean()
    mean_typeday[i] = df.groupby("Regular_day_or_holiday")[cols[i]].mean()
    mean_Season[i] = df.groupby("Season")[cols[i]].mean()
    mean_time[i] = df.groupby("Time")[cols[i]].mean()

for i in range(len(mean_week)):
    mean_weekday.append(mean_week[i][0])
    mean_weekend.append(mean_week[i][1])

for i in range(len(mean_typeday)):
    mean_holiday.append(mean_typeday[i][0])
    mean_regday.append(mean_typeday[i][1])

for i in range(len(mean_Season)):
    mean_winter.append(mean_Season[i][0])
    mean_summer.append(mean_Season[i][1])
    mean_monsoon.append(mean_Season[i][2])
    mean_post_monsoon.append(mean_Season[i][3])

data = df[df.columns[2:8]]

for i in range(len(cols)):
    mean_2015.append(df.loc[df['Year'] == 2015, cols[i]].mean())
    mean_2016.append(df.loc[df['Year'] == 2016, cols[i]].mean())
    mean_2017.append(df.loc[df['Year'] == 2017, cols[i]].mean())
    mean_2018.append(df.loc[df['Year'] == 2018, cols[i]].mean())
    mean_2019.append(df.loc[df['Year'] == 2019, cols[i]].mean())
    mean_2020.append(df.loc[df['Year'] == 2020, cols[i]].mean())

print(f_oneway(mean_2015, mean_2016, mean_2017, mean_2018, mean_2019, mean_2020))

plt.figure()
plt.title("Average polluntant levels")
plt.xlabel("Pollutant")
plt.ylabel("Levels")
plt.bar(cols, means)
plt.show()

plt.figure()
plt.title("Average polluntant levels over years")
plt.xlabel("year")
plt.ylabel("Levels")
plt.plot(years, means_year[0], label="PM2.5")
plt.plot(years, means_year[1], label="PM10")
plt.plot(years, means_year[2], label="NO2")
plt.plot(years, means_year[3], label="CO")
plt.plot(years, means_year[4], label="SO2")
plt.plot(years, means_year[5], label="O3")
plt.legend()
plt.show()


plt.figure()
plt.subplot(1,2,1)
x = np.arange(6)
plt.title("Average polluntant levels during weekdays")
plt.xlabel("Pollutant")
plt.ylabel("Levels")
plt.bar(x, mean_weekday, width=0.3, color="b")
plt.bar(x+0.3, mean_weekend,width=0.3, color="r")
plt.xticks(x, cols)
plt.legend(['Weekday', 'Weekend'])

plt.subplot(1,2,2)
plt.title("Average polluntant levels during Holidays")
plt.xlabel("Pollutant")
plt.ylabel("Levels")
plt.bar(x, mean_holiday, width=0.3, color="b")
plt.bar(x+0.3, mean_regday, width=0.3, color="r")
plt.xticks(x, cols)
plt.legend(['Holiday', 'Regular day'])
plt.show()

plt.figure()
x = np.arange(6)
plt.title("Average polluntant levels in different Seasons")
plt.xlabel("Pollutant")
plt.ylabel("Levels")
plt.bar(x-0.20, mean_winter, width=0.20, color="b")
plt.bar(x, mean_summer, width=0.20, color="r")
plt.bar(x+0.20, mean_monsoon, width=0.20, color="y")
plt.bar(x+0.40, mean_post_monsoon, width=0.20, color="g")
plt.xticks(x, cols)
plt.legend(['Winter', 'Summer', 'Monsoon', 'Post-Monsoon'])
plt.show()


plt.figure()
plt.title("Average polluntant levels during the day")
plt.xlabel("Time of day")
plt.ylabel("Levels")
plt.plot(np.arange(24), mean_time[0], label="PM2.5")
plt.plot(np.arange(24), mean_time[1], label="PM10")
plt.plot(np.arange(24), mean_time[2], label="NO2")
plt.plot(np.arange(24), mean_time[3], label="CO")
plt.plot(np.arange(24), mean_time[4], label="SO2")
plt.plot(np.arange(24), mean_time[5], label="O3")
plt.legend()
plt.show()

plt.figure()
heat_map = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
heat_map.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.show()