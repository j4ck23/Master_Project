import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import f_oneway


#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed.csv")

#------------------------------------------Plotting------------------------------------------
#Arrays and dictionarys to store data to be plotted.
cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
years = ['2015', '2016', '2017', '2018', '2019', '2020']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Weekday = ['Weekday', 'Weekend']
holiday = ['Holiday', 'Regular day']
means = []
means_year = {}
mean_week = {}
mean_typeday = {}
mean_Season = {}
mean_time = {}
mean_month = {}
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

for i in range(len(cols)):#For each Pollutant
    means.append(df[cols[i]].mean())#Find mean
    means_year[i] = df.groupby("Year")[cols[i]].mean()#Find mean for each year in dataset
    mean_week[i] = df.groupby("Weekday_or_weekend")[cols[i]].mean()#Find mean for the weekday day catorgory
    mean_typeday[i] = df.groupby("Regular_day_or_holiday")[cols[i]].mean()#Find mean for holiday
    mean_Season[i] = df.groupby("Season")[cols[i]].mean()#Find the mean for each season
    mean_time[i] = df.groupby("Time")[cols[i]].mean()#Find mean at different times of the day
    mean_month[i] = df.groupby("Month")[cols[i]].mean()#Find mean for each month

for i in range(len(mean_week)):#Format for plotting
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

data = df[df.columns[2:9]]#Select columns for correlation plot

for i in range(len(cols)):#Find mean for each Pollutant for each year for an anova test
    mean_2015.append(df.loc[df['Year'] == 2015, cols[i]].mean())
    mean_2016.append(df.loc[df['Year'] == 2016, cols[i]].mean())
    mean_2017.append(df.loc[df['Year'] == 2017, cols[i]].mean())
    mean_2018.append(df.loc[df['Year'] == 2018, cols[i]].mean())
    mean_2019.append(df.loc[df['Year'] == 2019, cols[i]].mean())
    mean_2020.append(df.loc[df['Year'] == 2020, cols[i]].mean())

print(f_oneway(mean_2015, mean_2016, mean_2017, mean_2018, mean_2019, mean_2020))#Anova test for pollution levels over the years

#Plot average levels for each pollutant
plt.figure()
plt.title("Average Pollutant levels")
plt.xlabel("Pollutant")
plt.ylabel("Levels")
plt.bar(cols, means)
plt.show()

#Plot average Pollutant for each year
plt.figure()
plt.title("Average Pollutant levels over years")
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


#plot average Pollutant during weekends vs weekdays and holidays vs normal days
plt.figure()
plt.subplot(1,2,1)
x = np.arange(6)
plt.title("Average Pollutant levels during weekdays")
plt.xlabel("Pollutant")
plt.ylabel("Levels")
plt.bar(x, mean_weekday, width=0.3, color="b")
plt.bar(x+0.3, mean_weekend,width=0.3, color="r")
plt.xticks(x, cols)
plt.legend(['Weekday', 'Weekend'])

plt.subplot(1,2,2)
plt.title("Average Pollutant levels during Holidays")
plt.xlabel("Pollutant")
plt.ylabel("Levels")
plt.bar(x, mean_holiday, width=0.3, color="b")
plt.bar(x+0.3, mean_regday, width=0.3, color="r")
plt.xticks(x, cols)
plt.legend(['Holiday', 'Regular day'])
plt.show()

#Plot average Pollutant levles in different seasons
plt.figure()
x = np.arange(6)
plt.title("Average Pollutant levels in different Seasons")
plt.xlabel("Pollutant")
plt.ylabel("Levels")
plt.bar(x-0.20, mean_winter, width=0.20, color="b")
plt.bar(x, mean_summer, width=0.20, color="r")
plt.bar(x+0.20, mean_monsoon, width=0.20, color="y")
plt.bar(x+0.40, mean_post_monsoon, width=0.20, color="g")
plt.xticks(x, cols)
plt.legend(['Winter', 'Summer', 'Monsoon', 'Post-Monsoon'])
plt.show()

#Plot average Pollutant levels during the day - Check for peaks in pollution
fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
plt.subplot(4,2,1)
plt.title("PM2.5")
plt.plot(np.arange(24), mean_time[0], label="PM2.5", color='r')

plt.subplot(4,2,2)
plt.title("PM10")
plt.plot(np.arange(24), mean_time[1], label="PM10")

plt.subplot(4,2,3)
plt.title("NO2")
plt.plot(np.arange(24), mean_time[2], label="NO2")

plt.subplot(4,2,4)
plt.title("CO")
plt.plot(np.arange(24), mean_time[3], label="CO")

plt.subplot(4,2,5)
plt.title("SO2")
plt.plot(np.arange(24), mean_time[4], label="SO2")

plt.subplot(4,2,6)
plt.title("O3")
plt.plot(np.arange(24), mean_time[5], label="O3")


plt.suptitle("Average Pollutant levels over a day")
plt.tight_layout()
fig.supxlabel("Hour")
fig.supylabel("Levels")
plt.show()

#Average Pollutant levles in different months
fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
plt.subplot(4,2,1)
plt.title("PM2.5")
plt.plot(months, mean_month[0], label="PM2.5")

plt.subplot(4,2,2)
plt.title("PM10")
plt.plot(months, mean_month[1], label="PM10")

plt.subplot(4,2,3)
plt.title("NO2")
plt.plot(months, mean_month[2], label="NO2")

plt.subplot(4,2,4)
plt.title("CO")
plt.plot(months, mean_month[3], label="CO")

plt.subplot(4,2,5)
plt.title("SO2")
plt.plot(months, mean_month[4], label="SO2")

plt.subplot(4,2,6)
plt.title("O3")
plt.plot(months, mean_month[5], label="O3")

plt.suptitle("Average Pollutant levels over a year")
plt.tight_layout()
fig.supxlabel("Year")
fig.supylabel("Levels")
plt.show()

#Correlation heatmap to check for correlation between Pollutants as well as air quality
plt.figure()
heat_map = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
heat_map.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.show()