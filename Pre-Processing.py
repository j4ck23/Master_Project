import pandas as pd

#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Data/station_hour_transformed.csv")
to_drop = ['NO', 'NOx', 'NH3', 'Benzene', 'Toluene', 'Xylene'] #Columns to be dropped
#The Pollutants in to drop are once not being investigated by this research, this was selceted by looking
#At the WHO website on what polluntants contribute to air pollution

#---------------------------------------Pre-Processing---------------------------------------
df = df.drop(columns=to_drop) #Drop columns
print(df.head(5)) #Check dataset
print(df.isnull().sum()) #Count null values
print(len(df)) #Check how many instaces are in the dataset - This can be used as a test to make sure the numbers
#Match what is said on Excel and the website the data is from

df = df.dropna() #Drop Rows with null or NAN values
print(df.head(5)) #Re cehck data
print(df.isnull().sum()) #Re check for nulls
print(len(df))#Check new size of dataset after removeing null data
df[['Date', ' Time']] = df['Datetime'].str.split(' ', expand=True) #Splits the datetime column into two
df = df.drop(columns='Datetime')#Drops the old column
print(df)

#The columns were dropper first as to not drop any rows that contetn full information for what we are looking for

df.to_csv("Processed.csv", index=False)#Output to CSV for future use - Save on computational power.