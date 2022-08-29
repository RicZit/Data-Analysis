import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

df = pd.read_excel('NBA Stats.xlsx')
print(df.head())
df.drop('Column1', axis=1, inplace=True) ##remove rebundant column
df.set_index('Year', inplace=True) ##set Year as index
JYears = df.index.where(df['Player'] == 'Michael Jordan*').dropna() ##Define the "Jordan Years", as the years of interest for the analysis
threePP = df['3P%'].where(df['Player'] == 'Michael Jordan*').dropna() ##Three Point Percentage (3P performed/3P attempted*100)
twoPP = df['2P%'].where(df['Player'] == 'Michael Jordan*').dropna()/1000 ##Two Point Percentage (In the data source the ratio was provided as multiplied by a factor 1000
MTPPperyear = []
TPperyear = []
M2PPperyear = []
mjmaxfp = []
for year in JYears:
    MTPPperyear.append(df['3P%'].where(df.index == year).mean()) ##Mean Triple point percentage
    M2PPperyear.append(df['2P%'].where(df.index == year).mean()/1000) ##Mean double point percentage
    if year == JYears[0]:
        maxfieldpoints = pd.DataFrame(df[df.index == year].groupby('Tm')['PTS'].sum())
    else:
        maxfieldpoints = pd.merge(maxfieldpoints, pd.Series(df[df.index == year].groupby('Tm')['PTS'].sum(), name=year), how='outer',
                                  right_index=True, left_index=True) ##Max point of field per team per year
    mjmaxfp.append(df['PTS'].where((df.index == year) & (df['Player']=='Michael Jordan*')).sum()) ##total point of Jordan per year
maxfieldpoints.rename({'PTS': JYears[0]}, axis=1)
print(maxfieldpoints)
print(mjmaxfp)
#########Plotting of relevant data
plt.figure(1)
plt.scatter(JYears, threePP, label='J3')
plt.plot(JYears, MTPPperyear, label='m3 per year in J years') ##3P Jordan vs yearly mean of players
plt.legend()
plt.figure(2)
plt.scatter(JYears, twoPP, label='J2')
plt.plot(JYears, M2PPperyear, label='m2 per year in J years') ##2P Jordan vs yearly mean of players
plt.legend()
plt.figure(3)
for rows in maxfieldpoints.index:
    if 'CHI' in rows:
        plt.scatter(JYears, maxfieldpoints.loc[rows], marker = '^', label=rows, color='m') ##Yearly point for Chicago Bulls
        plt.scatter(JYears, mjmaxfp, marker='*', label='MJmaxfieldpoints', color='r') ##Yearly point Jordan

    #else:
        #plt.scatter(JYears, maxfieldpoints.loc[rows], label=rows)
plt.legend()
plt.show()
