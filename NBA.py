import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

from typing import List

df = pd.read_excel('NBA Stats.xlsx')
print(df.head())
df.drop('Column1', axis=1, inplace=True)
df.set_index('Year', inplace=True)
JYears = df.index.where(df['Player'] == 'Michael Jordan*').dropna()
threePP = df['3P%'].where(df['Player'] == 'Michael Jordan*').dropna()
twoPP = df['2P%'].where(df['Player'] == 'Michael Jordan*').dropna()/1000
MTPPperyear = []
TPperyear = []
M2PPperyear = []
mjmaxfp = []
for year in JYears:
    MTPPperyear.append(df['3P%'].where(df.index == year).mean())
    M2PPperyear.append(df['2P%'].where(df.index == year).mean()/1000)
    if year == JYears[0]:
        maxfieldpoints = pd.DataFrame(df[df.index == year].groupby('Tm')['PTS'].sum())
    else:
        maxfieldpoints = pd.merge(maxfieldpoints, pd.Series(df[df.index == year].groupby('Tm')['PTS'].sum(), name=year), how='outer',
                                  right_index=True, left_index=True)
    mjmaxfp.append(df['PTS'].where((df.index == year) & (df['Player']=='Michael Jordan*')).sum())
maxfieldpoints.rename({'PTS': 1985.0}, axis=1)
print(maxfieldpoints)
print(mjmaxfp)
plt.figure(1)
plt.scatter(JYears, threePP, label='J3')
plt.plot(JYears, MTPPperyear, label='m3 per year in J years')
plt.legend()
plt.figure(2)
plt.scatter(JYears, twoPP, label='J2')
plt.plot(JYears, M2PPperyear, label='m2 per year in J years')
plt.legend()
plt.figure(3)
for rows in maxfieldpoints.index:
    if 'CHI' in rows:
        plt.scatter(JYears, maxfieldpoints.loc[rows], marker = '^', label=rows, color='m')
        plt.scatter(JYears, mjmaxfp, marker='*', label='MJmaxfieldpoints', color='r')

    #else:
        #plt.scatter(JYears, maxfieldpoints.loc[rows], label=rows)
plt.legend()
plt.show()