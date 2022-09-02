import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def select_stat(dataframe, field, perc_vector, limityear):
    return dataframe[(dataframe.index > limityear[0]) & (dataframe.index < limityear[1])][field].describe(percentiles=perc_vector)
def topplayers(dataframe, field, perc, limityear):
    shortdata = dataframe[(dataframe.index > limityear[0]-1) & (dataframe.index < limityear[-1])]
    #lim0 = shortdata[field[0]].groupby(level='Year').describe(percentiles=[perc])['99%']
    lim1 = shortdata[field[-2]].groupby(level='Year').describe(percentiles=[perc])['99%']
    lim2 = shortdata[field[-1]].groupby(level='Year').describe(percentiles=[perc])['99%']
    players = {}
    for year in np.arange(start=limityear[0], stop=limityear[-1], step=1):
        #print('Limit for {} in {} is {}\n'.format(field[0], year, lim0.loc[year]))
        print('Limit for {} in {} is {}\n'.format(field[1], year, lim1.loc[year]))
        print('Limit for {} in {} is {}\n'.format(field[2], year, lim2.loc[year]))
        #s0 = pd.Series(shortdata[field[0]] > lim0.loc[year])
        s1 = pd.Series(shortdata[field[-2]] > lim1.loc[year])
        s2 = pd.Series(shortdata[field[-1]] > lim2.loc[year])
        players[year] = shortdata[(s1) & (s2) & (shortdata.index == year)]['Player']
        if not(len(players[year])):
            del players[year]
    return players
def select_color(colors, counter):
    if counter > len(colors)-1:
        counter = -1
    color = colors[counter]
    return color, counter
def excel_exporter(index,datas,names,path):
    counter=-1
    for series in datas:
        counter+=1
        if not counter:
            data_to_save = pd.Series(series, name=names[0], index=index)
        else:
            data_to_save = pd.concat([data_to_save, pd.Series(series, name=names[counter], index=index)], axis=1)
    data_to_save.to_excel(path)
    return 1

df = pd.read_excel('NBA Stats.xlsx')
df.drop('Column1', axis=1, inplace=True)
df.dropna(subset=['Year'], inplace=True)
df.set_index('Year', inplace=True)
JYears = df.index.where(df['Player'] == 'Michael Jordan*').dropna()
SPYears = df.index.where(df['Player'] == 'Scottie Pippen*').dropna()
DRYears = df.index.where(df['Player'] == 'Dennis Rodman*').dropna()

threePP = df['3P'].where(df['Player'] == 'Michael Jordan*').dropna()
twoPP = df['2P'].where(df['Player'] == 'Michael Jordan*').dropna()
PTSPP = df['PTS'].where(df['Player'] == 'Michael Jordan*').dropna()
eJ = df['PER'].where(df['Player'] == 'Michael Jordan*').dropna()

SPthreePP = df['3P'].where(df['Player'] == 'Scottie Pippen*').dropna()
SPtwoPP = df['2P'].where(df['Player'] == 'Scottie Pippen*').dropna()
SPPTSPP = df['PTS'].where(df['Player'] == 'Scottie Pippen*').dropna()
eSP = df['PER'].where(df['Player'] == 'Scottie Pippen*').dropna()

DRthreePP = df['3P'].where(df['Player'] == 'Dennis Rodman*').dropna()
DRtwoPP = df['2P'].where(df['Player'] == 'Dennis Rodman*').dropna()
DRPTSPP = df['PTS'].where(df['Player'] == 'Dennis Rodman*').dropna()
eDR = df['PER'].where(df['Player'] == 'Dennis Rodman*').dropna()

MTPPperyear = []
M2PPperyear = []
MPTSPPperyear =[]
emeanperyear = []
emeanperyear97 = []
emeanperyear99 = []
cols = ['3P', '2P', 'PTS']
limityears = [JYears[0], JYears[-1]]
#statistics = []
#for fields in cols:
    #statistics.append(select_stat(df, fields, [0.9, 0.95, 0.97], limityears))
PERteam = df.groupby(['Tm', 'Year'])['PER'].apply(lambda grp: grp.nlargest(10).sum())
for year in JYears:
    MTPPperyear.append(df['3P'].where(df.index == year).mean())
    M2PPperyear.append(df['2P'].where(df.index == year).mean())
    MPTSPPperyear.append(df['PTS'].where(df.index == year).mean())
    a, b, c = df['PER'].where(df.index == year).describe([0.95, 0.97, 0.995])[['95%', '97%', '99.5%']]
    emeanperyear.append(a)
    emeanperyear97.append(b)
    emeanperyear99.append(c)
    if year == JYears[0]:
        maxfieldpoints = pd.DataFrame(df[df.index == year].groupby('Tm')['PTS'].sum())
        df_describe = df[(df.index == year)][cols].describe([0.9, 0.95, 0.97])
        statsPTS = pd.Series(df_describe['PTS'], name=year)
        statsp2 = pd.Series(df_describe['2P'], name=year)
        statsp3 = pd.Series(df_describe['3P'], name=year)
    else:
        df_describe = df[(df.index == year)][cols].describe([0.9, 0.95, 0.97])
        statsPTS = pd.concat([statsPTS, pd.Series(df_describe['PTS'], name=year)], axis=1)
        statsp2 = pd.concat([statsp2, pd.Series(df_describe['2P'], name=year)], axis=1)
        statsp3 = pd.concat([statsp3, pd.Series(df_describe['3P'], name=year)], axis=1)
        maxfieldpoints = pd.merge(maxfieldpoints, pd.Series(df[df.index == year].groupby('Tm')['PTS'].sum(), name=year),
                                  how='outer',
                                  right_index=True, left_index=True)
Best3bulls = pd.merge(PTSPP,SPPTSPP, left_index=True, right_index=True)
Best3bulls = pd.merge(Best3bulls, DRPTSPP, left_index=True, right_index=True).sum()

PERBest3bulls = pd.merge(eJ,eSP, left_index=True, right_index=True)
PERBest3bulls = pd.merge(PERBest3bulls, eDR, left_index=True, right_index=True).sum(axis=1)
maxfieldpoints.rename({'PTS': JYears[0]}, axis=1)
print('Top {} percentile of both 2P and 3P and PTS players in {} years are: {}'.format(97, JYears[:], topplayers(df, cols, 0.99,
                                                                                                        limityears).values()))
###################################PLOT#################################################################################
MTPP09=statsp3.loc['90%']
M2PP09=statsp2.loc['90%']
MTPP095=statsp3.loc['95%']
M2PP095=statsp2.loc['95%']
MTPP097=statsp3.loc['97%']
M2PP097=statsp2.loc['97%']
MTPP1=statsp3.loc['max']
M2PP1=statsp2.loc['max']
MPTSPP09=statsPTS.loc['90%']
MPTSPP095=statsPTS.loc['95%']
MPTSPP097=statsPTS.loc['97%']
MPTSPP1=statsPTS.loc['max']
excel_exporter(JYears,[threePP,MTPPperyear,MTPP09,MTPP095,MTPP097,MTPP1],
               ['threePP','MTPPperyear','MTPP09','MTPP095','MTPP097','MTPP1'], 'Export 3P data.xlsx')
excel_exporter(JYears,[twoPP,M2PPperyear,M2PP09,M2PP095,M2PP097,M2PP1],
               ['twoPP','M2PPperyear','M2PP09','M2PP095','M2PP097','M2PP1'], 'Export 2P data.xlsx')
excel_exporter(JYears,[PTSPP,MPTSPPperyear,MPTSPP09,MPTSPP095,MPTSPP097,MPTSPP1],
               ['PTSPP','MPTSPPperyear','MPTSPP09','MPTSPP095','MPTSPP097','MPTSPP1'], 'Export PTS data.xlsx')
excel_exporter(JYears,[eJ,emeanperyear,emeanperyear97,emeanperyear99],
               ['eJ','emeanperyear95','emeanperyear97','emeanperyear99'], 'Export PER data.xlsx')
plt.figure(1)
plt.scatter(JYears, threePP, marker='*', label='J3P')
plt.plot(JYears, MTPPperyear, label='mean 3P per year in J years')
plt.plot(JYears, MTPP09, label='3P 0.9 percentile in J years')
plt.plot(JYears, MTPP095, label='3P 0.95 percentile in J years')
plt.plot(JYears, MTPP097, label='3P 0.97 percentile in J years')
plt.plot(JYears, MTPP1, label='3P max in J years')
plt.legend()
plt.xlabel('J years')
plt.ylabel('3P')
plt.figure(2)
plt.scatter(JYears, twoPP, marker='*',label='J2P')
plt.plot(JYears, M2PPperyear, label='mean 2P per year in J years')
plt.plot(JYears, M2PP09, label='2P 0.9 percentile in J years')
plt.plot(JYears, M2PP095, label='2P 0.95 percentile in J years')
plt.plot(JYears, M2PP097, label='2P 0.97 percentile in J years')
plt.plot(JYears, M2PP1, label='2P max in J years')
plt.title('2P')
plt.ylabel('2P')
plt.xlabel('J years')
plt.legend()
plt.figure(3)
plt.scatter(JYears, PTSPP, marker='*', label='JPTS')
plt.scatter(SPYears, SPPTSPP, marker='*', label='SPPTS')
plt.scatter(DRYears, DRPTSPP, marker='*', label='DRPTS')
plt.plot(JYears, MPTSPPperyear, label='mean PTS per year in J years')
plt.plot(JYears, MPTSPP09, label='PTS 0.9 percentile in J years')
plt.plot(JYears, MPTSPP095, label='PTS 0.95 percentile in J years')
plt.plot(JYears, MPTSPP097, label='PTS 0.97 percentile in J years')
plt.plot(JYears, MPTSPP1, label='PTS max in J years')
plt.title('PTS')
plt.ylabel('PTS')
plt.xlabel('J years')
plt.legend()
plt.figure(4)
#for rows in maxfieldpoints.index:
    #if 'CHI' in rows:
        #plt.scatter(JYears, maxfieldpoints.loc[rows], marker='^', label=rows, color='m')
        #plt.scatter(JYears, PTSPP, marker='*', label='MJmaxfieldpoints', color='r')
        #plt.scatter(Best3bulls.index, Best3bulls, marker='*', label='B3maxfieldpoints', color='k')
    # else:
        # plt.scatter(JYears, maxfieldpoints.loc[rows], label=rows)
plt.plot(JYears, emeanperyear, color='k', label='League 95%')
plt.plot(JYears, emeanperyear97, color='g', label='League 97%')
plt.plot(JYears, emeanperyear99, 'r.-', label='League 99.5%')
plt.scatter(eJ.index, eJ, color='r', label='J')
plt.scatter(eSP.index, eSP, color='b', label='SP')
plt.scatter(eDR.index, eDR, color='m', label='DR')
plt.title('PER')
plt.xlabel('J years')
plt.ylabel('PER per player')
plt.legend()
plt.figure(5)
count = -1
teamPER = []
for team in PERteam.index.get_level_values(0).unique():
    holderPER = []
    for year in JYears:
        if len(PERteam.loc[team][PERteam.loc[team].index==year]):
            holderPER.append(PERteam.loc[team][PERteam.loc[team].index==year].iloc[0])
    if not(len(teamPER)) and len(PERteam.loc[team][PERteam.loc[team].index==year]):
        teamPER = pd.Series(holderPER, name=team)
    elif len(PERteam.loc[team][PERteam.loc[team].index==year]):
        teamPER = pd.concat([teamPER, pd.Series(holderPER, name=team)], axis=1)
print(teamPER)
for team in teamPER.columns:
    count += 1
    color, count = select_color(['r', 'b', 'k', 'g', 'm', 'y'], count)
    if team == 'CHI':
        plt.plot(teamPER.index, teamPER[team], marker='*', color=color, label=team)
    else:
        plt.scatter(teamPER.index, teamPER[team], color=color, label=team)
plt.title('PER per team')
plt.xlabel('Years form {}'.format(int(JYears[0])))
plt.ylabel('PER per top 10 players')
plt.legend(loc=8, ncol=5)
teamPER.to_excel('Export PER team data.xlsx')
plt.show()
