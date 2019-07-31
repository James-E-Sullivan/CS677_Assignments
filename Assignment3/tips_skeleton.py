# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 02:37:06 2019

@author: AvantikaDG and Eugene Pinsky
"""
import os
import pandas as pd
import numpy as np

# change with a file name 
file_name = os.path.join(r'C:\Users','epinsky','bu','python','data_science_with_Python','datasets','tips.csv')
df = pd.read_csv(file_name)

df['tip_percent'] = 100.0 * df['tip']/df['total_bill']

# average tip for lunch and for dinner
average_tip_lunch = np.mean(df.tip[df.time == 'Lunch']/df.total_bill[df.time == 'Lunch'])
average_tip_dinner = np.mean(df.tip[df.time == 'Dinner']/df.total_bill[df.time == 'Dinner'])

if average_tip_lunch > average_tip_dinner:
    print("Tips are higher during lunch.")
elif average_tip_lunch < average_tip_dinner:
    print("Tips are higher during dinner.")
else:
    print("Tips are equal during lunch and dinner")
    
# times for highest tip 
day_time = df.groupby(['day', 'time']).mean()
print("\n",day_time.loc[day_time['tip_percent'] == max(day_time['tip_percent'])]['tip']) 
    
  
# correlation between meal prices and tips3
correlation_tips_vs_meal = df.corr(method='pearson')['tip_percent']['total_bill']
correlation_tips_vs_meal = round(correlation_tips_vs_meal , 4)
print('correlation between meal prices and tipes is: ', correlation_tips_vs_meal)
if correlation_tips_vs_meal > 0 :
    print('tips increase with higher bill amount. ')
elif correlation_tips_vs_meal < 0:
    print("tips decrease with higher bill amount. ")
else:
    print(' no relationship between tips and bill amount ')

# correlation between size of group and tips
correlation_tips_vs_group = df.corr(method='pearson')['tip_percent']['size']
correlation_tips_vs_group = round(correlation_tips_vs_group, 4)
print('\n correlation between tips and group size: ', correlation_tips_vs_group)
if correlation_tips_vs_group > 0 :
    print('tips increase for larger groups ')
elif correlation_tips_vs_group < 0:
    print("tips decrease for larger groups ")
else:
    print(' no relationship between tips and group size ')
   
 
# percent of people smoking
print("\n Percentage of people who are smoking is", round(100*len(df[df.smoker == "Yes"])/len(df),2), "%")

# correlation between tips and time
time = list(range(len(df)))
correlation_tips_vs_time = df['tip_percent'].corr(pd.Series(time), method='pearson')
correlation_tips_vs_time = round(correlation_tips_vs_time,4)
print('\n correlation between tips and time: ', correlation_tips_vs_time)
if correlation_tips_vs_time > 0 :
    print('tips increase with time ')
elif correlation_tips_vs_time < 0:
    print("tips decrease with time ")
else:
    print(' no relationship between tips and time ')

# correlations in tip amounts between smokers and non-smokers
mean_tip_smokers_df = df.groupby(['smoker']).mean()  
mean_tip_smoke_yes = mean_tip_smokers_df['tip_percent'][0]
mean_tip_smoke_no = mean_tip_smokers_df['tip_percent'][1]  

print('\naverage tip for non_smokers: ', round(mean_tip_smoke_no)) 
print('\average tip for smokers: ', round(mean_tip_smoke_yes)) 

if mean_tip_smoke_no > mean_tip_smoke_yes:
    print("Non-smokers pay more tips.")
elif mean_tip_smoke_no < mean_tip_smoke_yes:
    print("Smokers pay more tips.")
else:
    print("Smokers and non-smokers pay equal tips")




