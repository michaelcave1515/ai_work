#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 07:46:29 2021

@author: mikecave
"""

import pandas as pd
import datetime as date
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import covidcast


hhs_url = 'https://healthdata.gov/sites/default/files/reported_hospital_capacity_admissions_facility_level_weekly_average_timeseries_20210103.csv'
c19_url= "https://raw.githubusercontent.com/youyanggu/covid19_projections/master/infection_estimates/latest_all_estimates_states.csv"
c19_url_usa = "https://raw.githubusercontent.com/youyanggu/covid19_projections/master/infection_estimates/latest_all_estimates_us.csv"

hhs_df = pd.read_csv(hhs_url)
df_states = pd.read_csv(c19_url)
df_usa = pd.read_csv(c19_url_usa)
state_facts = pd.read_csv('stateFacts.csv')

df_states = df_states.fillna(0)
df_states['date'] = pd.to_datetime(df_states['date'], format="%Y-%m-%d")
df_usa = df_usa.fillna(0)
df_usa['date'] = pd.to_datetime(df_usa['date'], format="%Y-%m-%d")

date = df_usa['date'].iloc[-1]
cases = df_usa['daily_positive_7day_ma'].iloc[-1]
tests = df_usa['daily_tests_7day_ma'].iloc[-1]
adj_pos_rate = df_usa['positivity_rate_7day_ma'].iloc[-1]

print(f'The report today is from {date:%Y-%m-%d} and the 7d MA is {cases:,} on {tests:,} tests \
with an adjusted positivity rate of {adj_pos_rate:.2%} data: covid19-projections.com')

df_states.to_csv('infectionEstimates.csv', index=False)
df_usa.to_csv('infectionEstimates_usa.csv', index=False)

deathsMax = df_usa['daily_deaths_7day_ma'].idxmax()
deathsMax = df_usa.iloc[deathsMax]['date']
print(deathsMax.strftime('%m-%d-%Y'))


infectionsMax = df_usa['new_infected_mean'].idxmax()
infectionsMax = df_usa.iloc[infectionsMax]['date']
print(infectionsMax.strftime('%m-%d-%Y'))

casesMax = df_usa['daily_positive_7day_ma'].idxmax()
casesMax = df_usa.iloc[casesMax]['date']
print(casesMax.strftime('%m-%d-%Y'))

testsMax = df_usa['daily_tests_7day_ma'].idxmax()
testsMax = df_usa.iloc[testsMax]['date']
print(testsMax.strftime('%m-%d-%Y'))

states_agg_max = df_states.loc[df_states.groupby('state').new_infected_mean.agg('idxmax')]

states_agg_max.to_csv('statesAggMax.csv', index=False)