import os
import pandas as pd
import numpy as np

#Code which combines all the separate air quality data into a master copy.
#We found using the individual files easier to handle in modeling.

def main():
    countries = ['bh','ca','gt','hk','hu','id','in','iq','kw','lk','lu','mn','no',
                'np','pe','pk','ug','vn']
    complete_data = pd.read_csv('data/bd_data.csv', header = 0, index_col = None).filter(['utc','value'], axis = 1)
    complete_data.rename(columns = {'utc': 'bh_utc', 'value': 'bh_value'})
    for country in countries:
        curr_country = pd.read_csv('data/'+country+'_data.csv', header = 0, index_col = None).filter(['utc','value'], axis = 1)
        curr_country.rename(columns = {'utc': country+'_utc', 'value': country+'_value'})
        complete_data = pd.concat([complete_data,curr_country], axis = 1)
    complete_data.to_csv('data/master_aq.csv', header = True, index = False)

print(os.getcwd())
main()
