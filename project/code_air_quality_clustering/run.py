import os

import preprocess as pp
import clustering as cl


def main():
    #the driver to run preprocessing and clustering 
    
    cd = os.getcwd()

    #locations which we have air quality data for
    aq_locations = ['bd','bh','ca','gt','hk','hu','id','in','iq','kw','lk','lu','mn','no',
                'np','pe','pk','ug','vm']

    #PREPROCESSING
    aq_array_stand = pp.run_preprocess(cd, aq_locations)
    #CLUSTERING
    cl.run_clustering(aq_array_stand)
    

main()
