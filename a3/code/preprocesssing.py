import os
from datetime import datetime
import pandas as pd
import numpy as np

#Move to the data directory to be able to access and write data.
os.chdir('../data')

#Define a date to use in the "recency" computation
r_date = datetime.strptime('2011-12-12 08:30', '%Y-%m-%d %H:%M')

#Changes the InvoiceDate column to datetime objects
def format_time(time):
    return datetime.strptime(time,'%Y-%m-%d %H:%M')

def get_f(customer):
    days = 0
    for i in range(1,len(customer.InvoiceDate)):
        curr = customer.InvoiceDate.iloc[i]
        prev = customer.InvoiceDate.iloc[i-1]
        days += (curr-prev).days
    days /= len(customer.InvoiceDate.unique())
    return days

def get_rfm(customer):
    #customer is a sub-dataframe containing the data for a unique customer.
    #In this function we return the RFM marketing metrics.
    # - Recency, Frequency and Monetary
    #As well, we get the number of returns a customer has.
    returned_orders = [] #Get InvoiceNo's which indicate a return.
    rts = 0 #Count the number of returns a customer makes
    for invoice in customer.InvoiceNo.unique():
        if ('C' in invoice):
            returned_orders.append(invoice)
            rts += 1
    returns = customer[customer.InvoiceNo.isin(returned_orders)]
    customer = customer[~customer.InvoiceNo.isin(returned_orders)]
    if(customer.size != 0): #Check if a customer has only ever returned and not ever bought.
        r = (r_date - max(customer.InvoiceDate)).days
        # if ((max(customer.InvoiceDate)-min(customer.InvoiceDate)).days != 0):
        #     f = (max(customer.InvoiceDate)-min(customer.InvoiceDate)).days/len(customer.InvoiceNo.unique())
        # else:
        #     f = 0
        f = get_f(customer)
        m = customer.UnitPrice.sum()
        return [r,f,m,rts,customer.CustomerID.unique()[0],customer.Country.unique()[0]]
    else: #If a customer has never made a purchase then we return 0 for all their rfm metrics.
        return [0,0,0,0,0,0]

def main():
    online_retail = pd.read_csv('online_retail.csv', header = 0, index_col = None)
    online_retail.InvoiceDate = online_retail.apply(lambda x: format_time(x.InvoiceDate), axis = 1)
    records = []
    customers = online_retail.CustomerID.unique()
    for cid in customers:
        records.append(get_rfm(online_retail[online_retail.CustomerID == cid]))
        if(records[-1] == [0,0,0,0,0,0]):
            online_retail = online_retail[~(online_retail.CustomerID == cid)]
    rfm = pd.DataFrame(data = records, columns = ['Recency','Frequency','Monetary','Returns','CustomerID','Country'])
    rfm.to_csv('rfm.csv', header = True, index = False)
    online_retail.to_csv('no_returns.csv', header = True, index = False)

def get_text(customer):
    if (customer.size != 0):
        products = ''
        for product in customer.Description:
            products += ' '+product
        return [products,customer.CustomerID.unique()[0],customer.Country.unique()[0]]
    else:
        return ['Empty',0,0]

def main_bonus():
    online_retail = pd.read_csv('online_retail.csv', header = 0, index_col = None)
    # online_retail.InvoiceDate = online_retail.apply(lambda x: format_time(x.InvoiceDate), axis = 1)
    records = []
    customers = online_retail.CustomerID.unique()
    for cid in customers:
        records.append(get_text(online_retail[online_retail.CustomerID == cid]))
    purchases = pd.DataFrame(data = records, columns = ['Description','CustomerID','Country'])
    purchases.to_csv('purchases.csv', header = True, index = False)

main_bonus()
