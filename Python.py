#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Functions are built for every specific purpose, and finally one function is built to call all of them for getting data for a
#given year


# In[1]:


import pandas as pd
import numpy as np
import requests
from zipfile import ZipFile
from bs4 import BeautifulSoup
from pandas import Timestamp


# In[2]:


pd.set_option('display.max_columns', None)


# # 1. Creating 'Download' folder and downloading files into it

# In[3]:


import os
c = os.getcwd()   #Get current working directory
c


# In[4]:


os.mkdir('Download')


# In[5]:


cwd = c + '\\Download'
cwd


# In[6]:


os.chdir(cwd)


# # 2. Downloading the zip files

# In[ ]:


##Function for getting required urls for renaming files.


# In[7]:


def get_url_for_download_and_rename(year, yr):
    index = 6 + yr.index(year)
    url = 'https://www.data.bsee.gov/Main/OGOR-A.aspx'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')
    first = soup.find('table', {'id':'ContentPlaceHolderBody_ASPxGridView1_DXMainTable', 'class':'dxgvTable'})

    req = first.find_all('tr') 

    first_td= req[index].find_all('td', class_ = 'dxgv')[0]
    l = first_td.find('a', class_="dxeHyperlink")
    l = l.get('href')

    second_td = req[index].find_all('td', class_ = 'dxgv')[1]
    a = second_td.text
    #print(a)
    a1 = a.split(" ")[0]
    a11 = a1.split("/")

    if len(a11[0]) == 1:
        a11[0] = '0' + a11[0]
    if len(a11[1]) == 1:
        a11[1] = '0' + a11[1]
    af = a11[2]+ a11[0] + a11[1]

    print(af)

    #os.rename('ogora' + str(year) + 'delimit.txt','ogora' + 'year' + '_' + af + '.txt')
    third_td = req[index].find_all('td', class_ = 'dxgv')[3]
    l1 = third_td.find('a', class_="dxeHyperlink")
    l1 = l1.get('href')
    
    urld = 'https://www.data.bsee.gov/' + l1
    
    urln = 'https://www.data.bsee.gov/' + l
    
    print(urld)
    print(urln)
    
    return af, urln, urld


# In[73]:


#af, urln, urld = get_url_for_download_and_rename(year)


# In[ ]:


##Function for downloading zip files


# In[8]:


def download_dataset(url):
    downloadUrl = url

    req = requests.get(downloadUrl)
    filename = req.url[downloadUrl.rfind('/')+1:]

    with open(filename, 'wb') as f:
        for chunk in req.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    return filename


# In[74]:


#filename = download_dataset(urld)


# # 3. Unzipping the files and processing

# In[ ]:


##Function for extracting zip files and then removing it


# In[9]:


def extract_zip_remove_zip(filename):
    
    with ZipFile(filename, 'r') as zip_object:
        zip_object.extractall()
    
    os.remove(filename)
    return


# In[75]:


#extract_zip_remove_zip(filename)


# In[76]:


#os.rename('ogora' + str(year) + 'delimit.txt','ogora' + str(year) + '_' + af + '.txt')


# In[ ]:


##Function for obtaining the list of names of columns


# In[10]:


def column_name_for_each_yr(urln):
    page = requests.get(urln)
    soup = BeautifulSoup(page.content, 'html.parser')
    t = soup.find('table')

    lc = []

    ln = t.find_all('tr')
    for i in range(1, len(ln)):
        if i != 11:
            lc.append(ln[i].find_all('td')[3].text.split("-")[0])
        else: 
            lc.append(ln[i].find_all('td')[3].find('b').text.split("-")[0])

    for i in range(len(lc)):
        lc[i] = lc[i].rstrip()
        
    return lc


# In[35]:


#lc = column_name_for_each_yr(urln)
#lc


# In[ ]:


##Function for creating the dataframe by importing the downloaded/extracted file and performing required operations


# In[11]:


def ops_df(l, year, af):
    df = pd.read_csv('ogora' + str(year) + '_' + str(af) + '.txt', sep=",", header=None)
    df.columns = l
    df = df[[l[0], l[2], l[4], l[5], l[6], l[11]]]
    return df


# In[78]:


#ops_df(lc, year)


# In[ ]:


##Function for calling all the required/previously described functions


# In[12]:


def final_function(year):
    yr = []
    for i in range(27):
        yr.append(2022 - i)
    index = 6 + yr.index(year)
    af, urln, urld = get_url_for_download_and_rename(year, yr)
    filename = download_dataset(urld)
    extract_zip_remove_zip(filename)
    print(index)
    print(year)
    print(cwd)
    os.rename('ogora' + str(year) + 'delimit.txt','ogora' + str(year) + '_' + af + '.txt')
    lc = column_name_for_each_yr(urln)
    df = ops_df(lc, year, af) 
    return df


# In[13]:


df1= final_function(2019)
df1


# In[14]:


df2 = final_function(2020)
df2


# In[15]:


df = pd.concat([df1, df2], axis = 0)
df


# In[16]:


df.to_csv('Production.csv', header=True, index=False, sep ='|', line_terminator='\r\n')


# # Basic Analysis

# In[17]:


df[df['LEASE_NUMBER'].isna()]    #NaN lease rows


# In[ ]:


#df = df.dropna(subset = 'LEASE_NUMBER')   #Depending upon requirements NaN Lease can be dropped using this formula.
                                           #commented out since it has cond=siderable production vol


# In[18]:


ch = df.groupby(['LEASE_NUMBER', 'PRODUCTION_DATE', 'PRODUCT_CODE'])['MON_O_PROD_VOL', 'MON_G_PROD_VOL'].sum()
ch


# # 4. Only download if the file is updated 

# In[20]:


cwd


# In[ ]:


##Function for comparing timestamp of downloaded file and that available on url


# In[19]:


def updated_get_url_for_download_and_rename(year, yr):
    index = 6 + yr.index(year)
    url = 'https://www.data.bsee.gov/Main/OGOR-A.aspx'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')
    first = soup.find('table', {'id':'ContentPlaceHolderBody_ASPxGridView1_DXMainTable', 'class':'dxgvTable'})

    req = first.find_all('tr') 

    second_td = req[index].find_all('td', class_ = 'dxgv')[1]
    a = second_td.text
    a1 = a.split(" ")[0]
    a11 = a1.split("/")

    if len(a11[0]) == 1:
        a11[0] = '0' + a11[0]
    if len(a11[1]) == 1:
        a11[1] = '0' + a11[1]
    af = a11[2]+a11[0]+a11[1]
    af = int(af)
    print(af)
    
    FOLDER_PATH = cwd
    
    l_fn = []

    def listDir(dir):
        fileNames = os.listdir(dir)
        for fileName in fileNames:
            print('filename:', fileName)
            if fileName.split(".")[-1] == 'txt':
                l_fn.append(fileName)
    
    if __name__ == '__main__':
        listDir(FOLDER_PATH)
    
    res = []

    for i in range(len(l_fn)):
        x = l_fn[i].split("_")[0]
        num = ''
        for c in x:
            if c.isdigit():
                num = num + c
        num = int(num)
        res.append(num)
        
        
    nindex = res.index(year)
    j = l_fn[nindex].split(".")[0]
    k = j.split("_")[1]
    k = int(k)
    print(k)
    
    first_td= req[index].find_all('td', class_ = 'dxgv')[0]
    l = first_td.find('a', class_="dxeHyperlink")
    l = l.get('href')

    #os.rename('ogora' + str(year) + 'delimit.txt','ogora' + 'year' + '_' + af + '.txt')
    third_td = req[index].find_all('td', class_ = 'dxgv')[3]
    l1 = third_td.find('a', class_="dxeHyperlink")
    l1 = l1.get('href')
    
    urld = 'https://www.data.bsee.gov/' + l1
    
    urln = 'https://www.data.bsee.gov/' + l
    
    print(urld)
    print(urln)
    
    return af, k, urln, urld


# In[ ]:


##Function for getting the updated dataframe if required, if more recent data is present for given year in the url,
#the most recent data will be downloaded, other wise it won't do anything


# In[20]:


def updated_final_function(year):
    yr = []
    for i in range(27):
        yr.append(2022 - i)
    index = 6 + yr.index(year)
    af, k, urln, urld= updated_get_url_for_download_and_rename(year, yr)
    if af > k:
        filename = download_dataset(urld)
        extract_zip_remove_zip(filename)
        print(index)
        print(year)
        print(cwd)
        os.rename('ogora' + str(year) + 'delimit.txt','ogora' + str(year) + '_' + str(af) + '.txt')
        lc = column_name_for_each_yr(urln)
        df = ops_df(lc, year, af)
    else:
        df = pd.DataFrame()
        print('Recent Data already downloaded')
    return df


# In[21]:


df_u = updated_final_function(2020)


# In[22]:


df_u


# In[23]:


df_v = updated_final_function(2019)


# In[24]:


df_v


# In[25]:


#Finally, use concat function to club dataframes together and export it as csv


# In[26]:


df = pd.concat([df1, df2], axis = 0)


# In[27]:


df.to_csv('Production.csv', header=True, index=False, sep ='|', line_terminator='\r\n')


# In[ ]:




