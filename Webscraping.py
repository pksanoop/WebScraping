#!/usr/bin/env python
# coding: utf-8

# # Webscraping Project 

# ## Importing Libraries

# In[173]:


from bs4 import BeautifulSoup # For HTML parsing, helps to extract info by identifying the headers and attributes
import requests # Sends website connection requests
from time import sleep # To prevent overwhelming the server between connections
from collections import Counter # Keep track of our term counts
import pandas as pd # For converting results to a dataframe and bar chart plots
import json # For parsing json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from matplotlib import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')


# In[380]:


#This is the hybrid and electric vehicle page
page_url = 'https://www.carpages.ca/used-cars/search/?fueltype_id%5B0%5D=3&fueltype_id%5B1%5D=7'


# In[3]:


result = requests.get(page_url)


# ## Request Status

# In[4]:


result.status_code


# ## Creating the function to use in the loop

# In[2]:


def get_page(page):
    url = f'https://www.carpages.ca/used-cars/search/?num_results=50&fueltype_id%5B0%5D=3&fueltype_id%5B1%5D=7&p={page}'
    result = requests.get(url) #result.status_code to check if request went through
    soup = BeautifulSoup(result.content)
    return soup


# ## Tags on where to grab the information

# ## Name

# In[11]:


#Finds the name of vehicles
soup.find('h4', class_='hN').get_text(strip=True)


# ## Year

# In[12]:


soup.find('h4', class_='hN').get_text(strip=True).split(' ')[0]


# ## Make

# In[13]:


soup.find('h4', class_='hN').get_text(strip=True).split(' ')[1]


# ## Model

# In[14]:


soup.find('h4', class_='hN').get_text(strip=True).split(' ')[2:]


# ## Price

# In[15]:


#Finds the price of vehicles
(soup.find('div', class_='l-column l-column--medium-4 push-none').get_text(strip=True)).split('+')[0]


# ## Location

# In[16]:


#Finds the location of vehicles
soup.find('p', class_='hN').get_text(strip=True)


# # Mileage

# In[17]:


#Finds the mileage of vehicles
soup.find('div', class_='grey l-column l-column--small-6 l-column--medium-4').get_text(strip=True)


# ## Loop to grab all of the information from the front page and append to list. Also grab urls from href for individual car pages

# In[60]:


price_of_vehicles = []
name_of_vehicles = []
make_of_vehicles = []
model_of_vehicles = []
year_of_vehicles = []
location_of_vehicles = []
mileage_of_vehicles = []
links = []

for page in tqdm(range(1, 44)): #not inclusive of last range
    soup = get_page(page)
    sleep(2)
    
    for price in soup.find_all('div', class_='l-column l-column--medium-4 push-none'):
        try: 
            price_of_vehicles.append(price.get_text(strip=True).split('+')[0])
        except:
            price_of_vehicles.append('N/A')
        
    for name in soup.find_all('h4', class_='hN'):
        try: 
            links.append(name.a['href']) #Selects the 'a-tag' within 'h4' and selects the href (url)
        except:
            links.append('N/A')
        try:
            name_of_vehicles.append(name.get_text(strip=True))
        except:
            name_of_vehicles.append('N/A')
            
    for make in soup.find_all('h4', class_='hN'):
            try:
                make_of_vehicles.append(make.get_text(strip=True).split(' ')[1])
            except:
                make_of_vehicles.append('N/A')
            
    for year in soup.find_all('h4', class_='hN'):
            try:
                year_of_vehicles.append(year.get_text(strip=True).split(' ')[0])
            except:
                year_of_vehicles.append('N/A')
                
    for model in soup.find_all('h4', class_='hN'):
            try:
                model_of_vehicles.append(model.get_text(strip=True).split(' ')[2:])
            except:
                model_of_vehicles.append('N/A')
    
    for location in soup.find_all('p', class_='hN'):
        try:
            location_of_vehicles.append(location.get_text(strip=True))
        except:
            location_of_vehicles.append('N/A')
    
    for mileage in soup.find_all('div', class_='grey l-column l-column--small-6 l-column--medium-4'):
        try:
            mileage_of_vehicles.append(mileage.get_text(strip=True))
        except:
            mileage_of_vehicles.append('N/A')


# In[62]:


#Add carpages.ca to the beginning of the URLs grabbed from href
new_links = []

for link in links:
    new_links.append('https://www.carpages.ca' + link)


# ## Loop to grab the information from the individual car pages

# In[66]:


body_of_vehicles = []
fuel_of_vehicles = []
drive_of_vehicles = []

for url in tqdm(new_links):
    x = False
    y = False
    z = False
    result = requests.get(url) #result.status_code to check if request went through
    soup = BeautifulSoup(result.content)
    sleep(2)
#Code will run to find 'Body Style', when found, x = True and following code will run. If not, x = False and continues along the outer code. 
    for body in soup.find_all('li', class_='box--light-grey round'): 
        if body.span.get_text(strip=True) == 'Body Style':
            x = True
            style = list(body)[3].get_text(strip=True)
            body_of_vehicles.append(style)
    if not x:
            body_of_vehicles.append('N/A')
            
    for fuel in soup.find_all('li', class_='box--light-grey round'):
        if fuel.span.get_text(strip=True) == 'Fuel Type':
            y = True
            style = list(fuel)[3].get_text(strip=True)
            fuel_of_vehicles.append(style)
    if not y:
            fuel_of_vehicles.append('N/A')
            
    for drive in soup.find_all('li', class_='box--light-grey round'):
        if drive.span.get_text(strip=True) == 'Drive Type':
            z = True
            style = list(drive)[3].get_text(strip=True)
            drive_of_vehicles.append(style)
    if not z:
            drive_of_vehicles.append('N/A')


# ## Model

# In[382]:


model_of_vehicles


# In[68]:


#Gives us the model by combining the remaining elements together, look above for reference
model_vehicles = []

for cars in model_of_vehicles:
    bodies = ' '.join(cars)
    model_vehicles.append(bodies)
    print(bodies)


# ## City and Province

# In[69]:


city_of_vehicles = []
province_of_vehicles = []

for city in location_of_vehicles:
    city_name = city.split(',')[0]
    city_of_vehicles.append(city_name)

for province in location_of_vehicles:
    province_name = province.split(',')[1]
    province_of_vehicles.append(province_name)


# ## This prints colour of car

# In[376]:


print(mileage_of_vehicles[1::2])


# ## This prints mileage of car

# In[48]:


print(mileage_of_vehicles[0::2])


# ## Creating the main dataframe

# In[359]:


carpages = pd.DataFrame({'price': price_of_vehicles, 'year': year_of_vehicles, 'make': make_of_vehicles, 'model': model_vehicles, 'city': city_of_vehicles,
                        'province': province_of_vehicles, 'mileage(KM)': mileage_of_vehicles[0::2], 'colour': mileage_of_vehicles[1::2], 'body': body_of_vehicles, 'fuel': fuel_of_vehicles, 'drive': drive_of_vehicles})

carpages


# ## Data Cleaning

# In[360]:


#Removes all rows with 'N/A' in the body column
carpages[carpages['body'] != 'N/A']

#Removes all rows with 'N/A' in the drive column
carpages[carpages['drive'] != 'N/A']

#Replace all empty values in 'colour' column with 'N/A'
carpages['colour'] = carpages['colour'].replace('', 'N/A')

#Replace all Gray values in 'colour' column with 'N/A'
carpages['colour'] = carpages['colour'].replace('Gray', 'Grey')


# In[361]:


#Removes $ from the price
carpages['price'] = carpages['price'].replace({'\$':''}, regex=True).replace('\W', '', regex=True)
carpages


#Removes KM from mileage
carpages['mileage(KM)'] = carpages['mileage(KM)'].str[:-2].replace('\W', '', regex=True)


# In[362]:


#Replace all 'CALL' from price to Not-a-Number (NaN)
carpages['price'] = carpages['price'].replace('CALL', np.nan)
carpages

#Replace all empty values from price to Not-a-Number (NaN)
carpages['price'] = carpages['price'].replace('', np.nan)
carpages

#Makes the price an integer
carpages['price']=pd.to_numeric(carpages['price'])
carpages

#Makes the year an integer
carpages['year'] = pd.to_numeric(carpages['year'])

#Replace all 'CA' from mileage to Not-a-Number (NaN)
carpages['mileage(KM)'] = carpages['mileage(KM)'].replace('CA', np.nan)
carpages

#Convert mileage to an number
carpages['mileage(KM)']=pd.to_numeric(carpages['mileage(KM)'])


# ## Save to CSV

# In[97]:


carpages.to_csv('carpagesv2.csv')


# ## Analysis

# In[338]:


carpages.info()


# In[339]:


#Show the distribution of numeric columns (price and mileage)
carpages.describe()


# In[101]:


#Show the no. of unique values in each column
carpages.nunique()


# ## Top 10 highest priced vehicles

# In[367]:


#Sort the data frame by the highest priced vehicles
carpages_price_sorted = carpages.sort_values(by='price', ascending = False)

#Remove all vehicles that are outliers (top 16 vehicles)
carpages_price_fixed = carpages_price_sorted.drop(carpages_price_sorted.index[:16])


# In[369]:


#Calculates top 10 vehicles by price
top_10_price = carpages_price_fixed.head(10)

top_10_price


# # Visualization

# In[299]:


plt.figure(figsize = (10,10))
sns.heatmap(carpages_price_fixed.corr(), annot = True);


# ## Which make is the most popular

# In[316]:


plt.figure(figsize = (15, 10))
carpages_price_fixed['make'].value_counts().plot(kind='bar', title = 'Make Popularity', color = ['red','blue','green','yellow','orange'])
plt.title('Make Popularity', fontsize = 20)
plt.xlabel('Make', fontsize=20);
plt.ylabel('Frequency Count', fontsize=20);


# ## Top 10 colours

# In[384]:


carpages_color_fixed = carpages[carpages['colour'] != 'N/A']


# In[385]:


top_10_colours = pd.DataFrame(carpages_color_fixed['colour'].value_counts(ascending = False).head(10))


# In[386]:


plt.figure(figsize = (15, 15))
top_10_colours.plot(kind="bar")
plt.title('Colour Popularity', fontsize = 15)
plt.xlabel('Colour', fontsize=15);
plt.ylabel('Frequency Count', fontsize=15);


# ## Calculates the average price of each make

# In[226]:


ax=carpages_price_fixed.groupby(['make'])['price'].mean().sort_values(ascending=False)
plt.figure(figsize=(20, 20))
ax.plot.bar()
plt.title('Make vs Average Price')
plt.xlabel('Make')
plt.ylabel('Price')
plt.show()


# ## Calculates the average price of the drive type

# In[220]:


plt.figure(figsize=(15,15))
sns.barplot(x='drive', y='price', data=carpages_price_fixed[carpages_price_fixed['drive'] != 'N/A'], ci=None)
plt.title('Drive vs Average Price')
plt.xlabel('Drive')
plt.ylabel('Price');


# ## Subplots

# In[134]:


plt.figure(figsize=(30, 30))
sns.set(font_scale=2)
plt.subplot(3,2,1)
sns.boxplot(x = 'make', y = 'price', data = carpages_price_fixed[carpages_price_fixed['make'] != 'N/A'])
plt.xticks(rotation=90)
plt.subplot(3,2,2)
sns.boxplot(x = 'fuel', y = 'price', data = carpages_price_fixed[carpages_price_fixed['fuel'] != 'N/A'])
plt.subplot(3,2,3)
sns.boxplot(x = 'drive', y = 'price', data = carpages_price_fixed[carpages_price_fixed['drive'] != 'N/A'])
plt.subplot(3,2,4)
sns.boxplot(x = 'body', y = 'price', data = carpages_price_fixed[carpages_price_fixed['body'] != 'N/A'])
plt.xticks(rotation=90)
plt.subplot(3,2,5)
sns.boxplot(x = 'year', y = 'price', data = carpages_price_fixed);
plt.xticks(rotation=90);


# ## Compare the price between electric and hybrid type cars given the vehicle year

# In[321]:


sns.catplot(x = 'year', y='price', hue = 'fuel', data = carpages_price_fixed, kind = 'bar', ci = False)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Year vs Price')
plt.xticks(rotation=90);


# ## Shows the price of vehicles based on their mileage

# In[327]:


sns.set(rc={'figure.figsize':(15, 15)})
sns.relplot(x = 'mileage(KM)', y = 'price', data = carpages_price_fixed, kind = 'scatter', ci = False)
plt.xlabel('Mileage(KM)')
plt.ylabel('Price')
plt.title('Price based on mileage');


# In[149]:


sns.pairplot(carpages_price_fixed, hue = 'fuel');


# In[225]:


ax=carpages_price_fixed.groupby(['body'])['price'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 10))
ax.plot.bar()
plt.title('Car Body vs Average Price')
plt.xlabel('Car Body')
plt.ylabel('Price')
plt.show()

