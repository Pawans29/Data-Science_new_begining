#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import holoviews as hv
import plotly.graph_objects as go
import plotly.express as pex
hv.extension('bokeh', 'matplotlib')


# In[2]:


nz_migration = pd.read_csv("C:/Users/DELL/Desktop/Project/migration_nz.csv")
nz_migration.head(10)


# In[3]:


nz_migration.describe()


# In[4]:


nz_migration.isnull().sum()


# In[5]:


nz_migration.info()


# Steps that we undertake to clean the data
# 
# We'll remove entries other than arrival and departure.
# We'll remove entries where proper country name is not present.
# We'll then group dataframe by Measure & Country attributes and sum up all entries

# In[6]:


nz_migration = nz_migration[nz_migration["Measure"]!="Net"]
nz_migration = nz_migration[~nz_migration["Country"].isin(["Not stated", "All countries"])]
nz_migration_grouped = nz_migration.groupby(by=["Measure","Country"]).sum()[["Value"]]
nz_migration_grouped = nz_migration_grouped.reset_index()
nz_migration_grouped.head()


# In[7]:


nz_migration_grouped.Country.unique()


# #### Sankey Diagram of Population Migration between New Zealand & Various Continents 

# ##### we will choose just a few continents to represnt the migation movements, we will filter out the rest, we have taken 5 continents, Asia, Australia, Africa, Europe, America

# In[8]:


continents = ["Asia", "Australia","Africa","Europe", "America"]
continent_wise_migration = nz_migration_grouped[nz_migration_grouped.Country.isin(continents)]
continent_wise_migration


# In[9]:


### we will create a sankey diagram by passing the aboove dataframe, in holoviews. it needs three parameters 1- Source, 2- Destination 3- Property flow from source to destination


# In[19]:


sankey1 = hv.Sankey(continent_wise_migration, kdims=["Measure", "Country"], vdims=["Value"])

sankey1.opts(cmap='Colorblind',label_position='left',
                                 node_alpha=1.0, node_width=40, node_sort=True,
                                 width=800, height=600, bgcolor="white",
                                 title="Population Migration between New Zealand and Other Continents")


# In[ ]:





# In[ ]:





# In[ ]:




