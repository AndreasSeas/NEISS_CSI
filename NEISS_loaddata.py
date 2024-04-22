#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 04:34:40 2023

@author: as822

m_neiss
"""

# =============================================================================
# load packages
# =============================================================================
from datetime import datetime
import csv

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib
import os, sys
import seaborn as sns
# sys.exit()
# =============================================================================
# load data
# =============================================================================
homedir=os.getcwd();
os.chdir(homedir+'/NEISS_Data')
filenames=os.listdir()
filenames.sort()
filenames.remove('.DS_Store')
years=[int(s.split(".")[0][-4:]) for s in filenames]

'''
A cervical spine injury was defined as any neck injury (code 89) with the 
following injury designations: 
    - fracture (code 57), 
    - dislocation (55), 
    - nerve damage (61). 

In this manner we were able to isolate all potential cervical spine injuries 
as well as any potential associated instances of neurological compromise. 
'''

bdpt_filt=[89];
diag_filt=[57, 55, 61];

df=pd.DataFrame();# empty df
for i,file in enumerate(filenames):
    print(file)
    df_temp=pd.read_table(file,encoding_errors='replace',low_memory=False)
    # filter based on bdpt
    bdpt1=df_temp.Body_Part.isin(bdpt_filt)
    bdpt2=df_temp.Body_Part.isin(bdpt_filt)
    bdpt_i=bdpt1 | bdpt2
    
    # filter based on diag
    diag1=df_temp.Diagnosis.isin(diag_filt)
    diag2=df_temp.Diagnosis_2.isin(diag_filt)
    diag_i=diag1 | diag2
    
    overlap_i=bdpt_i & diag_i

    df_temp=df_temp.loc[overlap_i,:]

    df_temp.insert(1,'year',years[i])
    df=pd.concat([df,df_temp]);

os.chdir(homedir)

df=df.reset_index()
df.to_csv('NEISS_collated.csv')
df_old=df;
