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
df=pd.read_csv('NEISS_collated_final.csv')
savefigs=False
savetabs=False

# =============================================================================
# clean columns
# =============================================================================
os.chdir(homedir)
## treatment date
# convert to datetime
df.Treatment_Date=pd.to_datetime(df.Treatment_Date);

## age
# identify and rename nans
df.loc[df.Age==0,'Age']=np.nan
# convert months to years
df.loc[df.Age>200,'Age']=(df.loc[df.Age>200,'Age']-200)/12;

## sex
df['Sex']=df['Sex'].replace({0:np.nan,
                             1:'Male',
                             2:'Female',
                             3:'Non-Binary/Other'})

## race
df['Race']=df['Race'].replace({0:np.nan,
                             1:'White',
                             2:'Black/African American',
                             3:'Other',
                             4:'Asian',
                             5:'American Indian/Alaska Native',
                             6:'Native Hawaiian/Pacific Islander'})

# refactor other race as well, some people not included here
df.loc[df["Other_Race"]=='ASIAN','Race']='Asian'
df.loc[df["Other_Race"]=='KOREAN','Race']='Asian'
df.loc[df["Other_Race"]=='NATIVE AMERICAN','Race']='American Indian/Alaska Native'
df.loc[df["Other_Race"]=='AMERICAN INDIAN','Race']='American Indian/Alaska Native'
df.loc[df["Other_Race"]=='BLACK/AA','Race']='Black/African American'

## hispanic
df['Hispanic']=df['Hispanic'].replace({0:np.nan,
                             1:'Hispanic',
                             2:'Not Hispanic'})
df.loc[df["Other_Race"]=='HISPANIC','Hispanic']='Hispanic'
df.loc[df["Other_Race"]=='HISP','Hispanic']='Hispanic'

## Body_Part
df['Body_Part']=df['Body_Part'].replace({0: 'Internal ', 30: 'Shoulder ', 31: 'Upper Trunk', 32: 'Elbow ', 33: 'Lower Arm ', 34: 'Wrist ', 35: 'Knee ', 36: 'Lower Leg ', 37: 'Ankle ', 38: 'Pubic Region ', 75: 'Head ', 76: 'Face ', 77: 'Eyeball ', 79: 'Lower Trunk', 80: 'Upper Arm', 81: 'Upper Leg ', 82: 'Hand ', 83: 'Foot ', 84: '25-50% of Body ', 85: 'All Parts Body ', 87: 'Not Stated/Unk ', 88: 'Mouth ', 89: 'Neck ', 92: 'Finger ', 93: 'Toe', 94: 'Ear '});

df['Body_Part_2']=df['Body_Part_2'].replace({0: 'Internal ', 30: 'Shoulder ', 31: 'Upper Trunk', 32: 'Elbow ', 33: 'Lower Arm ', 34: 'Wrist ', 35: 'Knee ', 36: 'Lower Leg ', 37: 'Ankle ', 38: 'Pubic Region ', 75: 'Head ', 76: 'Face ', 77: 'Eyeball ', 79: 'Lower Trunk', 80: 'Upper Arm', 81: 'Upper Leg ', 82: 'Hand ', 83: 'Foot ', 84: '25-50% of Body ', 85: 'All Parts Body ', 87: 'Not Stated/Unk ', 88: 'Mouth ', 89: 'Neck ', 92: 'Finger ', 93: 'Toe', 94: 'Ear '});

## Diagnosis
df['Diagnosis']=df['Diagnosis'].replace({41: 'Ingestion ', 42: 'Aspiration ', 46: 'Burns, Electrical ', 47: 'Burns, Not Specified', 48: 'Burns, Scald ', 49: 'Burns, Chemical ', 50: 'Amputaion ', 51: 'Burns, Thermal ', 52: 'Concussions', 53: 'Contusions, Abrasions', 54: 'Crushing', 55: 'Dislocation', 56: 'Foreign Body', 57: 'Fracture', 58: 'Hematoma', 59: 'Laceration', 60: 'Dental Injury', 61: 'Nerve Damage', 62: 'Internal Organ Injury', 63: 'Puncture', 64: 'Strain, Sprain', 65: 'Anoxia', 66: 'Hemorrhage', 67: 'Electric Shock', 68: 'Poisoning', 69: 'Submersion', 71: 'Other/Not Stated', 72: 'Avulsion', 73: 'Burns, Radiation', 74: 'Dermatitis, Conjunctivitis',});

df['Diagnosis_2']=df['Diagnosis_2'].replace({41: 'Ingestion ', 42: 'Aspiration ', 46: 'Burns, Electrical ', 47: 'Burns, Not Specified', 48: 'Burns, Scald ', 49: 'Burns, Chemical ', 50: 'Amputaion ', 51: 'Burns, Thermal ', 52: 'Concussions', 53: 'Contusions, Abrasions', 54: 'Crushing', 55: 'Dislocation', 56: 'Foreign Body', 57: 'Fracture', 58: 'Hematoma', 59: 'Laceration', 60: 'Dental Injury', 61: 'Nerve Damage', 62: 'Internal Organ Injury', 63: 'Puncture', 64: 'Strain, Sprain', 65: 'Anoxia', 66: 'Hemorrhage', 67: 'Electric Shock', 68: 'Poisoning', 69: 'Submersion', 71: 'Other/Not Stated', 72: 'Avulsion', 73: 'Burns, Radiation', 74: 'Dermatitis, Conjunctivitis',});

## Disposition
df['Disposition']=df['Disposition'].replace({1: 'Treated/Examined and Released ', 2: 'Treated and Transferred', 4: 'Treated and Admitted/Hospitalized ', 5: 'Held for Observation', 6: 'Left Without Being Seen', 8: 'Fatality, Incl. DOA, Died in ER', 9: 'Unknown, Not Stated ',});

##
df.insert(len(df.columns),'Disposition_refactored',df.Disposition)
df['Disposition_refactored']=df['Disposition_refactored'].replace({'Treated and Transferred':'Transferred',
                                                                    'Treated and Admitted/Hospitalized ':'Admitted',
                                                                    'Treated/Examined and Released ':'Released',
                                                                    'Held for Observation':'Admitted',
                                                                    'Left Without Being Seen':'Released',
                                                                    'Fatality, Incl. DOA, Died in ER':'Died in ED',
                                                                    'Unknown, Not Stated ':np.nan})


## Location
df['Location']=df['Location'].replace({0: 'Not Recorded', 1: 'Home', 2: 'Farm/Ranch', 4: 'Street or Highway', 5: 'Other Public Property', 6: 'Mobile/Manufactured Home', 7: 'Industrial', 8: 'School/Daycare', 9: 'Place of Recreation or Sports',});

## get number of levels fused
col_level=[col for col in df.columns if col.startswith('Level')]
df['num_levels']=df[col_level].sum(axis=1)
df.loc[df['num_levels']==0,'num_levels']=np.nan # cast zero levels as nan

df.to_csv('NEISS_cleaned.csv');