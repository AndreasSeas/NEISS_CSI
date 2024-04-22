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
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os, sys
import seaborn as sns
import statsmodels.api as statmod

# =============================================================================
# load data and prepare global variables
# =============================================================================
homedir=os.getcwd();
df=pd.read_csv('NEISS_cleaned.csv')
codes=pd.read_csv('us-national-electronic-injury-surveillance-system-neiss-product-codes.csv',index_col='Code')

savetabs=False
# get the codes for falls
set1=set(df.Product_1.unique())
set2=set(df.Product_2.unique())
set3=set(df.Product_3.unique())
allU=set1.union(set2,set3).intersection(set(codes.index))
hex_female='#56B4E9';
hex_male='#E69F00';

binary_save = True

ourcodes=codes.loc[list(allU),:]
ourcodes["N cases"]=np.nan

for code in ourcodes.index:
    ourcodes.loc[code,'N cases']=(df[["Product_1","Product_2","Product_3"]]==code).sum().sum()
    
# ourcodes.to_excel('ourcodes.xlsx')
# 

# =============================================================================
# "Demographics" Table
# =============================================================================
tempdf=df.loc[:,['Age','Weight']].dropna()
descr = statmod.stats.DescrStatsW(data=tempdf.Age,weights=tempdf.Weight);
print(descr.mean)
print(descr.std)

def getn(df,col):
    tot=df.Weight.sum();
    for uval in df[col].unique():
        val=df.loc[df[col]==uval,'Weight'].sum()
        pct=val/tot*100
        print("{} n = {:10.0f}, ({:3.1f}%)".format(uval,val,pct));
    
    val=df.loc[df[col].isnull(),'Weight'].sum()
    pct=val/tot*100
    print("null n = {:10.0f}, ({:3.1f}%)".format(val,pct));

col_levels=[col for col in df if col.startswith('Level')]
df['nolevel']=(df[col_levels].sum(axis=1)==0)
getn(df,'nolevel')

df15=df.loc[df.nolevel==False,:]
tempdf=df15.loc[:,['Age','Weight']].dropna()
descr = statmod.stats.DescrStatsW(data=tempdf.Age,weights=tempdf.Weight);
print(descr.mean)
print(descr.std)


df2=df.loc[df.nolevel,:]
tempdf=df2.loc[:,['Age','Weight']].dropna()
descr = statmod.stats.DescrStatsW(data=tempdf.Age,weights=tempdf.Weight);
print(descr.mean)
print(descr.std)

getn(df,'Race')

getn(df,'Sex')
            
getn(df,'Hispanic')

getn(df,'Fall')

getn(df,'Level_C1')

print('has level')


getn(df15,'Race')

getn(df15,'Sex')
            
getn(df15,'Hispanic')

getn(df15,'Fall')

print('no level')

getn(df2,'Race')

getn(df2,'Sex')
            
getn(df2,'Hispanic')

getn(df2,'Fall')

# =============================================================================
# "Demographics" Table for C1-3 vs C4-7
# =============================================================================

col_levels=[col for col in df if col.startswith('Level')]
name_levels=[i.split('_', 1)[1] for i in col_levels]

col_123=col_levels[0:3]
col_4567=col_levels[3:]

df[col_123].sum(axis=0)

df['col_123']=df[col_123].sum(axis=1)>0;
df['col_4567']=df[col_4567].sum(axis=1)>0;
df['col_both']=df['col_4567'] & df['col_123']

df123=df.loc[df['col_123']&~df['col_both'],:]
df4567=df.loc[df['col_4567']&~df['col_both'],:]

print('C123')
print(df123.Weight.sum())
getn(df123,'Race')

getn(df123,'Sex')
            
getn(df123,'Hispanic')

getn(df123,'Fall')


print('C4567')
print(df4567.Weight.sum())
getn(df4567,'Race')

getn(df4567,'Sex')
            
getn(df4567,'Hispanic')

getn(df4567,'Fall')


descr_123 = statmod.stats.DescrStatsW(data=df123.Age,weights=df123.Weight);
descr_4567 = statmod.stats.DescrStatsW(data=df4567.Age,weights=df4567.Weight);

[zstat,pvalue]=statmod.stats.CompareMeans(descr_123,descr_4567).ztest_ind();

print('C123')
print(df123.Weight.sum())
getn(df123,'Disposition')

print('C4567')
print(df4567.Weight.sum())
getn(df4567,'Disposition')

df_=df.copy()

# =============================================================================
# Figure 1, Patients Over Time
# =============================================================================

allplot=pd.DataFrame(data=None,columns=['# Patients','% Population','% Female','% Male'],
                     index=np.arange(df.year.min(),df.year.max()+1))

n_us_pop=[291109820,
            293947885,
            296842670,
            299753098,
            302743399,
            305694910,
            308512035,
            311182845,
            313876608,
            316651321,
            319375166,
            322033964,
            324607776,
            327210198,
            329791231,
            332140037,
            334319671,
            335942003,
            336997624,
            338289857];

for i,yr in enumerate(allplot.index):
    tempdf=df.loc[df.year==yr,['Weight']].dropna()
    allplot.loc[yr,'# Patients']=tempdf.Weight.sum()
    allplot.loc[yr,'% Population']=tempdf.Weight.sum()/n_us_pop[i]
    
    tempdf=df.loc[(df.year==yr)&(df.Sex=='Female'),['Weight']].dropna()
    allplot.loc[yr,'% Female']=tempdf.Weight.sum()/n_us_pop[i]
    
    tempdf=df.loc[(df.year==yr)&(df.Sex=='Male'),['Weight']].dropna()
    allplot.loc[yr,'% Male']=tempdf.Weight.sum()/n_us_pop[i]
    
fig,axs = plt.subplots(nrows=3,ncols=1,figsize=(6,12),sharex=True)
ax=axs[0];

x_all=allplot['% Population'];

x_all = statmod.add_constant(x_all.values.astype(float), prepend=False);
mod = statmod.OLS(allplot.index.values, x_all);
res_all = mod.fit();

x_all=allplot['% Female'];

x_all = statmod.add_constant(x_all.values.astype(float), prepend=False);
mod = statmod.OLS(allplot.index.values, x_all);
res_f = mod.fit();

x_all=allplot['% Male'];

x_all = statmod.add_constant(x_all.values.astype(float), prepend=False);
mod = statmod.OLS(allplot.index.values, x_all);
res_m = mod.fit();

ax.plot(allplot.index,allplot['% Population']*100000,'-o',color='#191919',
        label='all patients, p = {:0.3f}'.format(res_all.pvalues[0]))

ax.plot(allplot.index,allplot['% Female']*100000,'-o',color=hex_female,
        label='female, p = {:0.3f}'.format(res_f.pvalues[0]))

ax.plot(allplot.index,allplot['% Male']*100000,'-o',color=hex_male,
        label='male, p = {:0.3f}'.format(res_m.pvalues[0]))
# ax.fill_between(ageplot.index,
#                  ageplot['Uci'].astype(float),
#                  ageplot['Lci'].astype(float),
#                  color='#191919',alpha=0.5,edgecolor=None)#label='95% CI',)
ax.set_xticks(allplot.index,label=allplot.index,rotation=45)
ax.set_xticklabels(allplot.index,rotation=90)

# ax.set_xlabel('Year')
ax.set_ylabel('Incidence (per 100,000)')

lims=ax.get_ylim()
ax.set_ylim((lims[0],lims[1]+(lims[1]-lims[0])*0.12))

ax.text(2003,lims[1]+(lims[1]-lims[0])*0.1,'A',va='top',fontsize=20)
# ax.text(2003,lims[1]-(lims[1]-lims[0])*0.02,'A',va='top',fontsize=20)
ax.legend(loc='upper center')

# =============================================================================
# Age at injury over time
# =============================================================================
ageplot=pd.DataFrame(data=None,columns=['mean','Uci','Lci'],
                     index=np.arange(df.year.min(),df.year.max()+1))

#stats
tempdf=df.loc[:,['year','Age','Weight']].dropna()
x_all = statmod.add_constant(tempdf.year.values.astype(float), prepend=False);
y_all = tempdf.Age.values.astype(float)
wt_all = tempdf.Weight.values.astype(float)
mod = statmod.WLS(y_all, x_all,weights=wt_all);
res_all = mod.fit();

tempdf=df.loc[df.Sex=='Male',['year','Age','Weight']].dropna()
x_m = statmod.add_constant(tempdf.year.values.astype(float), prepend=False);
y_m = tempdf.Age.values.astype(float)
wt_m = tempdf.Weight.values.astype(float)
mod = statmod.WLS(y_m, x_m,weights=wt_m);
res_m = mod.fit();

tempdf=df.loc[df.Sex=='Female',['year','Age','Weight']].dropna()
x_f = statmod.add_constant(tempdf.year.values.astype(float), prepend=False);
y_f = tempdf.Age.values.astype(float)
wt_f = tempdf.Weight.values.astype(float)
mod = statmod.WLS(y_f, x_f,weights=wt_f);
res_f = mod.fit();

for yr in ageplot.index:
    tempdf=df.loc[df.year==yr,['Age','Weight']].dropna()
    descr = statmod.stats.DescrStatsW(data=tempdf.Age,weights=tempdf.Weight);
    
    ageplot.loc[yr,"mean"] = descr.mean;
    ageplot.loc[yr,"Uci"] = descr.mean+1.96*descr.std/np.sqrt(len(tempdf));
    ageplot.loc[yr,"Lci"] = descr.mean-1.96*descr.std/np.sqrt(len(tempdf));

ax=axs[1]

ax.plot(ageplot.index,ageplot['mean'],'-o',color='#191919',label='all patients, p = {:0.3f}'.format(res_all.pvalues[0]))
ax.fill_between(ageplot.index,
                 ageplot['Uci'].astype(float),
                 ageplot['Lci'].astype(float),
                 color='#191919',alpha=0.5,edgecolor=None)#label='95% CI',)
ax.set_xticks(ageplot.index,label=ageplot.index,rotation=45)
ax.set_xticklabels(ageplot.index,rotation=90)

# ax.set_xlabel('Year')
ax.set_ylabel('Patient Age')


# if binary_save: fig.savefig('NEISS_finalfigures_240406/AgexYear.png',dpi=300,)
# =============================================================================
# Age at injury over time, stratified by sex
# =============================================================================
ageplot=pd.DataFrame(data=None,columns=['mean','Uci','Lci'],
                     index=np.arange(df.year.min(),df.year.max()+1))

for yr in ageplot.index:
    tempdf=df.loc[(df.year==yr)&(df.Sex=='Female'),
                  ['Age','Weight']].dropna()
    descr = statmod.stats.DescrStatsW(data=tempdf.Age,weights=tempdf.Weight);
    
    ageplot.loc[yr,"mean"] = descr.mean;
    ageplot.loc[yr,"Uci"] = descr.mean+1.96*descr.std/np.sqrt(len(tempdf));
    ageplot.loc[yr,"Lci"] = descr.mean-1.96*descr.std/np.sqrt(len(tempdf));
    
# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4))

ax.plot(ageplot.index,ageplot['mean'],'-o',color='#56B4E9',label='female, p = {:0.3f}'.format(res_f.pvalues[0]))
ax.fill_between(ageplot.index,
                 ageplot['Uci'].astype(float),
                 ageplot['Lci'].astype(float),
                 color='#56B4E9',alpha=0.5,edgecolor=None)#label='95% CI',)

ageplot=pd.DataFrame(data=None,columns=['mean','Uci','Lci'],
                     index=np.arange(df.year.min(),df.year.max()+1))

for yr in ageplot.index:
    tempdf=df.loc[(df.year==yr)&(df.Sex=='Male'),
                  ['Age','Weight']].dropna()
    descr = statmod.stats.DescrStatsW(data=tempdf.Age,weights=tempdf.Weight);
    
    ageplot.loc[yr,"mean"] = descr.mean;
    ageplot.loc[yr,"Uci"] = descr.mean+1.96*descr.std/np.sqrt(len(tempdf));
    ageplot.loc[yr,"Lci"] = descr.mean-1.96*descr.std/np.sqrt(len(tempdf));
    
# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4))

ax.plot(ageplot.index,ageplot['mean'],'-o',color='#E69F00',label='male, p = {:0.3f}'.format(res_m.pvalues[0]))
ax.fill_between(ageplot.index,
                 ageplot['Uci'].astype(float),
                 ageplot['Lci'].astype(float),
                 color='#E69F00',alpha=0.5,edgecolor=None)#label='95% CI',)


# if binary_save: fig.savefig('NEISS_finalfigures_240406/AgexYearxSex.png',dpi=300,)


tempdf=df.loc[(df.Sex=='Male'),['Age','Weight']].dropna()
descr_m = statmod.stats.DescrStatsW(data=tempdf.Age,weights=tempdf.Weight);
tempdf_m=tempdf.copy()
tempdf=df.loc[(df.Sex=='Female'),['Age','Weight']].dropna()
descr_f = statmod.stats.DescrStatsW(data=tempdf.Age,weights=tempdf.Weight);
tempdf_f=tempdf.copy()

tempdf_m['Sex']='Male';
tempdf_f['Sex']='Female';

df_box=pd.concat([tempdf_m,tempdf_f],axis=0)

# sns.stripplot(data=df_box,x='Sex',y='Age',hue='Weight',dodge=True)
# sns.violinplot(data=df_box,x='Sex',y='Age',)
# sns.kdeplot(data=df_box,hue='Sex',y='Age',weights='Weight',multiple='layer',fill=True,alpha=0.5)
# sys.exit()
[zstat,pvalue]=statmod.stats.CompareMeans(descr_m,descr_f).ztest_ind();

lims=ax.get_ylim()
ax.set_ylim((lims[0],lims[1]+(lims[1]-lims[0])*0.1))

ax.text(2003,lims[1]+(lims[1]-lims[0])*0.08,'B',va='top',fontsize=20)


ax.legend(loc='lower right')

# =============================================================================
# Falls over time
# =============================================================================
fallsplot=pd.DataFrame(data=None,columns=['% Falls'],
                     index=np.arange(df.year.min(),df.year.max()+1))

for yr in ageplot.index:
    tempdf=df.loc[df.year==yr,['Fall','Weight']].dropna()
    fallsplot.loc[yr,'% Falls']=tempdf.loc[tempdf.Fall==1,'Weight'].sum()/tempdf.Weight.sum()*100
    
# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4))
ax=axs[2]

x_all=fallsplot['% Falls'];

fallsplot_f=pd.DataFrame(data=None,columns=['% Falls'],
                     index=np.arange(df.year.min(),df.year.max()+1))

for yr in ageplot.index:
    tempdf=df.loc[(df.year==yr)&(df.Sex=='Female'),
                  ['Fall','Weight']].dropna()
    fallsplot_f.loc[yr,'% Falls']=tempdf.loc[tempdf.Fall==1,'Weight'].sum()/tempdf.Weight.sum()*100

x_f=fallsplot_f['% Falls'];

fallsplot_m=pd.DataFrame(data=None,columns=['% Falls'],
                     index=np.arange(df.year.min(),df.year.max()+1))

for yr in ageplot.index:
    tempdf=df.loc[(df.year==yr)&(df.Sex=='Male'),
                  ['Fall','Weight']].dropna()
    fallsplot_m.loc[yr,'% Falls']=tempdf.loc[tempdf.Fall==1,'Weight'].sum()/tempdf.Weight.sum()*100

# ax.legend()

x_m=fallsplot_m['% Falls'];

[tstat,pvalue,df]=statmod.stats.ttest_ind(x_f.values,x_m.values);

x_all = statmod.add_constant(x_all.values.astype(float), prepend=False);
mod = statmod.OLS(fallsplot.index.values, x_all);
res_all = mod.fit();

x_m = statmod.add_constant(x_m.values.astype(float), prepend=False);
mod = statmod.OLS(fallsplot.index.values, x_m);
res_m = mod.fit();

x_f = statmod.add_constant(x_f.values.astype(float), prepend=False);
mod = statmod.OLS(fallsplot.index.values, x_f);
res_f = mod.fit();

ax.plot(fallsplot.index,fallsplot['% Falls'],'-o',color='#191919',label='all patients, p = {:0.3f}'.format(res_all.pvalues[0]))
# ax.fill_between(ageplot.index,
#                  ageplot['Uci'].astype(float),
#                  ageplot['Lci'].astype(float),
#                  color='#191919',alpha=0.5,edgecolor=None)#label='95% CI',)
ax.set_xticks(ageplot.index,label=ageplot.index,rotation=45)
ax.set_xticklabels(ageplot.index,rotation=90)

ax.set_xlabel('Year')
ax.set_ylabel('% Falls')

ax.plot(fallsplot_f.index,fallsplot_f['% Falls'],'-o',color='#56B4E9',label='female, p = {:0.3f}'.format(res_f.pvalues[0]))

ax.plot(fallsplot_m.index,fallsplot_m['% Falls'],'-o',color='#E69F00',label='male, p = {:0.3f}'.format(res_m.pvalues[0]))

lims=ax.get_ylim()
ax.set_ylim((lims[0],lims[1]+(lims[1]-lims[0])*0.1))

ax.text(2003,lims[1]+(lims[1]-lims[0])*0.08,'C',va='top',fontsize=20)

descr_m = statmod.stats.DescrStatsW(data=x_m[:,0]);
descr_f = statmod.stats.DescrStatsW(data=x_f[:,0]);

[tstat,pvalue,dof]=statmod.stats.CompareMeans(descr_m,descr_f).ttest_ind();


ax.legend(loc='upper right')
if binary_save: fig.savefig('fig1_by_year.tiff',dpi=400,)
if binary_save: fig.savefig('fig1_by_year.png',dpi=400,)
if binary_save: fig.savefig('fig1_by_year.eps',)

# =============================================================================
# Fig2 v2
# =============================================================================
### this seems way too complex, going back to the OG, maybe making it a bit nicer heatmap
df=df_.copy() # to save df
col_levels=[col for col in df if col.startswith('Level')]

df_melt=pd.melt(df_,id_vars=['Sex','Age','Weight'],value_vars=col_levels)
df_melt=df_melt.loc[df_melt.value==1,:]

df_melt['Cspine Level']=np.nan
for i,colname in enumerate(col_levels):
    df_melt.loc[df_melt.variable==colname,'Cspine Level']=-(i+1);

# plt.figure()
# sns.kdeplot(data=df_melt.loc[df_melt.Sex=='Male'],x='Age',y='levelN',
#             weights='Weight',fill=True,vmin=0,vmax=0.01,cmap=cmap)

# plt.figure()
# sns.kdeplot(data=df_melt.loc[df_melt.Sex=='Female'],x='Age',y='levelN',
#             weights='Weight',fill=True,vmin=0,vmax=0.01,cmap=cmap)

## another approach
from matplotlib.colors import LinearSegmentedColormap

g = sns.JointGrid(x="Age", y="Cspine Level", data=df_melt,height=5)

cmap_F=LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFFFFF-56B4E9-0F4D71
    (0.000, (1.000, 1.000, 1.000)),
    (0.500, (0.337, 0.706, 0.914)),
    (1.000, (0.059, 0.302, 0.443))))


sns.kdeplot(data=df_melt.loc[df_melt.Sex=='Female'],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=cmap_F,
            ax=g.ax_joint,
            shade=True,)


sns.kdeplot(data=df_melt.loc[df_melt.Sex=='Female'],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=None,
            colors='k',
            linewidths=0.5,
            ax=g.ax_joint,)

sns.distplot(df_melt.loc[df_melt.Sex=='Female','Age'],
             kde=False, 
             color=hex_female,
             bins=np.arange(0,102)+0.5,
             ax=g.ax_marg_x,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

sns.distplot(df_melt.loc[df_melt.Sex=='Female','Cspine Level'],
             kde=False, 
             color=hex_female, 
             ax=g.ax_marg_y,
             bins=np.arange(0,8)-7.5,
             vertical=True,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

g.ax_joint.set_yticks([0,-1,-2,-3,-4,-5,-6,-7,-8],
                      ['','C1','C2','C3','C4','C5','C6','C7',''])
g.ax_joint.set_xticks(np.arange(0,110,10))

xlim=g.ax_joint.get_xlim()
ylim=g.ax_joint.get_ylim()

if binary_save: g.savefig('fig2_a_LevelxAge_Female.eps',)

## another approach
g = sns.JointGrid(x="Age", y="Cspine Level", data=df_melt,height=5)

cmap_M=LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFFFFF-E69F00-5C4000
    (0.000, (1.000, 1.000, 1.000)),
    (0.500, (0.902, 0.624, 0.000)),
    (1.000, (0.361, 0.251, 0.000))))

sns.kdeplot(data=df_melt.loc[df_melt.Sex=='Male'],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=cmap_M,
            ax=g.ax_joint,
            shade=True,)

sns.kdeplot(data=df_melt.loc[df_melt.Sex=='Male'],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=None,
            colors='k',
            linewidths=0.5,
            ax=g.ax_joint,)

sns.distplot(df_melt.loc[df_melt.Sex=='Male','Age'],
             kde=False, 
             color=hex_male,
             bins=np.arange(0,102)+0.5,
             ax=g.ax_marg_x,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

sns.distplot(df_melt.loc[df_melt.Sex=='Male','Cspine Level'],
             kde=False, 
             color=hex_male,
             ax=g.ax_marg_y,
             bins=np.arange(0,8)-7.5,
             vertical=True,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

g.ax_joint.set_yticks([0,-1,-2,-3,-4,-5,-6,-7,-8],
                      ['','C1','C2','C3','C4','C5','C6','C7',''])
g.ax_joint.set_xticks(np.arange(0,110,10))

g.ax_joint.set_xlim(xlim)
g.ax_joint.set_ylim(ylim)

if binary_save: g.savefig('fig2_b_LevelxAge_Male.eps',)
# if binary_save: g.savefig('fig_by_year.tiff',dpi=400,)
# if binary_save: g.savefig('fig1_by_year.png',dpi=400,)
# if binary_save: g.savefig('fig1_by_year.eps',)

# =============================================================================
# Fig6 
# =============================================================================
### this seems way too complex, going back to the OG, maybe making it a bit nicer heatmap
df=df_.copy() # to save df
col_levels=[col for col in df if col.startswith('Level')]

df_melt=pd.melt(df_,id_vars=['Sex','Age','Weight','Fall'],value_vars=col_levels)
df_melt=df_melt.loc[df_melt.value==1,:]

df_melt['Cspine Level']=np.nan
for i,colname in enumerate(col_levels):
    df_melt.loc[df_melt.variable==colname,'Cspine Level']=-(i+1);

# plt.figure()
# sns.kdeplot(data=df_melt.loc[df_melt.Sex=='Male'],x='Age',y='levelN',
#             weights='Weight',fill=True,vmin=0,vmax=0.01,cmap=cmap)

# plt.figure()
# sns.kdeplot(data=df_melt.loc[df_melt.Sex=='Female'],x='Age',y='levelN',
#             weights='Weight',fill=True,vmin=0,vmax=0.01,cmap=cmap)

## another approach
from matplotlib.colors import LinearSegmentedColormap

g = sns.JointGrid(x="Age", y="Cspine Level", data=df_melt,height=5)

cmap_F=LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:FFFFFF-47.6:E386E5-100:C500C8
    (0.000, (1.000, 1.000, 1.000)),
    (0.476, (0.890, 0.525, 0.898)),
    (1.000, (0.773, 0.000, 0.784))))


sns.kdeplot(data=df_melt.loc[df_melt.Fall==1],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=cmap_F,
            ax=g.ax_joint,
            shade=True,)


sns.kdeplot(data=df_melt.loc[df_melt.Fall==1],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=None,
            colors='k',
            linewidths=0.5,
            ax=g.ax_joint,)

sns.distplot(df_melt.loc[df_melt.Fall==1,'Age'],
             kde=False, 
             color="#E386E5",
             bins=np.arange(0,102)+0.5,
             ax=g.ax_marg_x,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

sns.distplot(df_melt.loc[df_melt.Fall==1,'Cspine Level'],
             kde=False, 
             color="#E386E5", 
             ax=g.ax_marg_y,
             bins=np.arange(0,8)-7.5,
             vertical=True,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

g.ax_joint.set_yticks([0,-1,-2,-3,-4,-5,-6,-7,-8],
                      ['','C1','C2','C3','C4','C5','C6','C7',''])
g.ax_joint.set_xticks(np.arange(0,110,10))

xlim=g.ax_joint.get_xlim()
ylim=g.ax_joint.get_ylim()

if binary_save: g.savefig('fig6_a_LevelxAge_YesFall.eps',)

## another approach
g = sns.JointGrid(x="Age", y="Cspine Level", data=df_melt,height=5)

cmap_M=LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:FFFFFF-49.1:A7D180-100:6CB32C
    (0.000, (1.000, 1.000, 1.000)),
    (0.491, (0.655, 0.820, 0.502)),
    (1.000, (0.424, 0.702, 0.173))))

sns.kdeplot(data=df_melt.loc[df_melt.Fall==0],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=cmap_M,
            ax=g.ax_joint,
            shade=True,)


sns.kdeplot(data=df_melt.loc[df_melt.Fall==0],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=None,
            colors='k',
            linewidths=0.5,
            ax=g.ax_joint,)

sns.distplot(df_melt.loc[df_melt.Fall==0,'Age'],
             kde=False, 
             color="#A7D180",
             bins=np.arange(0,102)+0.5,
             ax=g.ax_marg_x,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

sns.distplot(df_melt.loc[df_melt.Fall==0,'Cspine Level'],
             kde=False, 
             color="#A7D180", 
             ax=g.ax_marg_y,
             bins=np.arange(0,8)-7.5,
             vertical=True,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

g.ax_joint.set_yticks([0,-1,-2,-3,-4,-5,-6,-7,-8],
                      ['','C1','C2','C3','C4','C5','C6','C7',''])
g.ax_joint.set_xticks(np.arange(0,110,10))

g.ax_joint.set_xlim(xlim)
g.ax_joint.set_ylim(ylim)

if binary_save: g.savefig('fig6_a_LevelxAge_NoFall.eps',)

# =============================================================================
# Fig5
# =============================================================================
### this seems way too complex, going back to the OG, maybe making it a bit nicer heatmap
df=df_.copy() # to save df
col_levels=[col for col in df if col.startswith('Level')]

df_melt=pd.melt(df_,id_vars=['Sex','Age','Weight','Fall','Race'],value_vars=col_levels)
df_melt=df_melt.loc[df_melt.value==1,:]

df_melt['Cspine Level']=np.nan
for i,colname in enumerate(col_levels):
    df_melt.loc[df_melt.variable==colname,'Cspine Level']=-(i+1);

# plt.figure()
# sns.kdeplot(data=df_melt.loc[df_melt.Sex=='Male'],x='Age',y='levelN',
#             weights='Weight',fill=True,vmin=0,vmax=0.01,cmap=cmap)

# plt.figure()
# sns.kdeplot(data=df_melt.loc[df_melt.Sex=='Female'],x='Age',y='levelN',
#             weights='Weight',fill=True,vmin=0,vmax=0.01,cmap=cmap)

## another approach
from matplotlib.colors import LinearSegmentedColormap

g = sns.JointGrid(x="Age", y="Cspine Level", data=df_melt,height=5)

cmap_F=LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:FFFFFF-47.6:D2A391-100:AA512F
    (0.000, (1.000, 1.000, 1.000)),
    (0.476, (0.824, 0.639, 0.569)),
    (1.000, (0.667, 0.318, 0.184))))


sns.kdeplot(data=df_melt.loc[df_melt.Race=="White"],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=cmap_F,
            ax=g.ax_joint,
            shade=True,)


sns.kdeplot(data=df_melt.loc[df_melt.Race=="White"],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=None,
            colors='k',
            linewidths=0.5,
            ax=g.ax_joint,)

sns.distplot(df_melt.loc[df_melt.Race=="White",'Age'],
             kde=False, 
             color="#D2A391",
             bins=np.arange(0,102)+0.5,
             ax=g.ax_marg_x,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

sns.distplot(df_melt.loc[df_melt.Race=="White",'Cspine Level'],
             kde=False, 
             color="#D2A391", 
             ax=g.ax_marg_y,
             bins=np.arange(0,8)-7.5,
             vertical=True,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

g.ax_joint.set_yticks([0,-1,-2,-3,-4,-5,-6,-7,-8],
                      ['','C1','C2','C3','C4','C5','C6','C7',''])
g.ax_joint.set_xticks(np.arange(0,110,10))

xlim=g.ax_joint.get_xlim()
ylim=g.ax_joint.get_ylim()

if binary_save: g.savefig('fig5_a_LevelxAge_White.eps',)

## another approach
g = sns.JointGrid(x="Age", y="Cspine Level", data=df_melt,height=5)

cmap_M=LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:FFFFFF-47.8:9FE7E4-100:48D1CC
    (0.000, (1.000, 1.000, 1.000)),
    (0.478, (0.624, 0.906, 0.894)),
    (1.000, (0.282, 0.820, 0.800))))

sns.kdeplot(data=df_melt.loc[df_melt.Race!="White"],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=cmap_M,
            ax=g.ax_joint,
            shade=True,)


sns.kdeplot(data=df_melt.loc[df_melt.Race!="White"],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=None,
            colors='k',
            linewidths=0.5,
            ax=g.ax_joint,)

sns.distplot(df_melt.loc[df_melt.Race!="White",'Age'],
             kde=False, 
             color="#9FE7E4",
             bins=np.arange(0,102)+0.5,
             ax=g.ax_marg_x,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

sns.distplot(df_melt.loc[df_melt.Race!="White",'Cspine Level'],
             kde=False, 
             color="#9FE7E4", 
             ax=g.ax_marg_y,
             bins=np.arange(0,8)-7.5,
             vertical=True,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

g.ax_joint.set_yticks([0,-1,-2,-3,-4,-5,-6,-7,-8],
                      ['','C1','C2','C3','C4','C5','C6','C7',''])
g.ax_joint.set_xticks(np.arange(0,110,10))

g.ax_joint.set_xlim(xlim)
g.ax_joint.set_ylim(ylim)

if binary_save: g.savefig('fig5_a_LevelxAge_NonWhite.eps',)

# =============================================================================
# Figure 3: all patients
# =============================================================================

## another approach
g = sns.JointGrid(x="Age", y="Cspine Level", data=df_melt,height=5)

cmap_M=LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:FFFFFF-49.7:808080-100:000000
    (0.000, (1.000, 1.000, 1.000)),
    (0.497, (0.502, 0.502, 0.502)),
    (1.000, (0.000, 0.000, 0.000))))

sns.kdeplot(data=df_melt.loc[df_melt.Fall>=0],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=cmap_M,
            ax=g.ax_joint,
            shade=True,)

sns.kdeplot(data=df_melt.loc[df_melt.Fall>=0],
            x='Age',
            y='Cspine Level',
            weights='Weight',
            cmap=None,
            colors='k',
            linewidths=0.5,
            ax=g.ax_joint,)

sns.distplot(df_melt.loc[df_melt.Fall>=0,'Age'],
             kde=False, 
             color="#808080",
             bins=np.arange(0,102)+0.5,
             ax=g.ax_marg_x,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

sns.distplot(df_melt.loc[df_melt.Fall>=0,'Cspine Level'],
             kde=False, 
             color="#808080",
             ax=g.ax_marg_y,
             bins=np.arange(0,8)-7.5,
             vertical=True,
             hist_kws=dict(edgecolor="k", linewidth=0.2,alpha=1))

g.ax_joint.set_yticks([0,-1,-2,-3,-4,-5,-6,-7,-8],
                      ['','C1','C2','C3','C4','C5','C6','C7',''])
g.ax_joint.set_xticks(np.arange(0,110,10))

g.ax_joint.set_xlim(xlim)
g.ax_joint.set_ylim(ylim)

if binary_save: g.savefig('fig3_a_LevelxAge_forall.eps',)

# =============================================================================
# Figure 4
# =============================================================================
from sklearn.cluster import KMeans


cols=['Cspine Level','Fall']
# cols=['Cspine Level','Age','Fall']
X = df_melt[cols].astype(float).to_numpy()
kmeans = KMeans(n_clusters=2,random_state=0,)
kmeans = kmeans.fit(X)

df_melt['labels']=kmeans.labels_;

fig, ax = plt.subplots()
sns.stripplot(data=df_melt,x='Cspine Level',y='Age',hue='labels',ax=ax,palette='Set1')
ax.get_legend().set_visible(False)
ax.set_xticks([0,1,2,3,4,5,6],
                      ['C7','C6','C5','C4','C3','C2','C1'])

if binary_save: fig.savefig('fig4KDE.eps',)

sys.exit()
