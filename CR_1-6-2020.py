#%%Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns
import pickle
import datetime as dt
from scipy.optimize import fsolve
from scipy.stats import norm
from datetime import datetime, date, timedelta
from pandas.tseries.offsets import BMonthEnd
#from pandas.tseries.offsets import BDay
#pip instal tqdm
from tqdm import tqdm_notebook as tqdm

#%% Load Dataset

#dataset=pd.read_excel('BaseDados.xlsx', sheet_name='Valores')
#dataset=pd.read_excel('BaseDadosTesla.xlsx', sheet_name='Valores') - T-Bill 3m
dataset=pd.read_excel('BaseDadosTesla2.xlsx', sheet_name='Valores') # - T-Bill 1y
#aux=pd.read_excel('BaseDadosTesla3.xlsx', sheet_name='Valores') # - T-Bill 1y wih Net Income
#dataset.drop(dataset.columns[0],axis=1, inplace=True) - apagar
dataset.head()

dataset['F'][dataset['F']>dataset.F.quantile(0.98)]=dataset.F.quantile(0.98)
dataset['ImpVolE'][dataset['ImpVolE']>dataset.ImpVolE.quantile(0.98)]=dataset.ImpVolE.quantile(0.98)
dataset['r'][dataset['r']>dataset.r.quantile(0.98)]=dataset.r.quantile(0.98)
dataset['E'][dataset['E']>dataset.E.quantile(0.98)]=dataset.E.quantile(0.98)
dataset['PX'][dataset['PX']>dataset.PX.quantile(0.98)]=dataset.PX.quantile(0.98)

# %% Last year daily returns
dataset['log_ret'] = np.log(dataset.E) - np.log(dataset.E.shift(1))

#%% Variable description

summary=dataset.describe(include='all')
summary=summary.transpose()
summary.head(len(summary))

#%% Vector V - Löffler and Posh (2011, Chapter 2)
w=252
dataset['V'] = dataset['F']+dataset['E']

#V - log returns
dataset['V_log_ret']=np.log(dataset.V) - np.log(dataset.V.shift(1))


#Sigma E calculation - Rolling window of previous 252 days
dataset['Vassalou_sig_e']=dataset['log_ret'].rolling(w).std()*np.sqrt(w)
dataset['LofflerPosch_sig_v']=dataset['V_log_ret'].rolling(w).std()*np.sqrt(w)
dataset['BharatShumway_sig_v']=dataset['log_ret'].rolling(w).std()*np.sqrt(w)*(dataset['E']/(dataset['E']+dataset['F']))


dataset.head()

#%% Non-Linear System of equations - Dias, José Carlos (2020 Credit Risk - Lecture Notes, Equations 7.1 and 7.2)

def non_linear_equations (x):
    V, sigma_V=x
    #d1 = (np.log(V* np.exp(r*T)/face_val_debt) + (r + 0.5*sigma_V**2)*T)/(sigma_V*np.sqrt(T))
    d1 = (np.log(V/face_val_debt) + (r + 0.5*sigma_V**2)*T)/(sigma_V*np.sqrt(T))
    d2 = d1 - sigma_V*np.sqrt(T)
    f1 = E - V*norm.cdf(d1) + np.exp(-r*T)*face_val_debt*norm.cdf(d2)
    f2=V/E*norm.cdf(d1)*sigma_V-sigma_E

    return (f1,f2)


#%% Function for Question 2 b) and c)

def non_linear_app (T,w, dataset,question={}):
    
    global face_val_debt
    global sigma_E
    global r
    global E
    
    if question=='b':

        for i in tqdm(range(w,len(dataset))):
            if dataset.Data[i]==(dataset.Data[i] + BMonthEnd(0)):
                print(dataset.Data[i])

                sigma_E=dataset['Vassalou_sig_e'][i]
                face_val_debt=dataset["F"][i]
                r=dataset["r"][i]
                E=dataset["E"][i]

                #Solver for Asset value and sigma_V - non linear system of equation
                V, sigma_V = fsolve(non_linear_equations, (E+face_val_debt,sigma_E*E/(E+face_val_debt)),xtol=1e-3) #As  Bharath and Shumway (2008, Page 1345)

                dataset.loc[i, 'sigma_V'] = sigma_V
                dataset.loc[i, 'V'] = V
                DD_b=(np.log(V/face_val_debt) + (r - 0.5*sigma_V**2)*T)/(sigma_V*np.sqrt(T))
                dataset.loc[i, 'DD_b']=DD_b
                dataset.loc[i, 'Merton_Prob_b']=norm.cdf(-DD_b)

    elif question=='c':

        for i in tqdm(range(w,len(dataset))):
            if dataset.Data[i]==(dataset.Data[i] + BMonthEnd(0)):
                print(dataset.Data[i])

                sigma_E=dataset['ImpVolE'][i]/100
                face_val_debt=dataset["F"][i]
                r=dataset["r"][i]
                E=dataset["E"][i]

                #Solver for Asset value and sigma_V - non linear system of equation
                V, sigma_V = fsolve(non_linear_equations, (E+face_val_debt,sigma_E*E/(E+face_val_debt)),xtol=1e-3) #As  Bharath and Shumway (2008, Page 1345)

                dataset.loc[i, 'sigma_V'] = sigma_V
                dataset.loc[i, 'V'] = V
                DD_c=(np.log(V/face_val_debt) + (r - 0.5*sigma_V**2)*T)/(sigma_V*np.sqrt(T))
                dataset.loc[i, 'DD_c']=DD_c

                dataset.loc[i, 'Merton_Prob_c']=norm.cdf(-DD_c)

# Question 2 -b) and c)

T=1 #Time horizon - 1 year
w=252 #Rolling window size
non_linear_app (T,w, dataset,question='b')

non_linear_app (T,w, dataset,question='c')


# %% Function to Bharath and Shumway (2008, Equation 2 and 3)

def equations (x):
    V=x
    #d1 = (np.log(V* np.exp(r*T)/face_val_debt) + (r + 0.5*sigma_V**2)*T)/(sigma_V*np.sqrt(T))
    d1 = (np.log(V/face_val_debt) + (r + 0.5*sigma_V**2)*T)/(sigma_V*np.sqrt(T))
    d2 = d1 - sigma_V*np.sqrt(T)
    f1 = E - V*norm.cdf(d1) + np.exp(-r*T)*face_val_debt*norm.cdf(d2)

    return f1


#%% Function for Question 2 d) and e)

def iterative_app (T,w, dataset,initial_guess={},question={}):
    global sigma_V
    global face_val_debt
    global r
    global E

    vector_V=np.zeros((w+1,1)) #Initialize vector firm asset value - V

    for i in tqdm(range(w,len(dataset))):
        if dataset.Data[i]==(dataset.Data[i] + BMonthEnd(0)):
            print(dataset.Data[i])

            if initial_guess==0:
                # Set initial guess for firm value volatily equal to the equity std [Vassalou and Xing (2004, Page 835)]
                sigma_V = dataset['Vassalou_sig_e'][i]
            elif initial_guess==1:
                # OR - Set initial guess for firm value volatily equal to std of the asset price vector V [Löffler and Posh (2011, Chapter 2)]]
                sigma_V = dataset['LofflerPosch_sig_v'][i]

            elif initial_guess==2:
                # OR - Set initial guess for firm value volatily equal to std of the asset price vector V [Bharath (20008, Chapter 2.1)]]
                sigma_V = dataset['BharatShumway_sig_v'][i]
            
            num_it=0
            while num_it < 10:
                num_it += 1
                
                # Loop over to calculate Va using initial guess for sigma_a
                for k in range(w+1):
                    
                    face_val_debt=dataset["F"][i-k]
                    r=dataset["r"][i-k]
                    E=dataset["E"][i-k]

                    #Solver for Asset value. (need solver because N(d1) and N(d2))
                    vector_V[k] = fsolve(equations, (E+face_val_debt))

                last_sigma_V = sigma_V
                vector_V_ret=np.log(vector_V/np.roll(vector_V,1))
                vector_V_ret[0] = np.nan
                sigma_V=np.nanstd(vector_V_ret)*np.sqrt(w)
                
                if abs(last_sigma_V - sigma_V) <= 1e-4:
                    #print(num_it)
                    break

            if question=='d':

                dataset.loc[i, 'sigma_V'] = sigma_V
                dataset.loc[i, 'V'] = vector_V[0]
                DD_d=(np.log(vector_V[0]/dataset["F"][i]) + (dataset["r"][i] - 0.5*sigma_V**2)*T)/(sigma_V*np.sqrt(T))
                dataset.loc[i, 'DD_d']=DD_d
                dataset.loc[i, 'Merton_Prob_d']=norm.cdf(-DD_d)

            elif question =='e':
                
                dataset.loc[i, 'sigma_V'] = sigma_V
                dataset.loc[i, 'V'] = vector_V[0]
                drift=(np.nanmean(vector_V_ret,axis=0))*w #with window
                #drift=(np.nanmean(vector_V_ret,axis=0)) #w/o window
                DD_e=(np.log(vector_V[0]/dataset["F"][i]) + (drift - 0.5*sigma_V**2)*T)/(sigma_V*np.sqrt(T)) #Bharath and Shumway (2008, Equations 6)
                dataset.loc[i, 'DD_e']=DD_e
                dataset.loc[i, 'Merton_Prob_e']=norm.cdf(-DD_e) #Bharath and Shumway (2008, Equations 7)
                dataset.loc[i, 'Drift']=drift


# %% Question 2 -d) and e)

T=1 #Time horizon - 1 year
w=252 #Rolling window size

# ierative approach - 2 d) - Initial Guess via Vassalou (0), Loffler Posch (1) or Bharath (2)
iterative_app (T,w, dataset,initial_guess=1,question='d')

# Iterative approach - 2 e) - Initial Guess via Vassalou (0), Loffler Posch (1) or Bharath (2)
iterative_app (T,w, dataset,initial_guess=1,question='e')


#%% Fuctio to calculate distance to default (DD) and the probability of default at the end of 
# each month using the naïve approach - (Bharath and Shumway (2008)


def naive_approach (T,w, dataset):
    
    
    dataset['sigma_V_naive']=pd.DataFrame(np.zeros((len(dataset),1))) #Initialize vector firm asset volatility -sigma_V
    #dataset['V_naive']=pd.DataFrame(np.zeros((len(dataset),1))) #Initialize vector firm asset volatility -sigma_V

    #dataset['DD_naive']=pd.DataFrame(np.zeros((len(dataset),1))) #Initialize vector Distance to Default - DD - question f)
    #dataset['Merton_Prob_naive']=pd.DataFrame(np.zeros((len(dataset),1))) #Initialize vectorMerton Prob to Default - pi_merton - question f)

    #Temporary column with log returns
    #dataset['last_year_ret_PX']=(np.log(dataset.PX) - np.log(dataset.PX.shift(1))).rolling(w).mean()*w
    dataset['last_year_ret_PX']=(np.log(dataset.PX) - np.log(dataset.PX.shift(1))).rolling(w).mean()

    for i in tqdm(range(w,len(dataset))):
        if dataset.Data[i]==(dataset.Data[i] + BMonthEnd(0)):
            print(dataset.Data[i])

            sigma_E=dataset['Vassalou_sig_e'][i]
            naive_D=dataset["F"][i]
            E=dataset["E"][i]
            naive_sigma_D=0.05+0.25*sigma_E
            naive_sigma_V=(E/(E+naive_D))*sigma_E+(naive_D/(E+naive_D))*naive_sigma_D
            naive_ret=dataset['last_year_ret_PX'][i]

            naive_DD=(np.log((E+naive_D)/naive_D)+(naive_ret-0.5*naive_sigma_V**2)*T)/(naive_sigma_V*np.sqrt(T))
            dataset.loc[i, 'DD_naive']=naive_DD             #Bharath and Shumway (2008, Equations 12)
            dataset.loc[i, 'Merton_Prob_naiveDD']=norm.cdf(-naive_DD) #Bharath and Shumway (2008, Equations 13)

    dataset.drop('last_year_ret_PX', axis=1, inplace=True)   #- apagar

#Run Naive approach - 2 f)
naive_approach (T,w, dataset)

#%% Plot analysis - 2 g)
import matplotlib.dates as mdates

#Chart to visualize the comparison between multiple models, using twinx to plot on opposite axis Drift (μ),
#  NI/TA (Net income / Total Assets) or r (1 year T-Bill) in the selected period

#Selected Period
#Start and End year
start="2010"
end="2019"

# Plot 2 lines with differents scales on same chart
fig = plt.figure(figsize=(12, 8))

line_weight = 3
alpha = .5
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0, 0, 1, 1])

# Twinx that joins the x-axis
ax2 = ax1.twinx()

#AXIS 1
#Model from 2 - b)
lns1 = ax1.plot(dataset.set_index('Data')['Merton_Prob_b'].dropna()[start:end], color='blue', \
    lw=line_weight, alpha=alpha, label='Merton Prob b)')#.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
 
#Model from 2 - c)
lns2= ax1.plot(dataset.set_index('Data')['Merton_Prob_c'].dropna()[start:end], color='green', \
    lw=line_weight, alpha=alpha, label='Merton Prob c)')#.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 

#Model from 2 - d)
lns3= ax1.plot(dataset.set_index('Data')['Merton_Prob_d'].dropna()[start:end], color='black', \
    lw=line_weight, alpha=alpha, label='Merton Prob d)')#.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
 
""" #Model from 2 - e)
lns4= ax1.plot(dataset.set_index('Data')['Merton_Prob_e'].dropna()[start:end], color='brown', \
    lw=line_weight, alpha=alpha, label='Merton Prob e)')#.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
 """
#Model from 2 - f)
lns5= ax1.plot(dataset.set_index('Data')['Merton_Prob_naiveDD'].dropna()[start:end], color='red', \
    lw=line_weight, alpha=alpha, label='Naive Probabilitie f)')#.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

#Compare with Drift, μ or NI/TA
#AXIS 2
#lns6 = ax2.plot(dataset.set_index('Data')['Drift'].dropna()[start:end], color='orange', lw=line_weight, alpha=alpha, label='Drift (Pts)')
#lns6 = ax2.plot(dataset.set_index('Data')['NITA'].dropna()[start:end], color='orange', lw=line_weight, alpha=alpha, label='NI/TA (%)')
#lns6 = ax2.plot(dataset.set_index('Data')['F'].dropna()[start:end], color='orange', lw=line_weight, alpha=alpha, label='Face Value of Debt (Millions of €')

# Solution for having multiple legends
leg = lns1 + lns2+ lns3+lns5#+ lns6
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=0)

#Vertican time span
ax1.axvspan(datetime(2014,1,1), datetime(2014,6,1),  alpha=0.5)
#plt.title('All Models vs Drift', fontsize=20)
plt.show()



#%% Aux charts

dataset.set_index('Data')[['Merton_Prob_b','Merton_Prob_c','Merton_Prob_d','Merton_Prob_naiveDD','Drift']].dropna()[start:end].plot(grid=True,figsize=(20, 10))

dataset.set_index('Data')[['Merton_Prob_d','Merton_Prob_e','Merton_Prob_naiveDD']].dropna()['2010':'2019'].plot(grid=True,figsize=(20, 10))#.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

dataset.set_index('Data')[['Merton_Prob_d','Merton_Prob_e']].dropna()['2016':'2019'].plot(grid=True,figsize=(20, 10))


#%% Aux calculations

#NI/TA ratio
dataset.loc[:,'NITA']=aux['Net Income']/dataset['V']