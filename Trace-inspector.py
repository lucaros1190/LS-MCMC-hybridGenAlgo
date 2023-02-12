
# Script to inspect the traces after the MCMC process.
# Created by Luca Rossini
# e-mail: luca.rossini@unitus.it
# Last update 23 May 2022


# List of import

from Parameters import *
from ODEdroso import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# Import the temperature data and field monitoring

data_exp = pd.read_csv("Perturbata.csv", sep = ",", header = 0)
data_exp.columns = ["Day", "Temperature", "Ad_male"]

# Replace NaN values to plot correctly the experimental data

data_exp = data_exp.replace('', np.nan)

# Retrieve the time vector from experimental data

day_ob = data_exp["Day"]
day_obs = day_ob.to_numpy()

# Retrieve the daily temperature array

Temp_data = data_exp['Temperature']
Temp = Temp_data.to_numpy()

# Retrieve monitoring data

yob = data_exp['Ad_male']
yobs = yob.to_numpy()


# Import the traces files

Df_Traces_a_EP = pd.read_csv("./MCMC-Res/Trace_a_EP-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_a_PA = pd.read_csv("./MCMC-Res/Trace_a_PA-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_a_Sur = pd.read_csv("./MCMC-Res/Trace_a_Sur-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_T_L_EP = pd.read_csv("./MCMC-Res/Trace_T_L_EP-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_T_L_PA = pd.read_csv("./MCMC-Res/Trace_T_L_PA-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_T_L_Sur = pd.read_csv("./MCMC-Res/Trace_T_L_Sur-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_T_M_EP = pd.read_csv("./MCMC-Res/Trace_T_M_EP-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_T_M_PA = pd.read_csv("./MCMC-Res/Trace_T_M_PA-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_T_M_Sur = pd.read_csv("./MCMC-Res/Trace_T_M_Sur-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_m_EP = pd.read_csv("./MCMC-Res/Trace_m_EP-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_m_PA = pd.read_csv("./MCMC-Res/Trace_m_PA-MCMC.csv", sep = ";", header = 1).transpose()

Df_Traces_m_Sur = pd.read_csv("./MCMC-Res/Trace_m_Sur-MCMC.csv", sep = ";", header = 1).transpose()


# Assign the traces files to numpy matrixes

Traces_a_EP = Df_Traces_a_EP.to_numpy()
Traces_a_PA = Df_Traces_a_PA.to_numpy()
Traces_a_Sur = Df_Traces_a_Sur.to_numpy()

Traces_T_L_EP = Df_Traces_T_L_EP.to_numpy()
Traces_T_L_PA = Df_Traces_T_L_PA.to_numpy()
Traces_T_L_Sur = Df_Traces_T_L_Sur.to_numpy()

Traces_T_M_EP = Df_Traces_T_M_EP.to_numpy()
Traces_T_M_PA = Df_Traces_T_M_PA.to_numpy()
Traces_T_M_Sur = Df_Traces_T_M_Sur.to_numpy()

Traces_m_EP = Df_Traces_m_EP.to_numpy()
Traces_m_PA = Df_Traces_m_PA.to_numpy()
Traces_m_Sur = Df_Traces_m_Sur.to_numpy()


# Get the number of columns

ColNum = Traces_a_EP.transpose().shape[1] - 1

# Set accepted chains counter

Counter = 0

# New dta frames for good and bad chains

GoodChain_a_EP = pd.DataFrame()
GoodChain_a_PA = pd.DataFrame()
GoodChain_a_Sur = pd.DataFrame()

GoodChain_T_L_EP = pd.DataFrame()
GoodChain_T_L_PA = pd.DataFrame()
GoodChain_T_L_Sur = pd.DataFrame()

GoodChain_T_M_EP = pd.DataFrame()
GoodChain_T_M_PA = pd.DataFrame()
GoodChain_T_M_Sur = pd.DataFrame()

GoodChain_m_EP = pd.DataFrame()
GoodChain_m_PA = pd.DataFrame()
GoodChain_m_Sur = pd.DataFrame()

BadChain_a_EP = pd.DataFrame()
BadChain_a_PA = pd.DataFrame()
BadChain_a_Sur = pd.DataFrame()

BadChain_T_L_EP = pd.DataFrame()
BadChain_T_L_PA = pd.DataFrame()
BadChain_T_L_Sur = pd.DataFrame()

BadChain_T_M_EP = pd.DataFrame()
BadChain_T_M_PA = pd.DataFrame()
BadChain_T_M_Sur = pd.DataFrame()

BadChain_m_EP = pd.DataFrame()
BadChain_m_PA = pd.DataFrame()
BadChain_m_Sur = pd.DataFrame()

# For loop to inspect the traces

for i in range(1, ColNum):

    # Calculate the average of the traces for a and m values

    a_EP_BestMCMC = np.mean(Traces_a_EP[i])
    a_PA_BestMCMC = np.mean(Traces_a_PA[i])
    a_Sur_BestMCMC = np.mean(Traces_a_Sur[i])

    T_L_EP_BestMCMC = np.mean(Traces_T_L_EP[i])
    T_L_PA_BestMCMC = np.mean(Traces_T_L_PA[i])
    T_L_Sur_BestMCMC = np.mean(Traces_T_L_Sur[i])

    T_M_EP_BestMCMC = np.mean(Traces_T_M_EP[i])
    T_M_PA_BestMCMC = np.mean(Traces_T_M_PA[i])
    T_M_Sur_BestMCMC = np.mean(Traces_T_M_Sur[i])

    m_EP_BestMCMC = np.mean(Traces_m_EP[i])
    m_PA_BestMCMC = np.mean(Traces_m_PA[i])
    m_Sur_BestMCMC = np.mean(Traces_m_Sur[i])

    # Set the chain name

    ChainName = ['Chain_{}'.format(i)]

    # Set the parameters to plot the different solutions

    BriPar_EP_StepMCMC = a_EP_BestMCMC, T_L_EP_BestMCMC, T_M_EP_BestMCMC, m_EP_BestMCMC
    BriPar_PA_StepMCMC = a_PA_BestMCMC, T_L_PA_BestMCMC, T_M_PA_BestMCMC, m_PA_BestMCMC
    BriPar_Sur_StepMCMC = a_Sur_BestMCMC, T_L_Sur_BestMCMC, T_M_Sur_BestMCMC, m_Sur_BestMCMC

    # Solve the equation

    y_malesMCMC = EqSolver(day_obs, InitCond_Acquired, BriPar_EP_StepMCMC, BriPar_PA_StepMCMC, BriPar_Sur_StepMCMC, FertPar_Acquired, MortPar_Acquired, SR, Temp)

    # Plot the chain

    plt.figure(1)

    plt.plot(day_obs, y_malesMCMC, color="C1", label=f"Simulated Adult males - MCMC")
    plt.scatter(day_obs, yobs, color="C2", label=f"Experimental Adult males")
    plt.xlabel('Time (days)')
    plt.ylabel('Population density')
    plt.legend()

    plt.show()

    # if / else statement to build the new database with the good traces

    print('\nChain number', i, '- do you want to accept this trace? y/n\n')

    Ans = input()

    if Ans == 'y':

        GoodChain_a_EP = GoodChain_a_EP.append([ChainName + Traces_a_EP[i].tolist()])
        GoodChain_a_PA = GoodChain_a_PA.append([ChainName + Traces_a_PA[i].tolist()])
        GoodChain_a_Sur = GoodChain_a_Sur.append([ChainName + Traces_a_Sur[i].tolist()])

        GoodChain_T_L_EP = GoodChain_T_L_EP.append([ChainName + Traces_T_L_EP[i].tolist()])
        GoodChain_T_L_PA = GoodChain_T_L_PA.append([ChainName + Traces_T_L_PA[i].tolist()])
        GoodChain_T_L_Sur = GoodChain_T_L_Sur.append([ChainName + Traces_T_L_Sur[i].tolist()])

        GoodChain_T_M_EP = GoodChain_T_M_EP.append([ChainName + Traces_T_M_EP[i].tolist()])
        GoodChain_T_M_PA = GoodChain_T_M_PA.append([ChainName + Traces_T_M_PA[i].tolist()])
        GoodChain_T_M_Sur = GoodChain_T_M_Sur.append([ChainName + Traces_T_M_Sur[i].tolist()])

        GoodChain_m_EP = GoodChain_m_EP.append([ChainName + Traces_m_EP[i].tolist()])
        GoodChain_m_PA = GoodChain_m_PA.append([ChainName + Traces_m_PA[i].tolist()])
        GoodChain_m_Sur = GoodChain_m_Sur.append([ChainName + Traces_m_Sur[i].tolist()])

        Counter = Counter + 1

    else:
    
        BadChain_a_EP = BadChain_a_EP.append([ChainName + Traces_a_EP[i].tolist()])
        BadChain_a_PA = BadChain_a_PA.append([ChainName + Traces_a_PA[i].tolist()])
        BadChain_a_Sur = BadChain_a_Sur.append([ChainName + Traces_a_Sur[i].tolist()])

        BadChain_T_L_EP = BadChain_T_L_EP.append([ChainName + Traces_T_L_EP[i].tolist()])
        BadChain_T_L_PA = BadChain_T_L_PA.append([ChainName + Traces_T_L_PA[i].tolist()])
        BadChain_T_L_Sur = BadChain_T_L_Sur.append([ChainName + Traces_T_L_Sur[i].tolist()])

        BadChain_T_M_EP = BadChain_a_EP.append([ChainName + Traces_T_M_EP[i].tolist()])
        BadChain_T_M_PA = BadChain_a_PA.append([ChainName + Traces_T_M_PA[i].tolist()])
        BadChain_T_M_Sur = BadChain_a_Sur.append([ChainName + Traces_T_M_Sur[i].tolist()])

        BadChain_m_EP = BadChain_m_EP.append([ChainName + Traces_m_EP[i].tolist()])
        BadChain_m_PA = BadChain_m_PA.append([ChainName + Traces_m_PA[i].tolist()])
        BadChain_m_Sur = BadChain_m_Sur.append([ChainName + Traces_m_Sur[i].tolist()])


# Order the data frames

GoodChain_a_EP = GoodChain_a_EP.transpose()
GoodChain_a_PA = GoodChain_a_PA.transpose()
GoodChain_a_Sur = GoodChain_a_Sur.transpose()

GoodChain_T_L_EP = GoodChain_T_L_EP.transpose()
GoodChain_T_L_PA = GoodChain_T_L_PA.transpose()
GoodChain_T_L_Sur = GoodChain_T_L_Sur.transpose()

GoodChain_T_M_EP = GoodChain_T_M_EP.transpose()
GoodChain_T_M_PA = GoodChain_T_M_PA.transpose()
GoodChain_T_M_Sur = GoodChain_T_M_Sur.transpose()

GoodChain_m_EP = GoodChain_m_EP.transpose()
GoodChain_m_PA = GoodChain_m_PA.transpose()
GoodChain_m_Sur = GoodChain_m_Sur.transpose()

BadChain_a_EP = BadChain_a_EP.transpose()
BadChain_a_PA = BadChain_a_PA.transpose()
BadChain_a_Sur = BadChain_a_Sur.transpose()

BadChain_T_L_EP = BadChain_T_L_EP.transpose()
BadChain_T_L_PA = BadChain_T_L_PA.transpose()
BadChain_T_L_Sur = BadChain_T_L_Sur.transpose()

BadChain_T_M_EP = BadChain_T_M_EP.transpose()
BadChain_T_M_PA = BadChain_T_M_PA.transpose()
BadChain_T_M_Sur = BadChain_T_M_Sur.transpose()

BadChain_m_EP = BadChain_m_EP.transpose()
BadChain_m_PA = BadChain_m_PA.transpose()
BadChain_m_Sur = BadChain_m_Sur.transpose()

# Save the data frames in .csv files

GoodChain_a_EP.to_csv('./MCMC-Res/SelectedChains/GoodTrace_a_EP-MCMC.csv', header = 0)
GoodChain_a_PA.to_csv('./MCMC-Res/SelectedChains/GoodTrace_a_PA-MCMC.csv', header = 0)
GoodChain_a_Sur.to_csv('./MCMC-Res/SelectedChains/GoodTrace_a_Sur-MCMC.csv', header = 0)

GoodChain_T_L_EP.to_csv('./MCMC-Res/SelectedChains/GoodTrace_T_L_EP-MCMC.csv', header = 0)
GoodChain_T_L_PA.to_csv('./MCMC-Res/SelectedChains/GoodTrace_T_L_PA-MCMC.csv', header = 0)
GoodChain_T_L_Sur.to_csv('./MCMC-Res/SelectedChains/GoodTrace_T_L_Sur-MCMC.csv', header = 0)

GoodChain_T_M_EP.to_csv('./MCMC-Res/SelectedChains/GoodTrace_T_M_EP-MCMC.csv', header = 0)
GoodChain_T_M_PA.to_csv('./MCMC-Res/SelectedChains/GoodTrace_T_M_PA-MCMC.csv', header = 0)
GoodChain_T_M_Sur.to_csv('./MCMC-Res/SelectedChains/GoodTrace_T_M_Sur-MCMC.csv', header = 0)

GoodChain_m_EP.to_csv('./MCMC-Res/SelectedChains/GoodTrace_m_EP-MCMC.csv', header = 0)
GoodChain_m_PA.to_csv('./MCMC-Res/SelectedChains/GoodTrace_m_PA-MCMC.csv', header = 0)
GoodChain_m_Sur.to_csv('./MCMC-Res/SelectedChains/GoodTrace_m_Sur-MCMC.csv', header = 0)

BadChain_a_EP.to_csv('./MCMC-Res/SelectedChains/BadTrace_a_EP-MCMC.csv', header = 0)
BadChain_a_PA.to_csv('./MCMC-Res/SelectedChains/BadTrace_a_PA-MCMC.csv', header = 0)
BadChain_a_Sur.to_csv('./MCMC-Res/SelectedChains/BadTrace_a_Sur-MCMC.csv', header = 0)

BadChain_T_L_EP.to_csv('./MCMC-Res/SelectedChains/BadTrace_T_L_EP-MCMC.csv', header = 0)
BadChain_T_L_PA.to_csv('./MCMC-Res/SelectedChains/BadTrace_T_L_PA-MCMC.csv', header = 0)
BadChain_T_L_Sur.to_csv('./MCMC-Res/SelectedChains/BadTrace_T_L_Sur-MCMC.csv', header = 0)

BadChain_T_M_EP.to_csv('./MCMC-Res/SelectedChains/BadTrace_T_M_EP-MCMC.csv', header = 0)
BadChain_T_M_PA.to_csv('./MCMC-Res/SelectedChains/BadTrace_T_M_PA-MCMC.csv', header = 0)
BadChain_T_M_Sur.to_csv('./MCMC-Res/SelectedChains/BadTrace_T_M_Sur-MCMC.csv', header = 0)

BadChain_m_EP.to_csv('./MCMC-Res/SelectedChains/BadTrace_m_EP-MCMC.csv', header = 0)
BadChain_m_PA.to_csv('./MCMC-Res/SelectedChains/BadTrace_m_PA-MCMC.csv', header = 0)
BadChain_m_Sur.to_csv('./MCMC-Res/SelectedChains/BadTrace_m_Sur-MCMC.csv', header = 0)

# Print final messages on the shell

print('\n    Finish! \n')
print('\n    Accepted chains =', Counter, '\n')
print('\n    Rejected chains =', ColNum - 1 - Counter, '\n')





