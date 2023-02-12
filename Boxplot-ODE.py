
# Script to prepare the dataset for boxplot
# Created by Luca Rossini
# e-mail: luca.rossini@unitus.it
# Last update 4 January 2023


# List of import

from Parameters import *
from ODEdroso import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn


# Import the temperature data and field monitoring

data_exp = pd.read_csv("Experimental_data-Campo1-Gen1.csv", sep = ",", header = 0)
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

Df_Traces_a_EP = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_a_EP-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_a_PA = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_a_PA-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_a_Sur = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_a_Sur-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_T_L_EP = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_T_L_EP-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_T_L_PA = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_T_L_PA-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_T_L_Sur = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_T_L_Sur-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_T_M_EP = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_T_M_EP-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_T_M_PA = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_T_M_PA-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_T_M_Sur = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_T_M_Sur-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_m_EP = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_m_EP-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_m_PA = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_m_PA-MCMC.csv", sep = ",", header = 1).transpose()

Df_Traces_m_Sur = pd.read_csv("./MCMC-Res/SelectedChains/GoodTrace_m_Sur-MCMC.csv", sep = ",", header = 1).transpose()


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

ColNum =  Traces_a_EP.transpose().shape[1] - 1


# Prepare the data frame to store numerical solutions

BoxPlot_DfMatrix = pd.DataFrame()
BoxPlot_DictMatrix = {}
BoxPlot_DictMatrix['Day'] = day_obs


# Prepare the array to store days and numerical solutions

PlotIndex = []
XLabels = []


# For loop to solve the equations

for i in range(1, ColNum):

    print('\n Solving equation number:', i)

    # Calculate the average of the traces for the Briere parameters

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

    # Set the parameters to plot the different solutions

    BriPar_EP_StepMCMC = a_EP_BestMCMC, T_L_EP_BestMCMC, T_M_EP_BestMCMC, m_EP_BestMCMC
    BriPar_PA_StepMCMC = a_PA_BestMCMC, T_L_PA_BestMCMC, T_M_PA_BestMCMC, m_PA_BestMCMC
    BriPar_Sur_StepMCMC = a_Sur_BestMCMC, T_L_Sur_BestMCMC, T_M_Sur_BestMCMC, m_Sur_BestMCMC

    # Solve the equation

    y_malesMCMC = EqSolver(day_obs, InitCond_Acquired, BriPar_EP_StepMCMC, BriPar_PA_StepMCMC, BriPar_Sur_StepMCMC, FertPar_Acquired, MortPar_Acquired, SR, Temp)
    
    # Append the model solutions to the new database
    
    StringName = 'Chain_{}'.format(i)

    if StringName not in BoxPlot_DictMatrix:

        BoxPlot_DictMatrix[StringName] = {}

    BoxPlot_DictMatrix[StringName] = y_malesMCMC
    

# Add all to the data frame before to generate the .csv file

BoxPlot_DfMatrix = BoxPlot_DfMatrix.from_dict(BoxPlot_DictMatrix, orient = 'columns')


# Save the data frames in .csv files

BoxPlot_DfMatrix.to_csv('./MCMC-Res/BoxPlot-DfMatrix.csv', header = 1, sep = ';', index = False)


# Make the boxplot

for i in range(0, len(day_obs)):

    PlotIndex = np.append(PlotIndex, i)
    
    if i == 0:
        XLabels = np.append(XLabels, ' ')
    
    elif i % 15 == 0:
        XLabels = np.append(XLabels, i)
    
    else:
        XLabels = np.append(XLabels, ' ')

plt.figure(1)

BoxPlot_DfMatrix = BoxPlot_DfMatrix.set_index('Day')
BoxPlot_DfMatrix = BoxPlot_DfMatrix.transpose()

boxplot = BoxPlot_DfMatrix.boxplot(column = PlotIndex.tolist(), grid = False)
plt.ylim(ymin = 0, ymax=5000)
plt.scatter(day_obs, yobs, c = 'red')
boxplot.set_xticklabels(XLabels.tolist())


# Make the plot with confidence intervals

y_Mean = []
y_CI = []

for i in range(0, len(day_obs)):

    # Compute the mean values
    
    TempConf_Mean = np.mean(BoxPlot_DfMatrix[i])
    y_Mean = np.append(y_Mean, TempConf_Mean)

    # Compute the confidence interval
    
    TempConf_CI = 1.96 * np.std(BoxPlot_DfMatrix[i]) / np.sqrt(len(BoxPlot_DfMatrix[i]))
    y_CI = np.append(y_CI, TempConf_CI)
    
fig, PlotCI = plt.subplots()
PlotCI.plot(day_obs, y_Mean, label = 'MCMC best fit')
PlotCI.scatter(day_obs, yobs, c = 'red', label = 'Field data')
PlotCI.fill_between(day_obs, y_Mean - y_CI, y_Mean + y_CI, color = 'b', alpha = .1)
plt.xlabel('Time (days)')
plt.ylabel('Number of adult males')
plt.ylim(ymin = 0, ymax=200)
plt.legend()

plt.show()


# Print final messages on the shell

print('\n    Finish! \n')





