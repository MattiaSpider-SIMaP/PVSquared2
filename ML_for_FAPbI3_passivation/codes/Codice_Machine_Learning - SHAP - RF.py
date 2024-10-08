# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:26:25 2023

@author: Mattia Ragni
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

plt.rcParams['figure.dpi'] = 300

TableName = "/home/mattia/Perovskite_passivation_project/PVSquared2/ML_for_FAPbI3_passivation/DATA/1&2&5-Rev&For-Clean(Voc-1000)_ML-means.txt"
pd.set_option('display.max_rows', None, 'display.max_columns', None)

# The molecular features & processing conditions are loaded as X
# All
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(°C)", "Molecular_Weight(u)", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Heavy_Atom_Count", "Complexity", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio", "Heteroatom_count"]
# SignificativeOLD (no heteroatom count)
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(°C)", "Molecular_Weight(u)", "Heavy_Atom_Count", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC)
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(°C)", "Molecular_Weight(u)", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC & H_count & C_count)
ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(°C)", "Molecular_Weight(u)", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC & H_count & C_count & Annealing) 
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Molecular_Weight(u)", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC & H_count & C_count & Annealing & Concentration) 
# ProcessingData_cols = ["DeviceName", "Molecular_Weight(u)", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# No Atom_count
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(°C)", "Molecular_Weight(u)", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]

# Load the table with full data
Full_Data = pd.read_csv(TableName, delim_whitespace=True, encoding='utf-8')

# # Check if the data is loaded and separated correctly
# print(Full_Data)

# Creo il dataframe con solo i valori di processazione e proprietà fisico-chimiche (input)
X = Full_Data[ProcessingData_cols]

# Creo il dataframe con solo i valori di Voc (output)
Y_Voc = Full_Data["Voc(mV)"]

# # Check if the data is loaded and separated correctly
# pd.set_option('display.max_rows', 10, 'display.max_columns', 5)
# print(X, Y_Voc)

print()

print("Dataframe caricati correttamente...")

print()
print("######################################################################")
print()

########################################################################################################################################################################################################

# cor = X.corr()
# mask = np.zeros_like(cor)
# plt.rcParams['font.family'] = "Arial"
# mask[np.triu_indices_from(mask)] = True

# with sns.axes_style("white"):
#     f, ax = plt.subplots(figsize=(12, 12))  # Imposta la dimensione della figura a 12x12
#     ax = sns.heatmap(cor, annot=True, annot_kws={"size": 16}, fmt=".2f",
#                      cmap=plt.cm.RdBu_r, linewidths=0.5, vmin=-1, vmax=1, mask=mask, square=True,
#                      cbar_kws={"shrink": 0.8225})  # Imposta la dimensione della barra laterale

#     # Riduci la dimensione delle etichette dell'asse x e dell'asse y
#     ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
#     ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

#     cbar = ax.collections[0].colorbar  # Ottieni l'asse della colorbar
#     cbar.ax.yaxis.set_tick_params(labelsize=16)  # Imposta la dimensione delle etichette della colorbar

# plt.show()

# print("Pearson Correlation Values calcolati...")
# print()
# print("######################################################################")

########################################################################################################################################################################################################

# Prepariamo training e data set con una percentuale a scelta
from sklearn.model_selection import train_test_split

#Qui inizialmente estraevo i valori di Voc e poi svolgevo su X e Y la suddivisione in train e test set
X_train, X_test, Y_Voc_train, Y_Voc_test = train_test_split(X, Y_Voc, test_size=0.2, random_state=42) # Test set is 20% of the total data, and the random state is set to ensure identical results for each run
print("Train e data sets suddivisi casualmente pronti...")
print()

# Visto che voglio suddividerli in base al DeviceName, perché non svolgere prima la suddivisione sulla tabella completa Y per poi ottenere la tabella con le processing data e i valori di Voc? 
# Seleziona la colonna "DeviceName" dalla tabella "Full_Data"
# device_names = Full_Data["DeviceName"]

# # Stampa delle righe in cui compare la stringa nella colonna 'Nome'
# # stringa_cercata = 'MEPEAI'
# # risultato = Full_Data[Full_Data["DeviceName"].str.contains(stringa_cercata)]
# # print(risultato)

# Suddivisione tenendo conto di Concentration e annealing 
# device_initials = [ 'MEPEAI-5mM-20T', 
#                     'MEPEAI-5mM-100T', 
#                     'MEPEAI-10mM-20T', 
#                     'MEPEAI-10mM-100T', 
#                     'MEPEAI-15mM-20T', 
#                     'MEPEAI-15mM-100T', 
#                     'MEPEABr-5mM-20T', 
#                     'MEPEABr-5mM-100T', 
#                     'MEPEABr-10mM-20T', 
#                     'MEPEABr-10mM-100T', 
#                     'MEPEABr-15mM-20T', 
#                     'MEPEABr-15mM-100T', 
#                     'MEPEACl-5mM-20T', 
#                     'MEPEACl-5mM-100T', 
#                     'MEPEACl-10mM-20T', 
#                     'MEPEACl-10mM-100T', 
#                     'MEPEACl-15mM-20T', 
#                     'MEPEACl-15mM-100T', 
#                     'n-BAI-5mM-20T', 
#                     'n-BAI-5mM-100T', 
#                     'n-BAI-10mM-20T', 
#                     'n-BAI-10mM-100T', 
#                     'n-BAI-15mM-20T', 
#                     'n-BAI-15mM-100T', 
#                     'iso-BAI-5mM-20T', 
#                     'iso-BAI-5mM-100T', 
#                     'iso-BAI-10mM-20T', 
#                     'iso-BAI-10mM-100T', 
#                     'iso-BAI-15mM-20T', 
#                     'iso-BAI-15mM-100T', 
#                     'n-OAI-5mM-20T', 
#                     'n-OAI-5mM-100T', 
#                     'n-OAI-10mM-20T', 
#                     'n-OAI-10mM-100T', 
#                     'n-OAI-15mM-20T', 
#                     'n-OAI-15mM-100T',      
#                     'BBr-5mM-20T', 
#                     'BBr-5mM-100T', 
#                     'BBr-10mM-20T', 
#                     'BBr-10mM-100T', 
#                     'BBr-15mM-20T', 
#                     'BBr-15mM-100T', 
#                     'HBr-5mM-20T', 
#                     'HBr-5mM-100T', 
#                     'HBr-10mM-20T', 
#                     'HBr-10mM-100T', 
#                     'HBr-15mM-20T', 
#                     'HBr-15mM-100T',
#                     'OATsO-5mM-20T', 
#                     'OATsO-5mM-100T', 
#                     'OATsO-10mM-20T', 
#                     'OATsO-10mM-100T', 
#                     'OATsO-15mM-20T', 
#                     'OATsO-15mM-100T'
#                     ]

# Only 20T
# device_initials = [ 'MEPEAI-5mM-20T',  
#                     'MEPEAI-10mM-20T',  
#                     'MEPEAI-15mM-20T',  
#                     'MEPEABr-5mM-20T', 
#                     'MEPEABr-10mM-20T',  
#                     'MEPEABr-15mM-20T',  
#                     'MEPEACl-5mM-20T',  
#                     'MEPEACl-10mM-20T',  
#                     'MEPEACl-15mM-20T',  
#                     'n-BAI-5mM-20T',  
#                     'n-BAI-10mM-20T',  
#                     'n-BAI-15mM-20T',  
#                     'iso-BAI-5mM-20T', 
#                     'iso-BAI-10mM-20T', 
#                     'iso-BAI-15mM-20T', 
#                     'n-OAI-5mM-20T', 
#                     'n-OAI-10mM-20T', 
#                     'n-OAI-15mM-20T',   
#                     'BBr-5mM-20T', 
#                     'BBr-10mM-20T', 
#                     'BBr-15mM-20T', 
#                     'HBr-5mM-20T', 
#                     'HBr-10mM-20T',  
#                     'HBr-15mM-20T', 
#                     'OATsO-5mM-20T',  
#                     'OATsO-10mM-20T',  
#                     'OATsO-15mM-20T' 
#                     ]

# # # Only 20T e 10mM
# device_initials = ['MEPEAI-10mM-20T',     
#                     'MEPEABr-10mM-20T',      
#                     'MEPEACl-10mM-20T',     
#                     'n-BAI-10mM-20T',  
#                     'iso-BAI-10mM-20T', 
#                     'n-OAI-10mM-20T', 
#                     'BBr-10mM-20T', 
#                     'HBr-10mM-20T',  
#                     'OATsO-10mM-20T',  
#                     ]

# # # Only 20T e 15mM
# # device_initials = [ 'MEPEAI-15mM-20T',    
# #                     'MEPEABr-15mM-20T',  
# #                     'MEPEACl-15mM-20T',  
# #                     'n-BAI-15mM-20T',  
# #                     'iso-BAI-15mM-20T',  
# #                     'n-OAI-15mM-20T',   
# #                     'BBr-15mM-20T', 
# #                     'HBr-15mM-20T', 
# #                     'OATsO-15mM-20T' 
# #                     ]

# # Trova gli indici per ciascuna sigla di dispositivo nella colonna "DeviceName"
# device_indices = [np.where(device_names.str.startswith(initial))[0] for initial in device_initials]

# # Dividi gli indici dei dati per ciascuna sigla in train e test
# train_indices, test_indices = [], []
# for indices in device_indices:
#     train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
#     train_indices.extend(train_idx)
#     test_indices.extend(test_idx)

# # Converti gli indici in array numpy
# train_indices = np.array(train_indices)
# test_indices = np.array(test_indices)

# # Utilizzo gli indici per ottenere i Full Data di train e test
# Full_train = Full_Data.iloc[train_indices]
# Full_test = Full_Data.iloc[test_indices]

# # print(Full_Data, Full_train, Full_test)
# # print("Full_Data:", Full_Data.shape,"Full_train:", Full_train.shape,"; Full_test:", Full_test.shape)

# # Infine li suddivido in dati di processazione e variabile target -> Voc
# X_train = Full_train[ProcessingData_cols]
# Y_Voc_train = Full_train["Voc(mV)"]
# X_test = Full_test[ProcessingData_cols]
# Y_Voc_test = Full_test["Voc(mV)"]

# # print(X_train, Y_Voc_train, X_test, Y_Voc_test)

# print("X_train:", X_train.shape,"; Y_Voc_train:", Y_Voc_train.shape)
# print("X_test:", X_test.shape,"; Y_Voc_test:", Y_Voc_test.shape)

# # Conteggio per verificare suddivisione 
# for initial in device_initials:
#     train_count = (X_train["DeviceName"].str.startswith(initial)).sum()
#     test_count = (X_test["DeviceName"].str.startswith(initial)).sum()
#     ratio = round(test_count / train_count, 2)
#     print(test_count, train_count, ratio, initial)
# print()    
    
# print("Train e data sets suddivisi in base ai nomi dei device pronti...")
# print()
# print("######################################################################")
# print()

########################################################################################################################################################################################################

#Applichiamo una Standardization così da eliminare il bias legato ad una particolare variabile a causa della sua grandezza numerica.
from sklearn.preprocessing import StandardScaler, RobustScaler

# Decommentare se non si vuole svolgere la standardizzazione
# X_noname = X_stand = X.drop("DeviceName", axis=1)
# X_train_noname = X_train.drop("DeviceName", axis=1)
# X_train_stand_no = X_train.drop("DeviceName", axis=1)
# X_test_stand = X_test.drop("DeviceName", axis=1)

#Standardizzazione valori di input con Standard Scaler
scaler = StandardScaler()
X_noname = X.drop("DeviceName", axis=1)
X_train_noname = X_train.drop("DeviceName", axis=1)
X_test_noname = X_test.drop("DeviceName", axis=1)
scaler.fit(X_noname)

X_stand = scaler.transform(X_noname)
X_train_stand_no = X_train_stand = scaler.transform(X_train_noname)
X_test_stand = scaler.transform(X_test_noname) 

print("Standardizzazione dei valori di input svolta...")
print()
print("######################################################################")
print()

#Shuffle dei dati 
np.random.seed(42)
i_rand = np.arange(X_train_stand_no.shape[0])
np.random.shuffle(i_rand)
X_train_stand = np.array(X_train_stand_no)[i_rand]
Y_Voc_train = np.array(Y_Voc_train)[i_rand]

print("Shuffle dei dati svolto...")
print()
print("######################################################################")
print()

# #Standardizzazione valori di input con Robust Scaler per diminuire influenza degli outliners
# scaler = RobustScaler()

# X_noname = X.drop("DeviceName", axis=1)
# X_train = X_train.drop("DeviceName", axis=1)
# X_test = X_test.drop("DeviceName", axis=1)
# scaler.fit(X_noname)

# X_stand = scaler.transform(X_noname)
# X_train_stand = scaler.transform(X_train)
# X_test_stand = scaler.transform(X_test) 

# print("Standardizzazione robusta dei valori di input svolta...")
# print()

#Visualize the mean and variance prior and after standardization
# fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# plt.style.use('default')
# ax[0].set_title("Mean")
# ax[0].scatter(np.arange(X_train_noname.shape[1]), np.mean(X_train_noname, axis=0), s=10, label='X_train', c="red")
# ax[0].scatter(np.arange(X_train_stand_no.shape[1]), np.mean(X_train_stand_no, axis=0), s=10, label='X_train_stand', c="blue")
# ax[0].legend()

# ax[1].set_title("Variance")
# ax[1].scatter(np.arange(X_train_noname.shape[1]), np.var(X_train_noname, axis=0), s=10, label='X_train', c="red")
# ax[1].scatter(np.arange(X_train_stand_no.shape[1]), np.var(X_train_stand_no, axis=0), s=10, label='X_train_stand', c="blue")
# ax[1].set_yscale('log')
# ax[1].legend()
# plt.show()

########################################################################################################################################################################################################

#Visulization functions
def prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat, model_name):
    from sklearn import metrics
    fontsize = 12
    plt.figure(figsize=(6,5))
    plt.style.use('default')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rcParams['font.family']="Arial"
    a = plt.scatter(y_train, y_train_hat, s=25,c='#b2df8a')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k:', lw=1.5)
    plt.xlabel('Observation', fontsize=fontsize)
    plt.ylabel('Prediction', fontsize=fontsize)
    plt.xticks([900, 950, 1000, 1050, 1100, 1150, 1200])
    plt.yticks([1050, 1100, 1150])
    plt.tick_params(direction='in')
    #plt.text(450,80,'Scaled',family="Arial",fontsize=fontsize)
    plt.xlim([1000,1200]) 
    plt.ylim([1000,1200])
    plt.title('{} - Train RMSE: {:.2e}, Test RMSE: {:.2e}'.format(model_name, np.sqrt(metrics.mean_squared_error(y_train, y_train_hat)), np.sqrt(metrics.mean_squared_error(y_test, y_test_hat))), fontsize=fontsize)
    b = plt.scatter(y_test, y_test_hat, s=25,c='#1f78b4')
    plt.legend((a,b),('Train','Test'),fontsize=fontsize,handletextpad=0.1,borderpad=0.1)
    plt.rcParams['font.family']="Arial"
    plt.tight_layout()
    #plt.savefig('Name.png', dpi = 1200)
    plt.show()

########################################################################################################################################################################################################

# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def rfr_model(X, y):

    # Perform Grid-Search
    gsc = GridSearchCV(
          estimator=RandomForestRegressor(random_state=42),
          param_grid={
              'max_depth': range(1,5),
              'n_estimators': (5,10,20,30,40,50,60,70,80,90,100,500,1000),
          },
          cv=5, 
          scoring= 'neg_mean_squared_error', 
          verbose=0,
          n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    rf_regressor = RandomForestRegressor(**best_params)  # Utilizza i migliori parametri trovati
        
    return rf_regressor, best_params

rf_regressor, best_params = rfr_model(X_train_stand, Y_Voc_train)
print('The best rf parameters: ', best_params)

# rf_regressor = RandomForestRegressor(max_depth = 4,
#                                      n_estimators=100,
#                                      random_state=42)

# Fit to the training set
rf_regressor.fit(X_train_stand, Y_Voc_train)
# Perform predictions on both training and test sets
Y_Voc_train_RF = rf_regressor.predict(X_train_stand)
Y_Voc_test_RF = rf_regressor.predict(X_test_stand)

print("Random Forest trainato...")
print()
print("######################################################################")
print()

#Visualize the results
# prediction_vs_ground_truth_fig(Y_Voc_train, Y_Voc_train_RF, Y_Voc_test, Y_Voc_test_RF, "Random Forest")
# rf_regressor.feature_importances_

########################################################################################################################################################################################################

# Using SHAP for Interpretability 
# N.B Problema grave della libreria shap a causa del cambiamento nelle ultime versioni di numpy riguardo bool e np.bool -> risolto installando numpy version 1.23.0

import shap

print("Inizio calcolo SHAP Values...")

# Random Forest fit on train set SHAP  ################################################################################################################################################################################################

# # Use shap to explain Random Forest results
# explainerRF = shap.TreeExplainer(rf_regressor, check_additivity=False)

# # Get SHAP values on standardized input values
# shap_values_RF = explainerRF.shap_values(X_train_stand_no)

# plt.figure()
# shap.summary_plot(shap_values_RF, X_train_noname, plot_type = "dot", plot_size=(12, 8),
#                             color=plt.get_cmap('plasma'),
#                             show = False)
# plt.rcParams['font.sans-serif'] = "Arial"
# plt.rcParams.update({'font.size': 60})

# # plt.title("Random Forest SHAP values", fontsize=20)

# # Change the colormap of the artists, UNCOMMENT FOR DEFAULT COLORMAP
# my_cmap = plt.get_cmap('viridis')
# for fc in plt.gcf().get_children():
#     for fcc in fc.get_children():
#         if hasattr(fcc, "set_cmap"):
#             fcc.set_cmap(my_cmap)

# plt.show()

# ### dependece variable plot #####
            
# # Extract the colormap from the summary plot
# cmap = plt.get_cmap('viridis')

# # Index of the feature "Molecular_Weight(u)" in your dataset
# feature_index = X_train_noname.columns.get_loc("Concentration(mM)")

# # Get the name of the chosen feature
# chosen_feature_name = X_train_noname.columns[feature_index]

# # Create a SHAP dependence plot for the chosen feature
# fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed
# shap.dependence_plot(
#     ind=feature_index,
#     shap_values=shap_values_RF,
#     features=X_train_stand_no,
#     feature_names=X_train_noname.columns,
#     title=f"Partial Dependence for {chosen_feature_name}",
#     ax=ax,
#     show=False
# )

# # Set the colormap for the artists of the first plot
# for fcc in ax.get_children():
#     if hasattr(fcc, "set_cmap"):
#         fcc.set_cmap(cmap)

# plt.show()

# Random Forest fit on all dataset SHAP  ################################################################################################################################################################################################

# Fit RF to all dataset 
rf_regressor.fit(X_stand, Y_Voc)

# Use shap to explain Random Forest results
# explainerRF = shap.TreeExplainer(rf_regressor, check_additivity=False)
explainerRF = shap.TreeExplainer(rf_regressor)

# Get SHAP values on standardized input values
shap_values_RF = explainerRF.shap_values(X_stand)
plt.rcParams['figure.dpi'] = 300

plt.figure()
shap.summary_plot(shap_values_RF, X_noname, plot_type = "dot",
                            color=plt.get_cmap('plasma'),
                            show = False)

plt.title("Random Forest SHAP values", fontsize=20)

# Change the colormap of the artists, UNCOMMENT FOR DEFAULT COLORMAP
my_cmap = plt.get_cmap('viridis')
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap(my_cmap)

# Visualizzare il grafico
plt.show()

# #### SHAP vs features plot #####
            
# for feature_name in ProcessingData_cols:
#     if feature_name == "DeviceName":
#         continue  
#     feature_index = X_noname.columns.get_loc(feature_name)
    
#     plt.rcParams['figure.dpi'] = 300
    
#     plt.figure(figsize=(12, 8))
#     shap.dependence_plot(
#         feature_index,  # Index of the feature you want to plot
#         shap_values_RF,  # SHAP values
#         X_noname,  # Feature values
#         display_features=X_noname,  # Feature values 
#         interaction_index=None,  
#         show=False  # Set to True if you want to display the plot immediately
#     )

#     plt.xlabel(feature_name)  # Set the X-axis label
#     plt.ylabel("SHAP value")  # Set the Y-axis label
#     # plt.title(f"SHAP Dependence Plot for {feature_name}", fontsize=16)

    
#     plt.show()

