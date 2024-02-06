# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 19:54:38 2023

@author: utente
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

plt.rcParams['figure.dpi'] = 600

TableName = "1&2&5-Rev&For-Clean(Voc-1000)_ML.txt"
TableCandidate = "Cationi_candidati_ML.txt"

pd.set_option('display.max_rows', 10, 'display.max_columns', 10)

# The molecular features & processing conditions are loaded as X
#All
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Heavy_Atom_Count", "Complexity", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio", "Heteroatom_count"]
#Significative
#ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
#SignificativeNEW (no TPSA & HBDC)
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Hydrogen_Bond_Acceptor_Count", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC & H_count & C_count)
ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
#No Atom_count
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]

# Load the table with full data
Full_Data = pd.read_csv(TableName, delim_whitespace=True, encoding="iso-8859-1")

# prefissi_da_rimuovere = ['MEPEACl-5mM-20T', 
#                          'MEPEACl-5mM-100T', 
#                          'MEPEACl-10mM-20T', 
#                          'MEPEACl-10mM-100T', 
#                          'MEPEACl-15mM-20T', 
#                          'MEPEACl-15mM-100T']

# # Filtra il DataFrame per rimuovere le righe che iniziano con i prefissi specificati
# Full_Data = Full_Data[~Full_Data["DeviceName"].str.startswith(tuple(prefissi_da_rimuovere))]

# # Check if the data is loaded and separated correctly
# print(Full_Data)

# Creo il dataframe con solo i valori di processazione e proprietà fisico-chimiche di interesse (input)
X = Full_Data[ProcessingData_cols]

# Creo il dataframe con solo i valori di Voc (output)
Y_Voc = Full_Data["Voc(mV)"]

# Load the table with candidates data
Candidates_Data = pd.read_csv(TableCandidate, delim_whitespace=True, encoding="iso-8859-1")

# Creo il dataframe con solo i valori di processazione e proprietà fisico-chimiche di interesse (input)
X_Candidates = Candidates_Data[ProcessingData_cols]

# # Check if the data is loaded and separated correctly
# pd.set_option('display.max_rows', 10, 'display.max_columns', 5)
# print(X, Y_Voc)
# print(Candidates_Data)

print()

print("Dataframe caricati correttamente...")

print()
print("######################################################################")
print()

########################################################################################################################################################################################################

# Imposto per il training dei modelli tutti i dati raccolti in Full_Data
X_train = X
Y_Voc_train = Y_Voc

# #No standardizzazione
# X_train_stand = X_train.drop("DeviceName", axis=1)
# X_Candidates_stand = X_Candidates.drop("DeviceName", axis=1)

#Standardizzazione valori di input con Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_noname = X_train.drop("DeviceName", axis=1)
X_Candidates_noname = X_Candidates.drop("DeviceName", axis=1)

scaler.fit(X_train_noname)
X_train_stand = scaler.transform(X_train_noname)
X_Candidates_stand = scaler.transform(X_Candidates_noname)



print("Training set standardizzato pronto...")
print()
print("######################################################################")
print()

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
              'n_estimators': (5,10,20,30,40,50,60,70,80,90,100,500,1000,2000),
          },
          cv=5, 
          scoring= 'neg_mean_squared_error', 
          verbose=0,
          n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    rf_regressor = RandomForestRegressor(random_state=42, **best_params)  # Utilizza i migliori parametri trovati
        
    return rf_regressor, best_params

rf_regressor, best_params = rfr_model(X_train_stand, Y_Voc_train)
print('The best rf parameters: ', best_params)

# rf_regressor = RandomForestRegressor(max_depth = 4,
#                                      n_estimators=50,
#                                      random_state=42)

# Fit to the training set
rf_regressor.fit(X_train_stand, Y_Voc_train)

print("Random Forest trainato...")
print()
print("######################################################################")
print()

########################################################################################################################################################################################################

# Perform predictions candidates
Y_Voc_predictions_RF = rf_regressor.predict(X_Candidates_stand)

# max_index = Y_Voc.argmax()  # Ottieni l'indice del valore massimo di Y_Voc
# max_row = Full_Data.iloc[max_index]  # Estrai la riga corrispondente all'indice massimo
# print(max_row)

print()

# Estrai le iniziali dei nomi dei device + concentrazione + temperatura
Candidates_Data['Iniziali_C_T'] = Candidates_Data['DeviceName'].str.extract(r'([A-Za-z\-]+-[0-9A-Za-z]+-[0-9A-Za-z]+)').astype(str)
Candidates_Data['Iniziali_C_T'] = Candidates_Data['DeviceName'].str.replace('T$', '°C')

# Plot dei valori di Voc previsti dai modelli 
predictions = [Y_Voc_predictions_RF]  
models = ['Random Forest'] 

for candidate, prediction in zip(Candidates_Data['Iniziali_C_T'], Y_Voc_predictions_RF):
    print(candidate, prediction)

for i, prediction in enumerate(predictions):
    model = models[i]  # Ottieni il nome del modello corrente
    sns.set_theme(style="ticks")
    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(figsize=(4, 6))  
    
    sns.boxplot(x=prediction, y=Candidates_Data['Iniziali_C_T'], orient='h', width=1, 
                palette="vlag", showmeans=True, meanline=True, meanprops={'color': 'black'})
    sns.stripplot(x=prediction, y=Candidates_Data['Iniziali_C_T'], size=10, linewidth=0.4,
                  edgecolor='black', marker='d', hue=prediction, palette='Blues')
    plt.xlabel('$\mathrm{{V}}_{{\mathrm{{oc}}}}$ (mV)', fontsize=12) # Scambia le etichette x e y
    plt.ylabel('Candidates organic cations')
    # plt.title('Voc previste su cationi candidati (' + model + ')')
    
    ax.set_yticklabels(Candidates_Data['Iniziali_C_T'], rotation=0) 
    ax.set_xlim(1090, 1130) 
    
    plt.legend().remove()
    
    plt.show()

print()

filtered_rows = Full_Data[Full_Data['DeviceName'].str.startswith('MEPEACl-10mM-20T')]
mean_voc = filtered_rows['Voc(mV)'].mean()
print("Best device MEPEACl-10mM-20°C mean value: " + str(mean_voc))

print("Valore previsto per PEACl-10mM-20°C =", str(Y_Voc_predictions_RF[7]))
