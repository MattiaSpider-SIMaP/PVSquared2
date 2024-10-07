# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:45:50 2024

@author: utente
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

plt.rcParams['figure.dpi'] = 300

TableName = "1&2&5-Rev&For-Clean(Voc-1000)_ML.txt"
pd.set_option('display.max_rows', 10, 'display.max_columns', 10)

# The molecular features & processing conditions are loaded as X
# All
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Heavy_Atom_Count", "Complexity", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio", "Heteroatom_count"]
# SignificativeOLD
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC)
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA, Complexity, & HBDC & HBAC & H_count & C_count)
ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA, Complexity, & HBDC & HBAC & H_count & C_count & Annealing) 
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA, Complexity, & HBDC & HBAC & H_count & C_count & Annealing & Concentration) 
# ProcessingData_cols = ["DeviceName", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# No Atom_count
# ProcessingData_cols = ["DeviceName", "Concentration(mM)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]

# Load the table with full data
Full_Data = pd.read_csv(TableName, delim_whitespace=True, encoding="iso-8859-1")

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

# Ottengo valori Voc legati ai nomi dei device
Y_Voc = Full_Data[["DeviceName", "Voc(mV)"]]

# Tutte le sigle in DeviceName
all_device_initials = [['MEPEAI-5mM-20T',
                        'MEPEAI-5mM-100T',
                        'MEPEAI-10mM-20T',
                        'MEPEAI-10mM-100T',
                        'MEPEAI-15mM-20T',
                        'MEPEAI-15mM-100T'],
                       ['MEPEABr-5mM-20T',
                        'MEPEABr-5mM-100T',
                        'MEPEABr-10mM-20T',
                        'MEPEABr-10mM-100T',
                        'MEPEABr-15mM-20T',
                        'MEPEABr-15mM-100T'],
                       ['MEPEACl-5mM-20T',
                        'MEPEACl-5mM-100T',
                        'MEPEACl-10mM-20T',
                        'MEPEACl-10mM-100T',
                        'MEPEACl-15mM-20T',
                        'MEPEACl-15mM-100T'],
                       ['n-BAI-5mM-20T',
                        'n-BAI-5mM-100T',
                        'n-BAI-10mM-20T',
                        'n-BAI-10mM-100T',
                        'n-BAI-15mM-20T',
                        'n-BAI-15mM-100T'],
                       ['iso-BAI-5mM-20T',
                        'iso-BAI-5mM-100T',
                        'iso-BAI-10mM-20T',
                        'iso-BAI-10mM-100T',
                        'iso-BAI-15mM-20T',
                        'iso-BAI-15mM-100T'],
                       ['n-OAI-5mM-20T',
                        'n-OAI-5mM-100T',
                        'n-OAI-10mM-20T',
                        'n-OAI-10mM-100T',
                        'n-OAI-15mM-20T',
                        'n-OAI-15mM-100T'],
                       ['BBr-5mM-20T',
                        'BBr-5mM-100T',
                        'BBr-10mM-20T',
                        'BBr-10mM-100T',
                        'BBr-15mM-20T',
                        'BBr-15mM-100T'],
                       ['HBr-5mM-20T',
                        'HBr-5mM-100T',
                        'HBr-10mM-20T',
                        'HBr-10mM-100T',
                        'HBr-15mM-20T',
                        'HBr-15mM-100T'],
                       ['OATsO-5mM-20T',
                        'OATsO-5mM-100T',
                        'OATsO-10mM-20T',
                        'OATsO-10mM-100T',
                        'OATsO-15mM-20T',
                        'OATsO-15mM-100T']]

#liste in cui verranno inseriti gli errori medi dei modelli per ogni ciclo 
rf_errors = []

#liste in cui verranno inseriti le std medie dei modelli per ogni ciclo 
rf_stds = []


# Validazione all but one per estrarre modello che generalizza meglio
for device_initials in all_device_initials:
    
    # Estrai l'iniziale comune dalla prima stringa nella lista
    import re 
    common_initial = re.split(r'-\d', device_initials[0])[0]
        
    # Creo train set con n-1 cationi e test set con n-esimo catione
    X_train = X[~X['DeviceName'].str.startswith(tuple(device_initials))]
    X_test = X[X['DeviceName'].str.startswith(tuple(device_initials))]
    Y_Voc_train = Y_Voc[~Y_Voc['DeviceName'].str.startswith(tuple(device_initials))]
    Y_Voc_test = Y_Voc[Y_Voc['DeviceName'].str.startswith(tuple(device_initials))]

    print()
    print("Train con n-1 cationi e test con n-esimo catione pronti...")
    print()
    print("######################################################################")
    print()

    # Ciclo per calcolare la media dei valori sperimentali per i 6 device nella lista corrente
    experimental_values_mean = []  # Lista per salvare le medie dei valori sperimentali
    for device_name in device_initials:
        # Filtra i valori Voc corrispondenti al nome device_name nella colonna DeviceName
        values = Y_Voc_test[Y_Voc_test["DeviceName"].str.contains(device_name)]["Voc(mV)"].values
        
        # print(values)
        
        # Calcola la media dei valori sperimentali
        values_mean = np.mean(values)
        experimental_values_mean.append(values_mean)
    
    print(experimental_values_mean)
    
    print()
    print("Medie valori sperimentali device con n-esimo catione calcolate...")
    print()
    print("######################################################################")
    print()
    
    # Tolgo colonna dei nomi
    Y_Voc_train = Y_Voc_train.drop("DeviceName", axis=1)
    Y_Voc_test = Y_Voc_test.drop("DeviceName", axis=1)
    
    # Trasformo da colonna a riga 
    Y_Voc_train = Y_Voc_train.values.ravel() 
    Y_Voc_test = Y_Voc_test.values.ravel() 
    
    # print(X_train)
    # print(X_test)
       
    ########################################################################################################################################################################################################
    
    #Standardizzazione valori di input con Standard Scaler
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_noname = X.drop("DeviceName", axis=1)
    X_train_noname = X_train.drop("DeviceName", axis=1)
    X_test_noname = X_test.drop("DeviceName", axis=1)
    
    #Standardizzazione
    scaler.fit(X_noname)
    X_train_stand = scaler.transform(X_train_noname)
    X_train_stand_no = scaler.transform(X_train_noname) #rinominazione per lo shuffle
    X_test_stand = scaler.transform(X_test_noname) 
    
    print("Standardizzazione dei valori di input svolta...")
    print()
    print("######################################################################")
    print()
    
    # #No standardizzazione
    # X_train_stand = X_train_noname
    # X_test_stand = X_test_noname
    
    #Shuffle dei dati 
    np.random.seed(1)
    i_rand = np.arange(X_train_stand_no.shape[0])
    np.random.shuffle(i_rand)
    X_train_stand = np.array(X_train_stand_no)[i_rand]
    Y_Voc_train = np.array(Y_Voc_train)[i_rand]

    print("Shuffle dei dati svolto...")
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
                  'n_estimators': (5,10,20,30,40,50,60,70,80,90,100,500,1000),
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
    #                                       n_estimators=20,
    #                                       random_state=42)
    
    # Fit to the training set
    rf_regressor.fit(X_train_stand, Y_Voc_train)
    # Perform predictions on both training and test sets
    Y_Voc_train_RF = rf_regressor.predict(X_train_stand)
    Y_Voc_test_RF = rf_regressor.predict(X_test_stand)
    
    print("Random Forest trainato...")
    print()
    print("######################################################################")
    print()
   
    ########################################################################################################################################################################################################
    
    # Otteniamo X_test_stand con solo 6 righe, una per ogni combinazione della lista corrente
    X_test_6 = pd.DataFrame(columns=X.columns)  
    for initial in device_initials:
        device_data = X[X['DeviceName'].str.startswith(initial)]
        if len(device_data) > 0:
            row = device_data.iloc[0]  # Prendi la prima riga del gruppo
            X_test_6 = X_test_6.append(row, ignore_index=True)
            
    # Applico stesso scaler anche a X_test_6
    X_test_6_noname = X_test_6.drop("DeviceName", axis=1)
    X_test_6_stand = scaler.transform(X_test_6_noname) 
    
    # # No standardizzazione 
    # X_test_6_stand = X_test_6.drop("DeviceName", axis=1)
        
    # Otteniamo le medie dei valori sperimentali per ogni combinazione contenuta nella lista corrente
    Y_Voc_test_n_mean = []
    for device in device_initials:
        device_data = Y_Voc[Y_Voc['DeviceName'].str.startswith(device)]
        mean_voc_value = np.mean(device_data['Voc(mV)'])
        Y_Voc_test_n_mean.append(mean_voc_value)

    # print(Y_Voc_test_n_mean)    

    # Ottieni predizioni per ogni modello sulle 6 combinazioni dell'n-esimo catione
    Y_Voc_test_RF = rf_regressor.predict(X_test_6_stand)

    # Ricaviamo l'RMSE e la std
    from sklearn import metrics
    rf_score = np.sqrt(metrics.mean_squared_error(Y_Voc_test_n_mean, Y_Voc_test_RF))
    
    rf_std = np.std(Y_Voc_test_n_mean - Y_Voc_test_RF)

    max_Voc = max(Y_Voc_test)
    min_Voc = min(Y_Voc_test)
    range_Voc = max_Voc - min_Voc
    
    scores = [rf_score]
    stds = [rf_std]
    
    scores_normalized = [score / range_Voc * 100 for score in scores]
    stds_normalized = [std / range_Voc * 100 for std in stds]
    
    # # Visualizza valori 
    # fig,ax = plt.subplots(figsize=(8,6))
    # print('RMSE scores for LR, KNN, RF, GB, SVR, NN,: ' ,scores)
    # print('Std. dev: ',stds)
    # fontsize = 12
    # plt.rc('xtick', labelsize=fontsize)
    # plt.rc('ytick', labelsize=fontsize)
    # model = ['Linear Regression', 'K Nearest Neighbors', 'Random Forest', 'Gradient Boosting', 'SVR', 'Neural Network']
    # ax.bar(model,scores,yerr=stds,alpha=0.7, capsize=2)
    # ax.set_ylabel('RMSE (mV)',fontsize=fontsize)
    # ax.set_title('Scores su n-esimo catione (' + common_initial + ')',fontsize=fontsize)
    # plt.xticks(rotation=90)
    # plt.ylim(0,100)
    # plt.show()
    # plt.rcParams['font.family']="Arial"
    
    # # Visualizza valori normalizzati
    # fig,ax = plt.subplots(figsize=(8,6))
    # print('RMSE scores for LR, KNN, RF, GB, SVR, NN,: ' ,scores_normalized)
    # print('Std. dev: ',stds_normalized)
    # fontsize = 12
    # plt.rc('xtick', labelsize=fontsize)
    # plt.rc('ytick', labelsize=fontsize)
    # model = ['Linear Regression', 'K Nearest Neighbors', 'Random Forest', 'Gradient Boosting', 'SVR', 'Neural Network']
    # ax.bar(model,scores_normalized,yerr=stds_normalized,alpha=0.7, capsize=2)
    # ax.set_ylabel('RMSE (%)',fontsize=fontsize)
    # ax.set_title('Scores su n-esimo catione normalizzati  (' + common_initial + ')', fontsize=fontsize)
    # plt.xticks(rotation=90)
    # plt.ylim(0,100)
    # plt.show()
    # plt.rcParams['font.family']="Arial"
    
    # Plot confronto tra dati sperimentali e dati previsi dai modelli
    data = [Y_Voc_test_n_mean, Y_Voc_test_RF]  # Lista di array dei dati per i diversi modelli
    
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # sns.stripplot(data=data, marker='o', color='white', edgecolor='black', size=5)
    # sns.boxplot(data=data, orient='v', width=0.5, palette="vlag", showmeans=True, meanline=True, meanprops={'color': 'red'})
    sns.boxplot(data=data, orient='v', width=0.5, palette="vlag")
    sns.stripplot(data=data, size=5, linewidth=0.7, palette="vlag", edgecolor='black')
    plt.xlabel('Model')
    plt.ylabel('Voc (mV)')
    plt.title('Comparison of experimental and model-predicted data (' + common_initial + ')')
    
    # Etichette sull'asse x per i modelli
    model_labels = ['experimental data', 'Random Forest']  # Aggiungi le etichette per gli altri modelli
    ax.set_xticklabels(model_labels, rotation=90)
    plt.ylim(900,1200)
    
    plt.show()
    
    #inserisco gli score ottenuti nelle liste
    rf_errors.append(rf_score)  
    
    print(rf_errors)
    
    #inserisco le std ottenute nelle liste
    rf_stds.append(rf_std)  

    print(rf_stds)
    
    print()
    print("All but one con", common_initial, "svolto")
    print()
    print("-------------------------------------------------------------------------")
    print()
    
#Ottengo il valore medio degli RMSE di ogni modello su tutti i cicli all but one
rf_mean_error = np.mean(rf_errors)   

#Ottengo il valore medio delle stds di ogni modello su tutti i cicli all but one 
rf_mean_std = np.mean(rf_stds)   
rf_std_RMSE = np.std(rf_errors)

ABO_scores = [rf_mean_error]
ABO_std = [rf_mean_std]

# Visualizza risultato
plt.rcParams['figure.dpi'] = 300
fig,ax = plt.subplots(figsize=(8,6))
print('Mean RMSE in all cicles for RF: ' , rf_errors)
print('All but one validation RMSE score for RF: ' , ABO_scores)
print('All but one validation std score for RF: ' , ABO_std)
print('All but one validation std (RMSE) score for RF: ' , rf_std_RMSE)
# print('Std. dev: ',stds)
fontsize = 12
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
model = ['Random Forest']
ax.bar(model, ABO_scores, yerr=rf_std_RMSE, alpha=0.7, capsize=4)
ax.set_ylabel('RMSE (mV)',fontsize=fontsize)
# ax.set_title('All but One validation scores (experimental means)',fontsize=fontsize)
plt.xticks(rotation=90)
plt.ylim(0,70)
plt.show()
plt.rcParams['font.family']="Arial"

plt.show

# Mean RMSE in all cicles for RF:  [38.74474398522641, 35.267852299621545, 24.3465392948663, 8.006577795903482, 9.17576694959679, 21.541052401708217, 13.305598854331505, 8.678062553070465, 30.118294621246093]
# All but one validation RMSE scores for RF:  [21.020498750618977]
# All but one validation std scores for RF:  [13.498605524786047]
