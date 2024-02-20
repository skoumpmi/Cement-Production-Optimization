from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
#from datetime import datetime, timedelta
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV,train_test_split
import configparser
import pickle

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor

#from sklearn_extensions.extreme_learning_machines import ELMRegressor
config = configparser.ConfigParser()
config.read('config.ini')

#data_n =pd.read_csv("siroko_mill_newdat_filtered.csv") #to dataset toy tsalikidi (etoimo)  <----

csv_files = [
    r"C:\Users\kioro\.spyder-py3\CM7_2021_12.csv",
    r"C:\Users\kioro\.spyder-py3\CM7_2022_01.csv",
    r"C:\Users\kioro\.spyder-py3\CM7_2022_04.csv",
    r"C:\Users\kioro\.spyder-py3\CM7_2022_05.csv",
    r"C:\Users\kioro\.spyder-py3\CM7_2022_06.csv",
    r"C:\Users\kioro\.spyder-py3\CM7_2022_07.csv",
    r"C:\Users\kioro\.spyder-py3\CM7_2022_08.csv",
    r"C:\Users\kioro\.spyder-py3\CM7_2022_09.csv",
    r"C:\Users\kioro\.spyder-py3\CM7_2022_10.csv",
    r"C:\Users\kioro\.spyder-py3\CM7_2022_11.csv",
    r"C:\Users\kioro\.spyder-py3\CM7_2022_12.csv"
]

# Read the first CSV file to get the columns
df_siroko = pd.read_csv(csv_files[0])

# Iterate over the remaining CSV files
for csv_file in csv_files[1:]:
    # Read each CSV file
    df = pd.read_csv(csv_file)
    
    # Get the common columns
    common_columns = set(df.columns) & set(df_siroko.columns)
    
    # Concatenate only the common columns
    df_siroko = pd.concat([df_siroko[common_columns], df[common_columns]], axis=0, ignore_index=True)

# # Print the resulting DataFrame
# print(df_siroko)

df_siroko['Timestamp']=pd.to_datetime(df_siroko['Timestamp']).dt.tz_localize(None)
tdi = pd.DatetimeIndex(df_siroko.Timestamp)
df_siroko.set_index(tdi, inplace=True)
df_siroko.index.name = 'Timestamp'

mill_motor_d = df_siroko["537-MD01/M01-J01.MV"]
mill_motor_d = pd.to_numeric(mill_motor_d, errors='coerce')
mill_motor_d_index=mill_motor_d.index

active=[]
non_active=[]
ex_list =  mill_motor_d
ex_list_index = mill_motor_d_index
c=0
for i in range(len(ex_list)):
    if math.isnan(ex_list[i]):
        continue
    if  ex_list[i] != 0 and c>=40:
        active.append(i)
    elif ex_list[i] !=0 and c<40:
        non_active.append(i)
        c+=1
    elif ex_list[i]==0:
        non_active.append(i)
        c=0
#active_n=np.unique(active).tolist()
#non_active_n=np.unique(non_active).tolist()
df_siroko=df_siroko.drop(df_siroko.index[non_active])
df_siroko=df_siroko.dropna()

df_siroko = df_siroko.apply(pd.to_numeric, errors='coerce')
df_siroko=df_siroko.dropna()


new_cols_name= {
'517-WF00/A01-F01.MV':'MillFeed',   
'537-RM01/A01-PC1.LMN': 'MillFeedSP',
'537-SR01/M01-S01.MV': 'SeparatorSpeed',
'537-HY01/A01-P01.MV':'GrindingPressure',
'537-HY01/A01-P11.SP_Out#Value':'GrindingPressureSP',
'537-WI01/N01-F01.MV':'WaterFlow',
'537-WI01/M01-TC32.SP':'WaterFlowSP',
'537-RM01/N01-P01.MV':'MillInletSubpressure',
'537-LD02/M01-Z11.SP':'MillInletSubpressureSP',
'537-RM01/N03-P01.MV':'MillOutletSubpressure',
'537-FN01/M01-S12.SP':'MillOutletSubpressureSP',
'537-BF01/A01-P01.MV':'BagFilter',
'537-HY01/A01-L01.MV':'GrindingLayerRoller3',
'537-GR02/M01-S11.SP_Out#Value':'GrindingAidSP',
'537-GR02/M01-S11.SP_Out#Value':'GrindingAidPV',
'235-FD02/A01-Z01.MV':'FlyAshPV',
'235-FD02/A01-ZC1.SP_Out#Value':'FlyAshSP',
'517-WF04/A01-F01.MV':'PozzolanePV',
'517-WF04/A01-F11.SP_Out#Value':'PozzolaneSP',
'517-WF02/A01-F01.MV':'LimestonePV',
'517-WF02/A01-F11.SP_Out#Value':'LimestoneSP',
'517-WF03/A01-F01.MV':'GypsumPV',
'517-WF03/A01-F11.SP_Out#Value':'GypsumSP',
'537-GA01/Y01.PosSig1#Value':'BranchingchuteOPEN',
'537-GA02/Y01.PosSig2#Value':'BranchingchuteCLOSE',
'537-FN01/M01-S01.MV':'FanSpeed',
'537-DM01/A01-VO2.MV':'Dust',
'537-RM01/N05-P01.MV':'MillDifferentialPressure',
'537-MD01/M01-J01.MV':'MillKW',
'517-BE01/M01-I01.MV':'ElevatorLoad',
'537-RM01/N02-T01.MV':'MillInletTemperature',
'537-RM01/N04-T01.MV':'MillExitTemperature',
'537-BV01/M01-Z11.SP':'MillExitTemperatureSP',
'537-BV01/M01-Z01.MV':'FreshAirDamperPV',
'537-LD02/M01-Z01.MV':'RecirculationDamper',
'537-SR01/M01-J01.MV':'SeparatorPower',
'537-RM01/A01-V02.MV':'MillVibrations',
'537-FN01/M01-J01.MV':'MillFanLoad',
'537-QCXV/RESIDUE.MV':'Residue',
'537-QCXV/RESIDUE_TG.MV':'ResidueTarget',
'537-QCXV/BLAINE.MV':'Blaine',
'537-QCXV/BLAINE_TG.MV':'BlaineTarget',
'537-DM01/A01-V02-HH_VALUE.MV':'EnvironmentalDust',
'537-MD01/N11-T01.MV':'MotorWindTemp',
'P_537CS01/FROM_CMO.R_BYTE02':'CMO_Ready',
'P_537CS01/537CS01_DB-DAD.DA_REAL_06':'SampleTime',
'P_537CS01/REZ_DATA.Y_Co05_PERCENT':'FlyAsh%',
'P_537CS01/REZ_DATA.Y_Co04_PERCENT':'Pozzolana%',
'P_537CS01/REZ_DATA.Y_Co03_PERCENT':'Limestone%',
'P_537CS01/REZ_DATA.Y_GY_PERCENT':'Gypsum%',
'537-QCXV/RECIPE.MV':'CementType'#,'537-FN01/M01-S13.PV_IN': 'SpecificAirFlow'
}
df_siroko.rename(columns=new_cols_name, inplace=True)


data=df_siroko[[
'MillFeed',
'SeparatorSpeed',
'GrindingPressure',
'WaterFlow',
'MillInletSubpressure',
'MillOutletSubpressure',
'BagFilter',
'GrindingLayerRoller3',
'GrindingAidPV',
'FlyAshPV',
'PozzolanePV',
'LimestonePV',
'GypsumPV',
'BranchingchuteOPEN',
'BranchingchuteCLOSE',
'FanSpeed',
'Dust',
'MillDifferentialPressure',
'MillKW',
'ElevatorLoad',
'MillInletTemperature',
'MillExitTemperature',
'FreshAirDamperPV',
'RecirculationDamper',
'SeparatorPower',
'MillVibrations',
'MillFanLoad',
'Residue',
'ResidueTarget',
'Blaine',
'BlaineTarget',
'EnvironmentalDust',
'MotorWindTemp',
'CMO_Ready',
'SampleTime',
'FlyAsh%',
'Pozzolana%',
'Limestone%',
'Gypsum%',
'CementType'#,'SpecificAirFlow'
]]


cv=2
def regr(model_kind,X,Y,name): #model kind = {'MLPReg" "RandomForest" / "Linear"(MLPRegressor me 0 hidden_layer_sizes )/"KNN" (KNeighborsRegressor) / "Linear2" (LinearRegression)}
    dct ={}
    Y = Y
    X = X.values.astype(np.float)
    scaler = MinMaxScaler()
    X = scaler.fit(X).transform(X)
    
    with open('scalers/'+model_kind+"_"+name+'X'+"_"+'_scaler.pkl','wb') as fid:
        pickle.dump(scaler,fid)

    Y = Y.values.astype(np.float)
    Y = scaler.fit(Y).transform(Y)
    
    with open('scalers/'+model_kind+"_"+name+'Y'+"_"+'_scaler.pkl','wb') as fid:
        pickle.dump(scaler,fid)    
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = False, random_state=0)
 
    if model_kind=='MLPReg':
        best_model_score=1.0
        hidden_layer_sizes= eval(('"'+config['MLPReg']['hidden_layer_sizes']+'"').replace('"(','[(').replace(')"',')]'))
        #hidden_layer_sizes= eval(('"'+config['MLPReg']['hidden_layer_sizes']+'"').replace("''(","[(").replace(")''",")]"))
        #hidden_layer_sizes= eval(('"['+config['MLPReg']['hidden_layer_sizes']+']"'))
        max_iter = [int(i) for i in config['MLPReg']['max_iter'].split(',')]
        param_grid = dict(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
        #param_grid = dict(max_iter=max_iter)
        grid = GridSearchCV(MLPRegressor(), param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=2, n_jobs=-1,verbose=5)
        grid.fit(X_train, y_train)
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        params = grid.cv_results_['params']
        best_score = 1.0
        
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            if abs(mean) < best_score:
                print('================================')
                best_score = abs(mean)
                best_param = param
                                            
                print(best_score)
        print(best_param)
        print(best_score)
        
        if best_score<best_model_score:
            best_model_score=best_score     
            
            model=MLPRegressor(hidden_layer_sizes=(5,), activation="tanh", solver="lbfgs", alpha=0, max_iter=best_param['max_iter'], verbose=False)
            #model=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation="tanh", solver="lbfgs", alpha=0, max_iter=best_param['max_iter'], verbose=False)
            # model.save('MLP.pkl')
            # filename = 'MLP.sav'
            # pickle.dump(model, open(filename, 'wb'))
            
    elif model_kind=="RandomForest":
        best_model_score=1.0
        #cv=2
        n_estimators =  [int(i) for i in config['RandomForest']['n_estimators'].split(',')]#[10, 15 , 20]
        max_depth = [int(i) for i in config['RandomForest']['max_depth'].split(',')]#[10, 15 , 20]
        param_grid = dict(n_estimators=n_estimators, max_depth = max_depth)
        grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=2, n_jobs=-1,verbose=5)
        grid.fit(X_train, y_train)
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        params = grid.cv_results_['params']
        best_score = 1.0

        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            if abs(mean) < best_score:
                print('================================')
                best_score = abs(mean)
                best_param = param
                                            
                print(best_score)
        print(best_param)
        print(best_score)
        
        if best_score<best_model_score:
            best_model_score=best_score        
            model = RandomForestRegressor(n_estimators=best_param['n_estimators'],max_depth = best_param['max_depth'] ,random_state=0)
           # model.save('RandomForest.pkl')
            # filename = 'RandomForest.sav'
            # pickle.dump(model, open(filename, 'wb'))
        
        
    # elif model_kind=="Linear":
    #     model=MLPRegressor(hidden_layer_sizes=(), activation="tanh",verbose=False)
    #     best_param=
        
        
    elif model_kind=="KNN":
        best_model_score=1.0
        n_neighbors = [int(i) for i in config['KNN']['n_neighbors'].split(',')]
        leaf_size  = [int(i) for i in config['KNN']['leaf_size'].split(',')]
        weights =  [i for i in config['KNN']['weights'].split(',')]
        param_grid = dict(n_neighbors = n_neighbors, leaf_size = leaf_size,  weights = weights)
        grid = GridSearchCV(KNeighborsRegressor(), param_grid=param_grid, scoring='neg_root_mean_squared_error',cv=2, n_jobs=-1, verbose=5) #,error_score='raise'
        grid.fit(X_train, y_train)
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        params = grid.cv_results_['params']
        best_score = 1.0
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            if abs(mean) < best_score:
                best_score = abs(mean)
                best_param = param
                print(best_score) 
        print(best_param)
        print(best_score)
        if best_score<best_model_score:
            best_model_score=best_score
            model = KNeighborsRegressor(n_neighbors=best_param['n_neighbors'], leaf_size=best_param['leaf_size'], weights=best_param["weights"])
            # model.save('KNN.pkl')
            # filename = 'KNN.sav'
            # pickle.dump(model, open(filename, 'wb'))
            
        
    elif model_kind=="Linear":
        best_model_score=1.0
        fit_intercept = [eval(i) for i in config['Linear']['fit_intercept'].split(',')]
        param_grid = dict(fit_intercept = fit_intercept)
        grid = GridSearchCV(LinearRegression(), param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, verbose= 5)
        grid.fit(X_train, y_train)
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        params = grid.cv_results_['params']
        best_score = 1.0
        for mean, stdev, param in zip(means, stds, params):
          print("%f (%f) with: %r" % (abs(mean), stdev, param))
          if abs(mean) < best_score:
              best_score = abs(mean)
              best_param = param 
              print(best_score)
              print(best_param)      
            
        if best_score<best_model_score:
            best_model_score=best_score    
            model = LinearRegression(fit_intercept=best_param['fit_intercept'])
            # model.save('linear.pkl')
            #filename = 'Linear.sav'
            #pickle.dump(model, open(filename, 'wb'))
    
    
    elif model_kind == "LGBM":
        best_model_score=1.0
        max_depth = [int(i) for i in config['LGBM']['max_depth'].split(',')]
        num_leaves = [int(i) for i in config['LGBM']['num_leaves'].split(',')]
        param_grid = dict(max_depth = max_depth,num_leaves=num_leaves)
        grid = GridSearchCV(LGBMRegressor(),param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, verbose= 5)
        grid.fit(X_train, y_train)
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        params = grid.cv_results_['params']
        best_score = 1.0
        for mean, stdev, param in zip(means, stds, params):
          print("%f (%f) with: %r" % (abs(mean), stdev, param))
          if abs(mean) < best_score:
              best_score = abs(mean)
              best_param = param 
              print(best_score)
              print(best_param)
        if best_score<best_model_score:
            best_model_score=best_score
            model = LGBMRegressor(max_depth=best_param['max_depth'],num_leaves = best_param['num_leaves'])
    
    
    elif model_kind == "XGBoost":
        best_model_score=1.0
        #objective = [i for i in config['XGBoost']['objective'].split(',')]
        n_estimators = [int(i) for i in config['XGBoost']['n_estimators'].split(',')]
        max_depth = [int(i) for i in config['XGBoost']['max_depth'].split(',')]
        param_grid = dict(n_estimators=n_estimators, max_depth = max_depth)
        grid = GridSearchCV(XGBRegressor(),param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, verbose= 5)
        grid.fit(X_train, y_train)
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        params = grid.cv_results_['params']
        best_score = 1.0
        for mean, stdev, param in zip(means, stds, params):
          print("%f (%f) with: %r" % (abs(mean), stdev, param))
          if abs(mean) < best_score:
              best_score = abs(mean)
              best_param = param 
              print(best_score)
              print(best_param)      
        if best_score<best_model_score:
            best_model_score=best_score
            model = XGBRegressor(n_estimators = best_param['n_estimators'],max_depth=best_param['max_depth'])
       
    # elif model_kind == "XGBoost":   
    #     model = XGBRegressor()
            
    elif model_kind == "GBR":
        best_model_score=1.0
        max_depth = [int(i) for i in config['GBR']['max_depth'].split(',')]
        n_estimators = [int(i) for i in config['GBR']['n_estimators'].split(',')]
        #learning_rate = [i for i in config['GBR']['learning_rate'].split(',')]
        #min_samples_split = [i for i in config['GBR']['max_depth'].split(',')]
        param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
        grid = GridSearchCV(GradientBoostingRegressor(),param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, verbose= 5)
        grid.fit(X_train, y_train)
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        params = grid.cv_results_['params']
        best_score = 1.0
        for mean, stdev, param in zip(means, stds, params):
          print("%f (%f) with: %r" % (abs(mean), stdev, param))
          if abs(mean) < best_score:
              best_score = abs(mean)
              best_param = param 
              print(best_score)
              print(best_param)      
        if best_score<best_model_score:
            best_model_score=best_score
            model = GradientBoostingRegressor(max_depth=best_param['max_depth'],n_estimators = best_param['n_estimators'])
    
    
    elif model_kind == "CatBoost":
        model = CatBoostRegressor()
    
    
    elif model_kind == "TTR":
        model = TransformedTargetRegressor()
    
    
    
    model.fit(X_train, y_train)
   #p=model.predict(X)
    ptest=model.predict(X_test)
    ptrain=model.predict(X_train)
        #return (prediction)


    output_theor_mean=np.mean(Y)

    trainlist1=list();trainlist2=list()
    testlist1=list();testlist2=list()
    testerror=list()
    power=2
    for j in range(len(y_train)):
        trainlist1.append(np.abs((y_train[j]-ptrain[j])**power))
        trainlist2.append(np.abs((y_train[j]-output_theor_mean)**power))
    

    for j in range(len(y_test)):
        testlist1.append(np.abs((y_test[j]-ptest[j])**power))
        testlist2.append(np.abs((y_test[j]-output_theor_mean)**power))
        testerror.append(y_test[j]-ptest[j])
    
    
    meanerror=np.mean(testerror)
    trainNRMPE=(np.sum(trainlist1)/np.sum(trainlist2))**(1/power)
    testNRMPE=(np.sum(testlist1)/np.sum(testlist2))**(1/power)

    #NRMPEdct['MLPRegressor'] = trainNRMPE,testNRMPE
    if model_kind=='MLPReg':
        dct = {'trainNRMPE':[trainNRMPE],'testNRMPE':[testNRMPE],'meanerror':[meanerror],"bestparam":[best_param]}
    elif model_kind=="RandomForest":
        dct = {'trainNRMPE':[trainNRMPE],'testNRMPE':[testNRMPE],'meanerror':[meanerror],"bestparam":[best_param]}
    # elif model_kind=="Linear":
    #     dct = {'trainNRMPE':[trainNRMPE],'testNRMPE':[testNRMPE],'meanerror':[meanerror],"bestparam":[best_param]}
    elif model_kind=="KNN":
        dct = {'trainNRMPE':[trainNRMPE],'testNRMPE':[testNRMPE],'meanerror':[meanerror],"bestparam":[best_param]}
    elif model_kind=="Linear":
        dct = {'trainNRMPE':[trainNRMPE],'testNRMPE':[testNRMPE],'meanerror':[meanerror],"bestparam":[best_param]}
    elif model_kind == "LGBM":
        dct = {'trainNRMPE':[trainNRMPE],'testNRMPE':[testNRMPE],'meanerror':[meanerror],"bestparam":[best_param]}
    elif model_kind == "XGBoost":   
        dct = {'trainNRMPE':[trainNRMPE],'testNRMPE':[testNRMPE],'meanerror':[meanerror],"bestparam":[best_param]} 
    elif model_kind == "GBR":
        dct = {'trainNRMPE':[trainNRMPE],'testNRMPE':[testNRMPE],'meanerror':[meanerror],"bestparam":[best_param]}
    elif model_kind == "CatBoost":    
        dct = {'trainNRMPE':[trainNRMPE],'testNRMPE':[testNRMPE],'meanerror':[meanerror]}
    elif model_kind == "TTR":
        dct = {'trainNRMPE':[trainNRMPE],'testNRMPE':[testNRMPE],'meanerror':[meanerror]}
        
        
    return(dct,model)

#sunartisi gia min NRMPE gia mia metabliti tou pp (sumfona me to poio model exei to mikrotero test_error epistrefei to model kai to antistoixo train_error & test_error)
def minimum(name,X,y):
    MLPRegressor={};
    RandomForest={};
    KNeighborsRe={};
    LinealModel={};
    LGBMRegressor={};
    XGBoostRegressor={};
    GBRRegressor={};
    CatBoostRegressor={};
    TTRRegressor={};
    MLPRegressor,modelMLPR = regr('MLPReg',X,y,name)
    RandomForest,modelRF = regr("RandomForest",X,y,name)
    KNeighborsRe,modelKNN = regr("KNN",X,y,name)
    LinealModel,modelL = regr("Linear",X,y,name)
    LGBMRegressor,modelLGBM = regr('LGBM',X,y,name)
    XGBoostRegressor,modelXGB = regr('XGBoost',X,y,name)
    GBRRegressor,modelGBR = regr('GBR',X,y,name)
    CatBoostRegressor,modelCatB = regr('CatBoost',X,y,name)
    TTRRegressor,modelTTR = regr('TTR',X,y,name)
    test_er_MLPr = MLPRegressor['testNRMPE']
    test_er_RandomF = RandomForest['testNRMPE']
    test_er_KNR = KNeighborsRe['testNRMPE']
    test_er_LM = LinealModel['testNRMPE']
    test_er_LGBM = LGBMRegressor['testNRMPE']
    test_er_XGB = XGBoostRegressor['testNRMPE']
    test_er_GBR = GBRRegressor['testNRMPE']
    test_er_CatB = CatBoostRegressor['testNRMPE']
    test_er_TTR = TTRRegressor['testNRMPE']
    min_er=min(test_er_MLPr,test_er_RandomF,test_er_KNR,test_er_LM,test_er_LGBM,test_er_XGB,test_er_GBR,test_er_CatB,test_er_TTR)
    if min_er==test_er_MLPr:
        model="MLPReg"
        test_error=MLPRegressor['testNRMPE']
        train_error=MLPRegressor['trainNRMPE']
        bp=MLPRegressor['bestparam']
        filename = name +'-MLP.sav'
        pickle.dump(modelMLPR, open(filename, 'wb'))
    elif min_er==test_er_RandomF:
        model="RandomForest"
        test_error=RandomForest['testNRMPE']
        train_error=RandomForest['trainNRMPE']
        bp=RandomForest['bestparam']
        filename = name + '-RandomForest.sav'
        pickle.dump(modelRF, open(filename, 'wb'))
    elif min_er==test_er_KNR:
        model="KNN"
        test_error=KNeighborsRe['testNRMPE']
        train_error=KNeighborsRe['trainNRMPE']
        bp=KNeighborsRe['bestparam']
        filename = name +'-KNN.sav'
        pickle.dump(modelKNN, open(filename, 'wb'))
    elif min_er==test_er_LM:
        model="Linear"
        test_error=LinealModel['testNRMPE']
        train_error=LinealModel['trainNRMPE']
        bp=LinealModel['bestparam']
        filename = name + '-Linear.sav'
        pickle.dump(modelL, open(filename, 'wb'))
    elif min_er==test_er_LGBM:
        model="LGBM"
        test_error=LGBMRegressor['testNRMPE']
        train_error=LGBMRegressor['trainNRMPE']
        bp=LGBMRegressor['bestparam']
        filename = name + '-LGBMRegressor.sav'
        pickle.dump(modelRF, open(filename, 'wb'))
    elif min_er==test_er_XGB:
        model="XGBoost"
        test_error=XGBoostRegressor['testNRMPE']
        train_error=XGBoostRegressor['trainNRMPE']
        bp=XGBoostRegressor['bestparam']
        filename = name + '-XGBoostRegressor.sav'
        pickle.dump(modelRF, open(filename, 'wb'))
    elif min_er==test_er_GBR:
        model="GBR"
        test_error=GBRRegressor['testNRMPE']
        train_error=GBRRegressor['trainNRMPE']
        bp=GBRRegressor['bestparam']
        filename = name + '-GBRRegressor.sav'
        pickle.dump(modelRF, open(filename, 'wb'))
    elif min_er==test_er_CatB:
        model="CatBoost"
        test_error=CatBoostRegressor['testNRMPE']
        train_error=CatBoostRegressor['trainNRMPE']
        bp='Defaults'
        filename = name + '-CatBoostRegressor.sav'
        pickle.dump(modelRF, open(filename, 'wb'))
    elif min_er==test_er_TTR:
        model="TTR"
        test_error=TTRRegressor['testNRMPE']
        train_error=TTRRegressor['trainNRMPE']
        bp='Defaults'
        filename = name + '-TTRRegressor.sav'
        pickle.dump(modelRF, open(filename, 'wb'))

        
        
    dict={"best model":model,"train error":train_error, "test error": test_error, "best param": bp}
    
    return(dict)


def feat_select_RFECV(data,y_name,X):
    from sklearn.pipeline import Pipeline
    class RfePipeline(Pipeline):
        @property
        def coef_(self):
            return self._final_estimator.coef_
        @property
        def feature_importances_(self):
            return self._final_estimator.feature_importances_
    MinMax_scaler = MinMaxScaler()
    #pipeline_LGBM=RfePipeline([('scaler', MinMax_scaler),('LGBM',LGBMRegressor())])
    pipeline_LR=RfePipeline([('scaler', MinMax_scaler),('LR',LinearRegression())])
    
    y=data[y_name]       
    #X = data.drop(columns=[y_name])
    #rfe = RFECV(estimator=pipeline_LGBM, n_jobs=-1,min_features_to_select=2, cv=8, scoring='neg_root_mean_squared_error')
    rfe = RFECV(estimator=pipeline_LR, n_jobs=-1,min_features_to_select=2, cv=8, scoring='neg_root_mean_squared_error')
    rfe.fit(X, y)
    selected_features = []
    for i in range(X.shape[1]):
        if rfe.support_[i]:
            selected_features.append(X.columns[i])
    return selected_features


def feat_select_SFS(data,y_name,X):
    from sklearn.pipeline import Pipeline
    class RfePipeline(Pipeline):
        @property
        def feature_importances_(self):
            return self._final_estimator.feature_importances_
    MinMax_scaler = MinMaxScaler()
    #pipeline_LGBM=RfePipeline([('scaler', MinMax_scaler),('LGBM',LGBMRegressor())])
    pipeline_LR=RfePipeline([('scaler', MinMax_scaler),('LR',LinearRegression())])
       
    y=data[y_name]     
    #X = data.drop(columns=[y_name])  
    #sfs1 = SFS(pipeline_LGBM,n_features_to_select='auto', tol=0.001, direction='forward', cv=8, scoring='neg_root_mean_squared_error', n_jobs=-1)
    sfs1 = SFS(pipeline_LR,n_features_to_select='auto', tol=0.001, direction='forward', cv=8, scoring='neg_root_mean_squared_error', n_jobs=-1)
    sfs1=sfs1.fit(X, y)
    selected_features = []
    for i in range(X.shape[1]):
        if sfs1.support_[i]:
            selected_features.append(X.columns[i])
    return selected_features

def feat_selection_results(data, y_name, X):
   

    selected_features_sfs = feat_select_SFS(data, y_name, X)
    selected_features_rfecv = feat_select_RFECV(data, y_name, X)

    common_features = list(set(selected_features_sfs) & set(selected_features_rfecv))
    
    results = pd.DataFrame()
    max_length = max(len(selected_features_sfs), len(selected_features_rfecv))
    selected_features_sfs += [None] * (max_length - len(selected_features_sfs))
    selected_features_rfecv += [None] * (max_length - len(selected_features_rfecv))
    common_features += [None] * (max_length - len(common_features))
    results['Selected Features (SFS)'] = selected_features_sfs
    results['Selected Features (RFECV)'] = selected_features_rfecv
    results['Common Selected Features'] = common_features
    filtered_items = filter(lambda item: item is not None, common_features)
    common_features_filt = list(filtered_items)
    #yname = pd.DataFrame([y_name], columns=['y_name'])
    #all_results= pd.concat([results, yname],axis = 1)
    #filename = r'D:\ITI\SIROKO\Mill_Data\New_data_metrics\AutoFeat_select_LGBM.xlsx'
    #with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
           # all_results.to_excel(writer, sheet_name=str(y_name), index=False)
    return common_features_filt



X_inputs = df_siroko[[
'SeparatorSpeed',
'MillInletSubpressure',
'MillOutletSubpressure',
'WaterFlow',
'MillFeed',
'BagFilter',
'MillInletTemperature', 
'Pozzolana%',
'Limestone%',
'FlyAsh%',
'Gypsum%',
'GrindingPressure',
'GrindingAidPV',
'CementType',
'GrindingLayerRoller3'
]]


x1=feat_selection_results(data,"MillKW",X_inputs)
x2=feat_selection_results(data,"MillDifferentialPressure",X_inputs)
x3=feat_selection_results(data,"SeparatorPower",X_inputs)
x4=feat_selection_results(data,"EnvironmentalDust",X_inputs)
x5=feat_selection_results(data,"Blaine",X_inputs)
x6=feat_selection_results(data,"MillExitTemperature",X_inputs)
x7=feat_selection_results(data,"Residue",X_inputs)



X1=data[x1]
y1=data[["MillKW"]]
#Mill_Motor = minimum("MillKW", X1,y1)


X2=data[x2]
y2=data[["MillDifferentialPressure"]]
#Mill_dp = minimum("MillDifferentialPressure", X1,y2)

X3=data[x3]
y3=data[["SeparatorPower"]]
#Separ_Motor = minimum("SeparatorPower", X1,y3)

X4=data[x4]
y4=data[["EnvironmentalDust"]]
#Env_Dust = minimum("EnvironmentalDust", X4,y4)

X5=data[x5]
y5=data[["Blaine"]]
#Blaine = minimum("Blaine", X5,y5)

X6=data[x6]
y6=data[["MillExitTemperature"]]
#MillExitTemperature = minimum("MillExitTemperature", X6,y6)

X7=data[x7]
y7=data[["Residue"]]
#Residue = minimum("Residue", X7,y7)



#MillFanLoad = minimum("MillFanLoad", X8,y8)
#SpecificAirFlow = minimum ("SpecificAirFlow",X9,y9)
#vib= minimum("Vibrations",X10,y10)
#vib1 = minimum("Vibrations",X11,y10)   <-



