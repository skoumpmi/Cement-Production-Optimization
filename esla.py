import json
import pandas as pd
import joblib
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pickle import load
from scipy.optimize import minimize,basinhopping,differential_evolution
from datetime import datetime, timedelta
from db import Database
from pymongo import ASCENDING,DESCENDING
from mill_models import TrainMillModel
from kiln_models import TrainKilnModel
import warnings
warnings.filterwarnings("ignore")
# app = Flask(__name__)


### get active models 
### chosen variables filter out what is not referenced 
### run model_loading_feat for every trained model after the filtering step
### load data from mongodb (most recent 6 months) (transform values to float64)
### 

class Optimization:
    Mill = 'Mill'
    Kiln = 'Kiln'
     
    manipulated_variables_by_machine_type =  {
        Mill: {
            "Manipulated Variables": {
                "Mill Feed":0,
                "Mill Inlet Subpressure":1,
                "Mill Outlet Subpressure":2,
                "Water Flow":3,
                "Separator Speed":4,
                "Mill Inlet Temperature":5, 
                "Grinding Pressure":6
            }
        },
        Kiln: {
            "Manipulated Variables": {
                "Totalfeed":0, ## -> same manipulated as mentioned in resolver.py
                # "HeatMainBurner":1,
                # "KilnInletTemp":2,
                # "PressTransportAir":3,
                # "PressMASAir":4,
                "PreheaterO2":1,  ## should change to 1?
                # "KilnInletPress":6,
                # "ClinkerCaO":7,
                # "CurrentIDFan":8,
                "SolidFuelFeed":2, ## should change to 2?
                # "KilnHoodPress":10,
                # "KilnSpeed":11,
                # "KilnVortexTemp":12,
                # "SecondaryAirTemp":13,
                # "TotalAirFlow":14,
                # "KilnFeedLSF":15
            }
        }
    }
	
 
         
    modelTypeToDatabaseCollection = {
        Mill:'data-2-csvvalues',
        Kiln:'kilnvalues'
    }


    @staticmethod
    def to_scaler_path(trained_models,variable_name,machine_type,means_all,scaler_type='X'):
        trained_model = None 
        for model in trained_models:
            if model.get('Model Variable','') == variable_name:
                trained_model = model 
                break
        
        if trained_model:
            prefix = Optimization.get_clustering_prefix_by_machine_type(machine_type,trained_model,means_all)
            best_model = trained_model['Train Result']['Metadata']['best model']
            model_id = trained_model.get('_id')
            path = os.path.join(os.getcwd(),'scalers',f"{prefix}{model_id}_{best_model}_{variable_name}{scaler_type.upper()}_scaler.pkl") 
        else:
            path = os.path.join(os.getcwd(),'scalers',f"ESLA_GENERATED_SCALER_{variable_name}{scaler_type.upper()}_scaler.pkl") 
        
        return path 
    
    @staticmethod
    def get_clustering_prefix_by_machine_type(machine_type,model,means_all):
        thresshold = model['Train Result']['Metadata'].get('thresshold')
        if not thresshold:
            return ""
        try:
            thresshold_numeric = float(thresshold)
        except:
            return ""


        if machine_type == Optimization.Mill:
            value =  means_all['Mill Feed']
        elif machine_type == Optimization.Kiln:
            value = means_all['Totalfeed']
        else:
            raise Exception('Unknown machine type')
        
        return "cluster-1-" if thresshold_numeric < value else "cluster-2-" 


    @staticmethod
    def to_best_model_path(trained_models,variable_name,machine_type,means_all):
        trained_model = None 
        for model in trained_models:
            if model.get('Model Variable','') == variable_name:
                trained_model = model 
                break
        if not trained_model:
            return None 
        
        if trained_model:
            best_model = trained_model['Train Result']['Metadata']['best model']

             

            prefix = Optimization.get_clustering_prefix_by_machine_type(machine_type,trained_model,means_all)
            # cluster_prefix = thresshold < 
            if best_model == 'GBR':
                 best_model = 'GBRRegressor'
            if best_model == 'CatBoost':
                best_model = 'CatBoostRegressor'
            if best_model == 'XGBoost':
                best_model = 'XGBoostRegressor'
            if best_model == 'TTR':
                best_model = 'TTRegressor'
            if best_model == 'MLPReg':
                best_model = 'MLP'
            if best_model == 'LGBM':
                best_model = "LGBMRegressor"
            model_id = trained_model.get('_id')
            path = os.path.join(os.getcwd(),'best_model',f"{prefix}{model_id}-{variable_name}-{best_model}.sav") 
   
        return path 

    @staticmethod
    def load_latest_n_days_data(optimization_model,days=90):
        model_type = optimization_model.get('machineType')
        ## 2880 -> aggregated values of 30 seconds for each day
        total_number_of_results = 2880 * days
        collection_name = Optimization.modelTypeToDatabaseCollection[model_type]
        from_date = datetime.now() - timedelta(days=days)
        date_string = from_date.strftime("%Y-%m-%d %H:%M:%S")
        # count_by_date = Database.collection(collection_name).count_documents({"times":{"$gte":date_string}})
        # if count_by_date >= total_number_of_results:
        # date_range = {"$and":[{"times":{"$lte":'2022-03-01'}},{"times":{"$gte":'2021-12-01'}}]} if model_type == Optimization.Mill else {"$and":[{"times":{"$lte":'2023-04-01'}},{"times":{"$gte":'2023-01-01'}}]}
        raw_data = Database.collection(collection_name).find({"times":{"$gte":date_string}}).sort("times",ASCENDING)
        # else:
        #     raw_data = Database.collection(collection_name).find().sort("times",DESCENDING).limit(int(total_number_of_results))
        # raw_data = Database.collection(collection_name).find({"$and":[{"times":{"$gte":'2021-12-01'}},{"times":{"$lte":'2021-12-31'}}]}).sort("times",ASCENDING)
        data = list()
        for item in raw_data:
            try:
                item['times'] = " ".join(item['times'].split('T'))
                item['times'] = item['times'].split('.')[0]
                try:
                    datetime.strptime(item['times'], "%Y-%m-%d %H:%M:%S")
                except:
                    continue 
                data.append(item)
            except Exception as e:
                data.append(item) 
            if model_type == Optimization.Mill:
                        try:
                            if not item.get('Specific Air Flow') and (item.get('Mill Feed') and float(item.get('Mill Feed')) > 0):
                                item["Specific Air Flow"]=3600*(((10 * (float(item["Mill Fan Load"])) ) / (1.287 * (abs(float(item["Mill Outlet Subpressure"])) - abs(float(item["Mill Inlet Subpressure"])))  + float(item["Bag Filter"]))) / float(item["Mill Feed"]))
                        except Exception as e: 
                            pass 

        df = pd.DataFrame(data)
        
          
        
        
        if model_type == Optimization.Mill:
            df = TrainMillModel.preproc(df)
        elif model_type == Optimization.Kiln:
            df = TrainKilnModel.preproc(df)
        else:
            raise Exception('Unknown machine type')
        if "__v" in df.columns:
            df = df.drop(['__v'],axis=1)
        if "_id" in df.columns:
            df = df.drop(['_id'],axis=1)
        if "times" in df.columns:
            df = df.drop(['times'],axis=1)
         
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.astype(np.float64)
        df = df.dropna()
        # print(df)
        return df 

    @staticmethod
    def get_chosen_variable_names(optimization_model):
        names = []
        for config in optimization_model.get('Chosen_variables',list()):
            keys = list(config.keys())
            name = keys[0]
            names.append(name)
        
        return list(set(names))

    @staticmethod
    def load_model_vars(optimization_model,active_models):
        config = dict()
        chosen_variable_names = Optimization.get_chosen_variable_names(optimization_model)
        for active_model in active_models:
            if active_model.get('Model Variable','') in chosen_variable_names:
                config = {**config,**Optimization.model_loading_feat(active_model)}
        return config 
    
    @staticmethod
    def model_loading_feat(trained_model): 
        return {trained_model['Model Variable']: ",".join(trained_model['Train Result']['Metadata']['selected_features'])}
    

    @staticmethod
    def should_optimize_by_machine_criteria(machine_type,latest_data_entry,dataset):
        if machine_type == Optimization.Mill:
           return latest_data_entry['Mill Feed'] > dataset.quantile(0.1)['Mill Feed'] and latest_data_entry['Mill KW'] > dataset.quantile(0.1)['Mill KW']  
        elif machine_type == Optimization.Kiln:
           return latest_data_entry['KilnAmps'] > dataset.quantile(0.1)['KilnAmps']  and latest_data_entry['Totalfeed'] > dataset.quantile(0.1)['Totalfeed'] 
        else:
            raise Exception(f'Unknown machine type: {machine_type}')

    @staticmethod
    def optim(platfdict,trained_models,input_data):#,manipul_variables_names
         
        machine_type = platfdict.get('machineType')
        if not machine_type or machine_type not in [Optimization.Mill,Optimization.Kiln]:
            raise Exception(f"Unknown machine type {machine_type}")
        varbs = Optimization.load_model_vars(platfdict,trained_models)
        
        map_keys =list( Optimization.manipulated_variables_by_machine_type[machine_type]['Manipulated Variables'].keys())

        
        df = Optimization.load_latest_n_days_data(platfdict,180)

        chosen_variables=platfdict['Chosen_variables']
        chosen_variables=platfdict['Chosen_variables']
        chosen_variables_names=list((list(chosen_variables_nm.keys()) for chosen_variables_nm in chosen_variables))
        chosen_variables_names = [var_unique for var in chosen_variables_names for var_unique in var]
        maps = {"Manipulated Variables":{}}
        for i in range(len(chosen_variables_names)):
            # if chosen_variables_names[i] in map_keys:
            maps["Manipulated Variables"][chosen_variables_names[i]] = i 

        for key in map_keys:
            if not maps['Manipulated Variables'].get(key):
                maps['Manipulated Variables'][key] = len(list(maps.keys())) 



       
        df_input = df[chosen_variables_names]
        #df_var = df[manipul_variables_names]
        df_var = df[list(maps['Manipulated Variables'].keys())]
        df_last = df_var.iloc[-1]
        # means =df_input.mean(axis=0).values
        stds = df_input.std(axis=0).values
        # mins = df_input.min().values
        # maxes = df_input.max().values
        # means_mp =df_var.mean(axis=0).values
        # stds_mp = df_var.std(axis=0).values
        # mins_mp = df_var.quantile(0.25).values
        # maxes_mp = df_var.quantile(0.75).values
        means_all = df.mean()


        if not Optimization.should_optimize_by_machine_criteria(machine_type,df_last,df):
            return None
        
        def opt(x):
            f=0
            for i in range(len(chosen_variables_names)):
                keyscurr_keys=chosen_variables[i][chosen_variables_names[i]].keys()
                keyscurr = chosen_variables[i][chosen_variables_names[i]]
                key = list(chosen_variables[i].keys())[0]
                scaler = MinMaxScaler()
                scaler_x_path = Optimization.to_scaler_path(trained_models,key,machine_type,means_all,'X')
                scaler_y_path = Optimization.to_scaler_path(trained_models,key,machine_type,means_all,'Y')
                if not os.path.exists(scaler_x_path):
                    if not key in list(maps['Manipulated Variables'].keys()):
                        #scaler = MinMaxScaler()
                        X = df[varbs['{}'.format(key)].split(',')]
                        X = scaler.fit(X).transform(X)
                        with open(scaler_x_path, 'wb') as fid:
                                pickle.dump(scaler, fid)   
                    else:
                        #scaler = MinMaxScaler()
                        X = df[key].values
                        X = scaler.fit(X.reshape(-1, 1)).transform(X.reshape(-1, 1))
                        with open(scaler_x_path, 'wb') as fid:
                                pickle.dump(scaler, fid) 
                scaler_X = load(open(scaler_x_path, 'rb'))
                if not os.path.exists(scaler_y_path):
                    #scaler = MinMaxScaler()
                    Y = df[key].values
                    Y = scaler.fit(Y.reshape(-1, 1)).transform(Y.reshape(-1, 1))
                    with open(scaler_y_path, 'wb') as fid:
                        pickle.dump(scaler, fid)
                scaler_Y = load(open(scaler_y_path, 'rb'))

                best_model_path = Optimization.to_best_model_path(trained_models,key,machine_type,means_all)
             
                if best_model_path:
                    loaded_model = pickle.load(open(best_model_path, 'rb'))
              
                if 'Bounds' in keyscurr_keys and keyscurr["Bounds"].get('Used'):
                    if 'Minimum' in keyscurr['Bounds'].keys():
                        minimum = float(keyscurr['Bounds']['Minimum'])
                        if key in list(maps['Manipulated Variables'].keys()):
                            f+=1000*float(keyscurr['Bounds']['Weight'])*((x[maps["Manipulated Variables"]["{}".format(key)]])/stds[i])**2*(x[maps["Manipulated Variables"]["{}".format(key)]]<minimum)
                            #f+=1000*float(keyscurr['Bounds']['Weight'])*((x[mapping['Mill Feed]])/stds["mill Feed"])**2*(x[mapping['Mill Feed]<minimum)
                        else:
                            f+=1000*float(keyscurr['Bounds']['Weight'])*((Optimization.init(key,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])**2*(Optimization.init(key,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)<minimum)
                    if 'Maximum' in keyscurr['Bounds'].keys():
                        maximum=float(keyscurr['Bounds']['Maximum'])
                        if key in list(maps['Manipulated Variables'].keys()):
                            f+=1000*float(keyscurr['Bounds']['Weight'])*((x[maps["Manipulated Variables"]["{}".format(key)]])/stds[i])**2*(x[maps["Manipulated Variables"]["{}".format(key)]]>maximum)
                        else:
                            f+=1000*float(keyscurr['Bounds']['Weight'])*((Optimization.init(key,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])**2*(Optimization.init(key,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)>maximum)
                if 'Monotony' in keyscurr_keys and keyscurr['Monotony'].get('Used'):
                    if keyscurr["Monotony"].get('Used'):
                        monot=keyscurr['Monotony']
                        if monot['MaxMin']=='Minimisation':
                            if key in list(maps['Manipulated Variables'].keys()):
                                f+=int(monot['Weight'])*(x[maps["Manipulated Variables"]["{}".format(key)]])/stds[i]
                            else:
                                f+=int(monot['Weight'])*(Optimization.init(key,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i]
                        else:
                            if key in list(maps['Manipulated Variables'].keys()):
                                f-=int(monot['Weight'])*(x[i])/stds[i]
                            else:
                                f-=int(monot['Weight'])*(Optimization.init(key,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i]
                elif 'Target' in keyscurr_keys and keyscurr["Target"].get('Used'):
                    # if keyscurr["Target"]["Used"]!=None:
                    target=keyscurr['Target']
                    setp=float(target['SetPoint'])
                    if key in list(maps['Manipulated Variables'].keys()):
                        f+=int(target['Weight'])*((x[i]-setp)/stds[i])**2
                    else:
                        f+=int(target['Weight'])*((Optimization.init(key,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)-setp)/stds[i])**2
                
            return f 
        sett=platfdict['Settings']
        # bounds=[(mins_mp[i],maxes_mp[i]) for i in range(0,len(list(maps['Manipulated Variables'].keys())))]
        
        temp_bounds=[(round(0.7*list(df_last)[i],4), round(1.3*list(df_last)[i],4)) for i in range(0,len(list(maps['Manipulated Variables'].keys())))]
        bounds = list()
        for t in temp_bounds:
            a,b = t 
            bounds.append(tuple(sorted([a,b])))

        res=differential_evolution(opt,bounds = bounds,maxiter=int(sett['Maximum number of iterations']),popsize=int(sett['Population size']),disp=True)
        actual=opt(df[list(maps['Manipulated Variables'].keys())].values[-1])
        i = 0
        # res.
        recommendations = {}
        for key in maps['Manipulated Variables'].keys():
            if key in map_keys:
                recommendations[key] = res.x[i]
            i+=1
        result_config = {
            "optimization_model":str(platfdict.get('_id')),
            "recommendations": recommendations,
            "evaluation": {
                "actual": actual,
                "differential_evolution": res.fun,
                "status": str(res.fun < actual).lower(),
                "status_text":"Good Recommendation" if res.fun < actual else 'Bad Recommendation',
                "comparison_percentage": (res.fun - actual)*100 / actual,
            },
            "number_of_iterations": res.nit
        }
        return result_config

    @staticmethod
    def init(var,varbs,maps,means,scaler_X,scaler_Y,loaded_model,x):
        pred_matrix = []
        for item in varbs['{}'.format(var)].split(','):
            if item in list(maps['Manipulated Variables'].keys()):
                pred_matrix.append(x[maps['Manipulated Variables']['{}'.format(item)]])
            else:
                pred_matrix.append(round(means['{}'.format(item)],4))              
        pr=loaded_model.predict(np.asarray([np.asarray(pred_matrix).reshape(1, -1)]).reshape(1, -1))
        predt = scaler_Y.inverse_transform(np.asarray((loaded_model.predict(np.asarray([np.asarray(pred_matrix).reshape(1, -1)]).reshape(1, -1)))).reshape(1, -1))[0][0]
        return predt
