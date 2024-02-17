from flask import Flask, request, jsonify, json, render_template,redirect, url_for
import json
import pandas as pd
import joblib
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pickle import load
from scipy.optimize import minimize,basinhopping,differential_evolution
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)

@app.route("/de", methods=['POST'])
def get_params():
    
    try:
        request_body = request.get_json()
        
    except Exception as e:
        print("====================" + str(e) + "====================")
    
    optim(request_body)#,manipul_variables_names
    
    return jsonify({"The differential evolution optimal values are estimated":200})
def optim(platfdict):#,manipul_variables_names
    if platfdict['Mode']=='Mill':
        #HERE CAN READ DATA FROM THE DATABASE
        df =  pd.read_csv('data_all.csv', index_col=0)
        df=pd.read_csv('final_df.csv',index_col=0).iloc[3600:3900]
        selected_folder = "models_new"#'models_new'
        with open ('vars3.json', 'r') as f:
            varbs = json.load(f)
        
        ind=0
        
    if platfdict['Mode']=='Kiln':
        df = pd.read_csv('subdata_up135_1.csv', index_col=0)
        df_last = df.iloc[:,-1]
        selected_folder = 'kilnmodels'
        with open ('vars4.json', 'r') as f:
            varbs = json.load(f)
        ind=1
    with open ('mapping1.json', 'r') as f:
       maps  = json.load(f)
     
    chosen_variables=platfdict['Chosen_variables']
    chosen_variables=platfdict['Chosen_variables']
    chosen_variables_names=list((list(chosen_variables_nm.keys()) for chosen_variables_nm in chosen_variables))
    chosen_variables_names = [var_unique for var in chosen_variables_names for var_unique in var]
    mode = platfdict['Mode']
    df_input = df[chosen_variables_names]
    df_var = df[list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())].reset_index(drop=True)
    means =df_input.mean(axis=0).values
    stds = df_input.std(axis=0).values
    mins = df_input.min().values
    maxes = df_input.max().values
    means_mp=df_var.mean(axis=0)
    stds_mp = df_var.std(axis=0).values
    mins_mp = df_var.min().values
    maxes_mp = df_var.max().values
    means_all = df.mean()
    def opt(x):
        f=0
        for i in range(len(chosen_variables_names)):
            
            keyscurr_keys=chosen_variables[i][chosen_variables_names[i]].keys()
            keyscurr = chosen_variables[i][chosen_variables_names[i]]
            key = list(chosen_variables[i].keys())[0]
            scaler = MinMaxScaler()
            if not os.path.exists(os.path.join(os.getcwd(),'scalers1')+str(key)+'_x_scaler.pkl'):
                if not key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys()):
                    #scaler = MinMaxScaler()
                    X = df[varbs['{}'.format(key)].split(',')]
                    X = scaler.fit(X).transform(X)
                    with open('scalers1/'+ key + '_x_scaler.pkl', 'wb') as fid:
                            pickle.dump(scaler, fid)   
                else:
                    #scaler = MinMaxScaler()
                    X = df[key].values
                    X = scaler.fit(X.reshape(-1, 1)).transform(X.reshape(-1, 1))
                    with open('scalers1/'+ key + '_x_scaler.pkl', 'wb') as fid:
                            pickle.dump(scaler, fid) 
            scaler_X = load(open('scalers1/'+ key +'_x_scaler.pkl', 'rb'))
            if not os.path.exists(os.path.join(os.getcwd(),'scalers1')+str(key)+'_y_scaler.pkl'):
                #scaler = MinMaxScaler()
                Y = df[key].values
                Y = scaler.fit(Y.reshape(-1, 1)).transform(Y.reshape(-1, 1))
                with open('scalers1/' + key + '_y_scaler.pkl', 'wb') as fid:
                    pickle.dump(scaler, fid)
            scaler_Y = load(open('scalers1/' + key +'_y_scaler.pkl', 'rb'))
            
            for file in os.listdir("./{}".format(selected_folder)):
                if file.split('-')[0]==key:
                    #loaded_model = pickle.load(open('./models_new/{}'.format(file), 'rb'))
                    loaded_model = pickle.load(open("./{}/{}".format(selected_folder,file), 'rb'))
                    #loaded_model = joblib.load(open('./Best_models v3/{}'.format(file), 'rb'))
                    loaded_model.n_jobs = 1
            
            if 'Bounds' in keyscurr_keys and keyscurr["Bounds"]["Used"]!=None:
                if 'Minimum' in keyscurr['Bounds'].keys():
                    minimum = float(keyscurr['Bounds']['Minimum'])
                    if key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys()):
                        tx1.write(str(f)+'\n')
                        tx1.write('--------------------------'+'\n')
                        
                        f+=1000*float(keyscurr['Bounds']['Weight'])*((x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]])/stds[i])**2*(x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]]<minimum)
                        
                    else:
                        f+=1000*float(keyscurr['Bounds']['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])**2*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)<minimum)
                        
                if 'Maximum' in keyscurr['Bounds'].keys():
                    maximum=float(keyscurr['Bounds']['Maximum'])
                    if key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys()):
                        
                        f+=1000*float(keyscurr['Bounds']['Weight'])*((x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]])/stds[i])**2*(x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]]>maximum)
                    else:
                        
                        f+=1000*float(keyscurr['Bounds']['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])**2*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)>maximum)
            
            if 'Monotony' in keyscurr_keys:
                if keyscurr["Monotony"]["Used"]!=None:
                    monot=keyscurr['Monotony']
                    if monot['MaxMin']=='Minimisation':
                        if key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys()):
                            f+=int(monot['Weight'])*(x[[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]][0]["{}".format(key)]])/stds[i]
                        else:
                            f+=int(monot['Weight'])*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i]
                    else:
                        if key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys()):
                            f-=int(monot['Weight'])*(x[i])/stds[i]
                        else:
                            f-=int(monot['Weight'])*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i]
            elif 'Target' in keyscurr_keys:
                if keyscurr["Target"]["Used"]!=None:
                    target=keyscurr['Target']
                    setp=float(target['SetPoint'])
                if key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys()):
                    f+=int(target['Weight'])*((x[i]-setp)/stds[i])**2
                else:
                    f+=int(target['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)-setp)/stds[i])**2
        return f 
    sett=platfdict['Settings']
    mins_mp = df_var.quantile(0.005).values
    maxes_mp = df_var.quantile(0.995).values
    bounds=[(mins_mp[i],maxes_mp[i]) for i in range(0,len(list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())))]
    res=differential_evolution(opt,bounds = bounds,maxiter=int(sett['Maximum number of iterations']),popsize=int(sett['Population size']),disp=True)
    actual=opt(df[list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())].mean())
    return actual#,res#,actual
    

def init(var,mode,ind,varbs,maps,means,scaler_X,scaler_Y,loaded_model,x):
    pred_matrix = []
    for item in varbs['{}'.format(var)].split(','):
        if item in list(maps['Manipulated Variables'][ind]['{}'.format(mode)].keys()):
            pred_matrix.append(x[maps['Manipulated Variables'][ind]['{}'.format(mode)]['{}'.format(item)]])
        else:
            pred_matrix.append(round(means['{}'.format(item)],4))              
    pred_matrix = scaler_X.transform(np.asarray(pred_matrix).reshape(1, -1)) 
    pr=loaded_model.predict(np.asarray([np.asarray(pred_matrix).reshape(1, -1)]).reshape(1, -1))
    predt = scaler_Y.inverse_transform(np.asarray((loaded_model.predict(np.asarray([np.asarray(pred_matrix).reshape(1, -1)]).reshape(1, -1)))).reshape(1, -1))[0][0]
    return predt


app.run(host='localhost', port=5003)
