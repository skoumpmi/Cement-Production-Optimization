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
        #get the data
        ##df =  pd.read_csv('data_all.csv', index_col=0)
        #df_input = df[['Mill Feed', 'Separator Speed', 'Grinding Pressure', 'Water Flow', 'Mill Inlet Subpressure', 
        #'Mill Outlet Subpressure', 'Mill Fan Load', 'Bag Filter', 'Mill KW','Separator Power','Mill Vibrations']]
        #manipul_variables_names = ['Mill Feed','Mill Inlet Subpressure','Mill Outlet Subpressure','Water Flow','Separator Speed','Mill Inlet Temperature',
        #'Grinding Pressure']
    except Exception as e:
        print("====================" + str(e) + "====================")
    #if __name__ == '__main__':
    optim(request_body)#,manipul_variables_names
    
    return jsonify({"The differential evolution optimal values are estimated":200})
def optim(platfdict):#,manipul_variables_names
    if platfdict['Mode']=='Mill':
        #df =  pd.read_csv('data_all.csv', index_col=0)
        df=pd.read_csv('final_df.csv',index_col=0).iloc[3600:3900]
        df_last = df.iloc[:,-1]
        selected_folder = "models_new"#'models_new'
        with open ('vars3.json', 'r') as f:
            varbs = json.load(f)
        #with open ('vars_old.json', 'r') as f:
            #varbs = json.load(f)
        ind=0
        #with open ('mapping.json', 'r') as f:
            #maps = json.load(f)
    #C:\Users\skoumpmi\PycharmProjects\siroko\klin new try
    if platfdict['Mode']=='Kiln':
        #df =  pd.read_csv('df_Kiln.csv', index_col=0)
        #df = df[(df['Totalfeed']>135.0)] 'subdata_up135_1.csv'
        #df = pd.read_csv('subdata_up135.csv', index_col=0)
        df = pd.read_csv('subdata_up135_1.csv', index_col=0)
        df_last = df.iloc[:,-1]
        selected_folder = 'kilnmodels'#'klin new try'
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
    #"""
    df_input = df[chosen_variables_names]
    print(df_input.mean())
    #df_var = df[manipul_variables_names]
    #print(list(maps['Manipulated Variables'][ind]))
    df_var = df[list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())].reset_index(drop=True)
    #print('-----------------DF VAR----------------------------')
    #print(df_var)
    #breakpoint()
    means =df_input.mean(axis=0).values
    stds = df_input.std(axis=0).values
    print(means)
    print(stds)
    print([means[i]+3*stds[i] for i in range(0,len(chosen_variables_names))])
    print([means[i]-3*stds[i] for i in range(0,len(chosen_variables_names))])
    #print(stds)
    mins = df_input.min().values
    maxes = df_input.max().values
    #means_mp =df_var.mean(axis=0).values
    means_mp=df_var.mean(axis=0)
    #print('-----MEANS MP------------------')
    #print(means_mp)
    stds_mp = df_var.std(axis=0).values
    mins_mp = df_var.min().values
    maxes_mp = df_var.max().values
    means_all = df.mean()
    tx1 = open("penalties3.txt", "w")
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
            #C:\Users\skoumpmi\PycharmProjects\siroko\Best_models v3
            #for file in os.listdir("./Best_models v3"):
            ##for file in os.listdir("./models_new"):
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
                        #print('minimum penalty for var {} is:{}'.format(str(key),str(1000*float(keyscurr['Bounds']['Weight'])*((x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]])/stds[i])**2*(x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]]<minimum))))
                        #print([maps["Manipulated Variables"][ind][['{}'.format(platfdict['Mode'])]]["{}".format(key)]])
                        f+=1000*float(keyscurr['Bounds']['Weight'])*((x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]])/stds[i])**2*(x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]]<minimum)
                        #f+=1000*float(keyscurr['Bounds']['Weight'])*((x[mapping['Mill Feed]])/stds["mill Feed"])**2*(x[mapping['Mill Feed]<minimum)
                        tx1.write('minimum penalty'+'\n')
                        tx1.write(key+'\n')
                        tx1.write(str(1000*float(keyscurr['Bounds']['Weight'])*((x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]])/stds[i])**2*(x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]]<minimum))+'\n')
                    else:
                        f+=1000*float(keyscurr['Bounds']['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])**2*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)<minimum)
                        #print('minimum penalty for var {} is:{}'.format(str(key),str(1000*float(keyscurr['Bounds']['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])**2*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)<minimum))))
                        
                        tx1.write('minimum penalty'+'\n')
                        tx1.write(key+'\n')
                        #tx1.write(str(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))+'\n')
                        tx1.write(str(1000*float(keyscurr['Bounds']['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])**2*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)<minimum))+'\n')
                if 'Maximum' in keyscurr['Bounds'].keys():
                    maximum=float(keyscurr['Bounds']['Maximum'])
                    if key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys()):
                        tx1.write('maximum penalty'+'\n')
                        tx1.write(key+'\n')
                        tx1.write(str(1000*float(keyscurr['Bounds']['Weight'])*((x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]])/stds[i])**2*(x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]]>maximum))+'\n')
                        #print('maximum penalty for var {} is:{}'.format(str(key),str(1000*float(keyscurr['Bounds']['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])**2*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)<minimum))))
                        f+=1000*float(keyscurr['Bounds']['Weight'])*((x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]])/stds[i])**2*(x[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]["{}".format(key)]]>maximum)
                    else:
                        tx1.write('mmaximum penalty'+'\n')
                        tx1.write(key+'\n')
                        #tx1.write(str(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))+'\n')
                        tx1.write(str(1000*float(keyscurr['Bounds']['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])**2*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)>maximum))+'\n')
                        f+=1000*float(keyscurr['Bounds']['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])**2*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)>maximum)
            #tx1.write(str(f)+'\n')
            if 'Monotony' in keyscurr_keys:
                if keyscurr["Monotony"]["Used"]!=None:
                    monot=keyscurr['Monotony']
                    if monot['MaxMin']=='Minimisation':
                        if key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys()):
                            #print(key)
                            #print([maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]][0])
                            #print([maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]][0]["{}".format(key)])
                            #if key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict[['{}'.format(platfdict['Mode'])]])].keys()):
                            f+=int(monot['Weight'])*(x[[maps["Manipulated Variables"][ind]['{}'.format(platfdict['Mode'])]][0]["{}".format(key)]])/stds[i]
                            #f+=int(monot['Weight'])*(x[maps["Manipulated Variables"][0][['{}'.format(platfdict['Mode'])]]["{}".format(key)]])/stds[i]
                            #tx1.write('Monotony'+'\n')
                            #tx1.write(key+'\n')
                            #tx1.write(str(int(monot['Weight'])*(x[maps["Manipulated Variables"][0][['{}'.format(platfdict['Mode'])]]["{}".format(key)]])/stds[i])+'\n')
                        else:
                            f+=int(monot['Weight'])*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i]
                            tx1.write('Monotony'+'\n')
                            tx1.write(key+'\n')
                            tx1.write(str(int(monot['Weight'])*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])+'\n')
                    else:
                        if key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys()):
                            f-=int(monot['Weight'])*(x[i])/stds[i]
                            #tx1.write('Monotony'+'\n')
                            #tx1.write(key+'\n')
                            #tx1.write(str(-int(monot['Weight'])*(x[i])/stds[i])+'\n')
                        else:
                            f-=int(monot['Weight'])*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i]
                            #tx1.write('Monotony'+'\n')
                            #tx1.write(key+'\n')
                            #tx1.write(str(-int(monot['Weight'])*(init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x))/stds[i])+'\n')
            elif 'Target' in keyscurr_keys:
                if keyscurr["Target"]["Used"]!=None:
                    target=keyscurr['Target']
                    setp=float(target['SetPoint'])
                if key in list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys()):
                    tx1.write('Target'+'\n')
                    tx1.write(key+'\n')
                    tx1.write(str(setp)+'\n')
                    tx1.write(str(int(target['Weight'])*((x[i]-setp)/stds[i])**2)+'\n')
                    f+=int(target['Weight'])*((x[i]-setp)/stds[i])**2
                else:
                    tx1.write('Target'+'\n')
                    tx1.write(key+'\n')
                    tx1.write(str(setp)+'\n')
                    tx1.write(str(int(target['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)-setp)/stds[i])**2)+'\n')
                    f+=int(target['Weight'])*((init(key,mode,ind,varbs,maps,means_all,scaler_X,scaler_Y,loaded_model,x)-setp)/stds[i])**2
            tx1.write('--------------------------'+'\n')
        #print(f)
        return f 
    sett=platfdict['Settings']
    mins_mp = df_var.quantile(0.005).values
    maxes_mp = df_var.quantile(0.995).values
    bounds=[(mins_mp[i],maxes_mp[i]) for i in range(0,len(list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())))]
    
    ##bounds=[(round(0.7*list(df_var)[i],4), round(1.3*list(df_var)[i],4)) for i in range(0,len(list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())))]
    #bounds=[(round(means_mp[i]-3*stds_mp[i],4), round(means_mp[i]+3*stds_mp[i],4)) for i in range(0,len(list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())))]
    #bounds=[means_mp for i in range(0,len(list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())))]
    #bounds = means_mp
    print(bounds)
    #breakpoint()
    #breakpoint()
    #bounds=[(round(0.7*list(df_last)[i],4), round(1.3*list(df_last)[i],4)) for i in range(0,len(list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())))]
    print(bounds)
    res=differential_evolution(opt,bounds = bounds,maxiter=int(sett['Maximum number of iterations']),popsize=int(sett['Population size']),disp=True)
    #actual=opt(df[list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())].values[-1])
    actual=opt(df[list(maps['Manipulated Variables'][ind]['{}'.format(platfdict['Mode'])].keys())].mean())
    print(res)
    print(res.x)


    print(actual)
    return actual#,res#,actual
    

def init(var,mode,ind,varbs,maps,means,scaler_X,scaler_Y,loaded_model,x):
    pred_matrix = []
    for item in varbs['{}'.format(var)].split(','):
        if item in list(maps['Manipulated Variables'][ind]['{}'.format(mode)].keys()):
            pred_matrix.append(x[maps['Manipulated Variables'][ind]['{}'.format(mode)]['{}'.format(item)]])
        else:
            pred_matrix.append(round(means['{}'.format(item)],4))              
    #if var not in ['Blaine', 'Residue']:
    pred_matrix = scaler_X.transform(np.asarray(pred_matrix).reshape(1, -1)) 
    ##print('pred mateix is')
    ##print(pred_matrix)
    ##print(np.asarray([np.asarray(pred_matrix).reshape(1, -1)]).reshape(1, -1))
    #pr=loaded_model.predict(np.asarray([np.asarray(pred_matrix)[0].reshape(1, -1)])[0].reshape(1, -1))
    pr=loaded_model.predict(np.asarray([np.asarray(pred_matrix).reshape(1, -1)]).reshape(1, -1))
    #print(pr)
    #predt = scaler_Y.inverse_transform(np.asarray((loaded_model.predict(np.asarray(pred_matrix)[0].reshape(1, -1))[0])).reshape(1, -1))[0][0]
    predt = scaler_Y.inverse_transform(np.asarray((loaded_model.predict(np.asarray([np.asarray(pred_matrix).reshape(1, -1)]).reshape(1, -1)))).reshape(1, -1))[0][0]
    #print('var is:{}'.format(var))
    #print(predt)
    #if var not in ['Blaine', 'Residue']:
    #predt = scaler_Y.inverse_transform(np.asarray((loaded_model.predict(np.asarray(pred_matrix).reshape(1, -1))[0])).reshape(1, -1))[0][0]
    #####predt = scaler_Y.inverse_transform(np.asarray((loaded_model.predict(np.asarray(pred_matrix)[0].reshape(1, -1))[0])).reshape(1, -1))[0][0]
    #else:
    #predt = loaded_model.predict(np.asarray(pred_matrix).reshape(1, -1))[0][0]
    return predt

#if __name__ == '__main__':
#,"Grinding Layer Roller 3":7,"Bag Filter":8,"Grinding Aid PV":9
app.run(host='localhost', port=5003)
