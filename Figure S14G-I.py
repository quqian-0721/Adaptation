# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:54:23 2022

@author: lenovo
"""




import numpy as np
import pandas as pd
import numpy as np
import copy
import xlrd
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

book = xlrd.open_workbook('data/全国105条河所有环境因素_updated.xlsx')
print('sheet页名称:',book.sheet_names())
sheet = book.sheet_by_index(0)
rows = sheet.nrows
cols = sheet.ncols
data=[]
rows_name=[]
print(0)
for i in range(1,rows):
    temp=[]
    features_name=[]
    for j in range(3,cols):
        if is_number(sheet.cell(i,j).value):
            temp.append(float(sheet.cell(i,j).value))
        else:
            if sheet.cell(i,j).value=='':
                temp.append(np.nan)
            else:
                print(sheet.cell(i,j))
        features_name.append(sheet.cell(0,j).value)
    rows_name.append(sheet.cell(i,1).value)
    data.append(temp)
data=np.array(data)

rivers_type=[]
for i in range(1,rows):
    rivers_type.append(sheet.cell(i,2).value)
rivers_type_dict={}
i=0
for each_type in list(set(rivers_type)):
    rivers_type_dict[each_type]=i
    i+=1
rivers_type_data=[]
for each_type in rivers_type:
    temp=np.zeros(len(rivers_type_dict))
    temp[rivers_type_dict[each_type]]+=1
    rivers_type_data.append(temp)
rivers_type_data=np.array(rivers_type_data)
data=np.hstack((rivers_type_data,data))
features_name=['river type_0','river type_1']+features_name
files_name=[]

for i in range(1,rows):
    files_name.append(sheet.cell(i,0).value)



# have_features_name=['pH','DO','TDS','DOC','TC','TN','叶绿素a（ug/L）','Na','Mg','Al','K','Ca','Cr','Mn','Ni','Cu',\
#                     'Zn','As','Mo','Ag','Cd','IC','NH4-N','Longitude','Latitude']+features_name[50:52]+features_name[58:59]+\
#     features_name[60:62]+features_name[73:74]+features_name[75:76]+features_name[83:85]


have_features_name=['pH','DO','TDS','DOC','Na','Mg','39K（ug/L）','Ca','Mn','Ni','Cu',\
                    'Zn','Cd','NH4-N','Longitude','Latitude']+features_name[50:52]+features_name[58:59]+\
    features_name[60:62]+features_name[75:76]+features_name[83:85]      

# have_features_name=copy.deepcopy(features_name)    
data_dict={}
for i in range(len(files_name)):
    data_dict[files_name[i]]=data[i,:]
# for i in range(2):
#     features_name.pop()

book = xlrd.open_workbook('data/all_env_bac_edited_zero.xlsx')
print('sheet页名称:',book.sheet_names())
sheet = book.sheet_by_index(0)
rows = sheet.nrows
cols = sheet.ncols

new_files_name=[]
all_y=[]
for i in range(1,rows):
    new_files_name.append(sheet.cell(i,0).value)
    all_y.append([float(sheet.cell(i,-2).value),float(sheet.cell(i,-1).value)])

data=[]
for each_name in new_files_name:
    data.append(data_dict[each_name])
# X=np.array(data)

X_unimputed=np.array(data)
all_y=np.array(all_y)

import copy
import numpy as np


have_features_index=[]
for each_feature_name in have_features_name:
    flag=0
    for i in range(len(features_name)):
        if each_feature_name in features_name[i]:
            have_features_index.append(i)
            flag=1
            break
            
    if flag==0:
        print(each_feature_name)
            # break
X_unimputed=X_unimputed[:,have_features_index]
new_features_name=[]
for each_index in have_features_index:
    new_features_name.append(features_name[each_index])
features_name_uncomputed=copy.deepcopy(new_features_name)
import random
from sklearn.utils import resample  
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import RepeatedKFold


from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb    
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
# from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler

ind_svm_r2=[]
ind_xgb_r2=[]
ind_rf_r2=[]
ind_gbr_r2=[]
ind_mlp_r2=[]
    
bagging_ind_svm_r2=[]
bagging_ind_xgb_r2=[]
bagging_ind_rf_r2=[]
bagging_ind_gbr_r2=[]
bagging_ind_mlp_r2=[]


for epochs in range(500):
    i=2
    features_name=copy.deepcopy(features_name_uncomputed)
    split_index=list(range(len(X_unimputed)))
    random.shuffle(split_index)
    independent_test_index=split_index[:10]
    independent_train_index=split_index[10:]
    X_unimputed_test=X_unimputed[independent_test_index]
    X_unimputed_train=X_unimputed[independent_train_index]
    
    all_y_test=all_y[independent_test_index]
    all_y_train=all_y[independent_train_index]
    
    X_unimputed_test_df=pd.DataFrame(X_unimputed_test,columns=features_name)
    X_unimputed_train_df=pd.DataFrame(X_unimputed_train,columns=features_name)
    X_unimputed_test_df.to_csv(f'independent_split/jackknife/X_unimputed_test_{epochs}_{i}.csv',index=None)
    X_unimputed_train_df.to_csv(f'independent_split/jackknife/X_unimputed_train_{epochs}_{i}.csv',index=None)
    
    all_y_test_df=pd.DataFrame(all_y_test,columns=['stage1','stage2'])
    all_y_train_df=pd.DataFrame(all_y_train,columns=['stage1','stage2'])
    all_y_test_df.to_csv(f'independent_split/jackknife/all_y_test_{epochs}_{i}.csv',index=None)
    all_y_train_df.to_csv(f'independent_split/jackknife/all_y_train_{epochs}_{i}.csv',index=None)
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X_unimputed_train)
    
    X_train=imp.transform(X_unimputed_train)
    X_test=imp.transform(X_unimputed_test)
    
    ss = MinMaxScaler()
    scaler = ss.fit(X_train)
    X_train= scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    
    old_X_train=copy.deepcopy(X_train)
    old_features_name=copy.deepcopy(features_name)
    for m in range(old_X_train.shape[1]):
        for j in range(m+1,old_X_train.shape[1]):
            X_train=np.column_stack((X_train,old_X_train[:,m]*old_X_train[:,j]))
            features_name.append(old_features_name[m]+'+'+old_features_name[j])
            
    old_X_test=copy.deepcopy(X_test)
    for m in range(old_X_test.shape[1]):
        for j in range(m+1,old_X_test.shape[1]):
            X_test=np.column_stack((X_test,old_X_test[:,m]*old_X_test[:,j]))
    
    # chose_featrues_name=['DOC (mg/L)','pH+44Ca（ug/L）','pH+60Ni（ug/L）',\
    #                       '44Ca（ug/L）+65Cu（ug/L）',\
    #                           '44Ca（ug/L）+PM2.5年平均年浓度','66Zn（ug/L）+森林覆盖率%',\
    #                               'Longitude+人口密度（人/km²）','地区GDP+全年公路货物运输总量（万吨）',\
    #                                   '人口密度（人/km²）+PM2.5年平均年浓度']
    
    # chose_featrues_name=['pH+人均地区GDP','TDS(mg/L)+44Ca（ug/L）','TDS(mg/L)+人均地区GDP','DOC (mg/L)+65Cu（ug/L）','DOC (mg/L)+森林覆盖率%','24Mg（ug/L）+65Cu（ug/L）',\
    #                       '24Mg（ug/L）+人均地区GDP','24Mg（ug/L）+森林覆盖率%',\
    #                           '44Ca（ug/L）+PM2.5年平均年浓度','55Mn（ug/L）+地区GDP','60Ni（ug/L）+PM2.5年平均年浓度',\
    #                               '66Zn（ug/L）+地区GDP','66Zn（ug/L）+人均地区GDP','66Zn（ug/L）+森林覆盖率%',\
    #                                   'NH4-N(mg/L)+Longitude','地区GDP+全年公路货物运输总量（万吨）',\
    #                                       '人均地区GDP+全市全年旅游总人数(万人次)','人口密度（人/km²）+全市全年旅游总人数(万人次)']
        
    chose_featrues_name=['pH+DO(mg/L)','pH+Longitude','DO(mg/L)+66Zn（ug/L）','DO(mg/L)+111Cd（ug/L）','TDS(mg/L)+人均地区GDP',\
                          'DOC (mg/L)+森林覆盖率%','23Na（ug/L）+地区GDP','24Mg（ug/L）+PM2.5年平均年浓度','44Ca（ug/L）+人均地区GDP',\
                              '55Mn（ug/L）+NH4-N(mg/L)','55Mn（ug/L）+人均地区GDP','66Zn（ug/L）+地区GDP','66Zn（ug/L）+人均地区GDP','111Cd（ug/L）+人均地区GDP','111Cd（ug/L）+全市全年旅游总人数(万人次)',\
                                  'Longitude+森林覆盖率%','Longitude+PM2.5年平均年浓度']
        
    temp_index=[]
    new_features_name=[]
    for m in range(len(features_name)):
        if features_name[m] in chose_featrues_name:
            temp_index.append(m)
            new_features_name.append(features_name[m])
    X_train=X_train[:,temp_index]
    X_test=X_test[:,temp_index]
    
    ss = MinMaxScaler()
    scaler = ss.fit(X_train)
    X_train_svm= scaler.transform(X_train)
    X_test_svm= scaler.transform(X_test)
    new_y=np.sum(all_y_train,1)
    all_y_train=np.column_stack((all_y_train,new_y))
    new_y=np.sum(all_y_test,1)
    all_y_test=np.column_stack((all_y_test,new_y))
    

    nfold=0



    temp_y_train=all_y_train[:,i]
    temp_y_test=all_y_test[:,i]

    
    model_rf=RandomForestRegressor(max_depth=4, random_state=0).fit(X_train_svm, temp_y_train)
    y_pred_rf=model_rf.predict(X_test_svm)

    
    model_bagging_rf=BaggingRegressor(RandomForestRegressor(max_depth=4, random_state=0),n_estimators=10, random_state=0).fit(X_train_svm, temp_y_train)     
    y_pred_bagging_rf=model_bagging_rf.predict(X_test_svm)

            
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000).fit(X_train_svm, temp_y_train)
    y_pred_xgb=model_xgb.predict(X_test_svm)

    
    model_bagging_xgb = BaggingRegressor(xgb.XGBRegressor(objective ='reg:squarederror',max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000),n_estimators=10, random_state=0).fit(X_train_svm, temp_y_train)  
    y_pred_bagging_xgb=model_bagging_xgb.predict(X_test_svm)

            
    model_gbr=GradientBoostingRegressor(n_estimators=1000).fit(X_train_svm, temp_y_train)
    y_pred_gbr=model_gbr.predict(X_test_svm)

        
    model_bagging_gbr= BaggingRegressor(GradientBoostingRegressor(n_estimators=1000),n_estimators=10, random_state=0).fit(X_train_svm, temp_y_train)
    y_pred_bagging_gbr=model_bagging_gbr.predict(X_test_svm)

    
    
    f=open('independent_final_results/cross_data/jackknife/nobagging/ind_test_'+str(epochs)+'_'+str(i)+'.txt','w')
    for n in range(len(temp_y_test)):
        f.write(str(temp_y_test[n])+'\t'+\
                str(y_pred_rf[n])+'\t'+str(y_pred_gbr[n])+'\t'+str(y_pred_xgb[n])+'\n')
    f.close()


    
    


    f=open('independent_final_results/cross_data/jackknife/bagging/ind_test_'+str(epochs)+'_'+str(i)+'.txt','w')
    for n in range(len(temp_y_test)):
        f.write(str(temp_y_test[n])+'\t'+\
                str(y_pred_bagging_rf[n])+'\t'+str(y_pred_bagging_gbr[n])+'\t'+str(y_pred_bagging_xgb[n])+'\n')
    f.close()

        


    ind_xgb_r2.append(np.corrcoef(temp_y_test, y_pred_xgb)[0,1])
    ind_rf_r2.append(np.corrcoef(temp_y_test, y_pred_rf)[0,1])
    ind_gbr_r2.append(np.corrcoef(temp_y_test, y_pred_gbr)[0,1])




    bagging_ind_xgb_r2.append(np.corrcoef(temp_y_test, y_pred_bagging_xgb)[0,1])
    bagging_ind_rf_r2.append(np.corrcoef(temp_y_test, y_pred_bagging_rf)[0,1])
    bagging_ind_gbr_r2.append(np.corrcoef(temp_y_test, y_pred_bagging_gbr)[0,1])



