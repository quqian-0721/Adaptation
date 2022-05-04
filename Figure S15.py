# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:11:00 2022

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



have_features_name=['pH','DO','TDS','DOC','Na','Mg','39K（ug/L）','Ca','Mn','Ni','Cu',\
                    'Zn','Cd','NH4-N','Longitude','Latitude']+features_name[50:52]+features_name[58:59]+\
    features_name[60:62]+features_name[75:76]+features_name[83:85]      

data_dict={}
for i in range(len(files_name)):
    data_dict[files_name[i]]=data[i,:]


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
features_name=copy.deepcopy(new_features_name)




from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_unimputed)

X_imputed=imp.transform(X_unimputed)

from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler()
scaler = ss.fit(X_imputed)
X_imputed= scaler.transform(X_imputed)



old_X_imputed=copy.deepcopy(X_imputed)
old_features_name=copy.deepcopy(features_name)
for i in range(old_X_imputed.shape[1]):
    for j in range(i+1,old_X_imputed.shape[1]):
        X_imputed=np.column_stack((X_imputed,old_X_imputed[:,i]*old_X_imputed[:,j]))
        features_name.append(old_features_name[i]+'+'+old_features_name[j])



# chose_featrues_name=['DOC (mg/L)','pH+44Ca（ug/L）','pH+60Ni（ug/L）',\
# '44Ca（ug/L）+65Cu（ug/L）',\
# '44Ca（ug/L）+PM2.5年平均年浓度','66Zn（ug/L）+森林覆盖率%',\
# 'Longitude+人口密度（人/km²）','地区GDP+全年公路货物运输总量（万吨）',\
#     '人口密度（人/km²）+PM2.5年平均年浓度']

# chose_featrues_name=['pH+人均地区GDP','TDS(mg/L)+44Ca（ug/L）','TDS(mg/L)+人均地区GDP','DOC (mg/L)+65Cu（ug/L）','DOC (mg/L)+森林覆盖率%','24Mg（ug/L）+65Cu（ug/L）',\
# '24Mg（ug/L）+人均地区GDP','24Mg（ug/L）+森林覆盖率%',\
# '44Ca（ug/L）+PM2.5年平均年浓度','55Mn（ug/L）+地区GDP','60Ni（ug/L）+PM2.5年平均年浓度',\
# '66Zn（ug/L）+地区GDP','66Zn（ug/L）+人均地区GDP','66Zn（ug/L）+森林覆盖率%',\
# 'NH4-N(mg/L)+Longitude','地区GDP+全年公路货物运输总量（万吨）',\
#     '人均地区GDP+全市全年旅游总人数(万人次)','人口密度（人/km²）+全市全年旅游总人数(万人次)']

chose_featrues_name=['pH+DO(mg/L)','pH+Longitude','DO(mg/L)+66Zn（ug/L）','DO(mg/L)+111Cd（ug/L）','TDS(mg/L)+人均地区GDP',\
'DOC (mg/L)+森林覆盖率%','23Na（ug/L）+地区GDP','24Mg（ug/L）+PM2.5年平均年浓度','44Ca（ug/L）+人均地区GDP',\
'55Mn（ug/L）+NH4-N(mg/L)','55Mn（ug/L）+人均地区GDP','66Zn（ug/L）+地区GDP','66Zn（ug/L）+人均地区GDP','111Cd（ug/L）+人均地区GDP','111Cd（ug/L）+全市全年旅游总人数(万人次)',\
'Longitude+森林覆盖率%','Longitude+PM2.5年平均年浓度']
    
    
temp_index=[]
new_features_name=[]
for i in range(len(features_name)):
    if features_name[i] in chose_featrues_name:
        temp_index.append(i)
        new_features_name.append(features_name[i])
X=X_imputed[:,temp_index]


from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import RepeatedKFold
random_states = 0
kf = KFold(n_splits=10, random_state=random_states,shuffle=True)

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
import random
ss = MinMaxScaler()
scaler = ss.fit(X)
X_svm= scaler.transform(X)

new_y=np.sum(all_y,1)
all_y=np.column_stack((all_y,new_y))

i=2

ave_rf_r2=[]
ave_xgb_r2=[]
ave_gbr_r2=[]

ave_bagging_rf_r2=[]
ave_bagging_xgb_r2=[]
ave_bagging_gbr_r2=[]

for shuffle_times in range(500):
    nfold=0
    svm_r2=[]
    xgb_r2=[]
    rf_r2=[]
    gbr_r2=[]
    mlp_r2=[]
    
    bagging_svm_r2=[]
    bagging_xgb_r2=[]
    bagging_rf_r2=[]
    bagging_gbr_r2=[]
    bagging_mlp_r2=[]
    
    for train, test in kf.split(X_svm):
        old_temp_y=all_y[:,i]
        temp_y=np.random.permutation(old_temp_y)
        
        X_train_fold, X_test_fold, y_train_fold, y_test_fold \
            = X_svm[train], X_svm[test], temp_y[train], temp_y[test]
        
        model_rf=RandomForestRegressor(max_depth=4, random_state=0).fit(X_train_fold, y_train_fold)
        y_pred_rf=model_rf.predict(X_test_fold)

    
        model_bagging_rf=BaggingRegressor(RandomForestRegressor(max_depth=4, random_state=0),n_estimators=10, random_state=0).fit(X_train_fold, y_train_fold)     
        y_pred_bagging_rf=model_bagging_rf.predict(X_test_fold)

            
        model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000).fit(X_train_fold, y_train_fold)
        y_pred_xgb=model_xgb.predict(X_test_fold)

        model_bagging_xgb = BaggingRegressor(xgb.XGBRegressor(objective ='reg:squarederror',max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000),n_estimators=10, random_state=0).fit(X_train_fold, y_train_fold)  
        y_pred_bagging_xgb=model_bagging_xgb.predict(X_test_fold)

            
        model_gbr=GradientBoostingRegressor(n_estimators=1000).fit(X_train_fold, y_train_fold)
        y_pred_gbr=model_gbr.predict(X_test_fold)

        
        model_bagging_gbr= BaggingRegressor(GradientBoostingRegressor(n_estimators=1000),n_estimators=10, random_state=0).fit(X_train_fold, y_train_fold)
        y_pred_bagging_gbr=model_bagging_gbr.predict(X_test_fold)



    
    
        f=open('permutationtest/cross_data/nobagging/test_'+str(shuffle_times)+'_'+str(nfold)+'_'+str(i)+'.txt','w')
        for n in range(len(y_test_fold)):
            f.write(str(y_test_fold[n])+'\t'+\
                str(y_pred_rf[n])+'\t'+str(y_pred_gbr[n])+'\t'+str(y_pred_xgb[n])+'\n')
        f.close()
    

    
    
        f=open('permutationtest/cross_data/bagging/test_'+str(shuffle_times)+'_'+str(nfold)+'_'+str(i)+'.txt','w')
        for n in range(len(y_test_fold)):
            f.write(str(y_test_fold[n])+'\t'+str(y_pred_bagging_rf[n])+'\t'+str(y_pred_bagging_gbr[n])+'\t'+str(y_pred_bagging_xgb[n])+'\n')
        f.close()

    
        nfold+=1
        
            
        xgb_r2.append(np.corrcoef(y_test_fold, y_pred_xgb)[0,1])
        rf_r2.append(np.corrcoef(y_test_fold, y_pred_rf)[0,1])
        gbr_r2.append(np.corrcoef(y_test_fold, y_pred_gbr)[0,1])




        bagging_xgb_r2.append(np.corrcoef(y_test_fold, y_pred_bagging_xgb)[0,1])
        bagging_rf_r2.append(np.corrcoef(y_test_fold, y_pred_bagging_rf)[0,1])
        bagging_gbr_r2.append(np.corrcoef(y_test_fold, y_pred_bagging_gbr)[0,1])
    
    ave_xgb_r2.append(np.average(xgb_r2))
    ave_rf_r2.append(np.average(rf_r2))
    ave_gbr_r2.append(np.average(gbr_r2))
    ave_bagging_xgb_r2.append(np.average(bagging_xgb_r2))
    ave_bagging_rf_r2.append(np.average(bagging_rf_r2))
    ave_bagging_gbr_r2.append(np.average(bagging_gbr_r2))

    




        
