# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 21:30:15 2021

@author: lenovo
"""



import numpy as np
import pandas as pd
import numpy as np
import xlwt
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

# have_features_name=['pH','DO','Mg','Ca','Mn','Zn','Cd','Longitude']+features_name[51:52]+features_name[58:59]+\
#    features_name[75:76]+features_name[83:84]    
    


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
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_unimputed)

X=imp.transform(X_unimputed)

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
X=X[:,have_features_index]
new_features_name=[]
for each_index in have_features_index:
    new_features_name.append(features_name[each_index])
features_name=copy.deepcopy(new_features_name)
# f=open('new_data_have_data_without_bac_results_chose_random_states/features.txt','w',encoding="utf-8")
# for i in range(len(features_name)):
#     f.write(features_name[i]+'\n')
# f.close()

#
from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler()
scaler = ss.fit(X)
X= scaler.transform(X)

old_X=copy.deepcopy(X)
old_features_name=copy.deepcopy(features_name)
for i in range(old_X.shape[1]):
    for j in range(i+1,old_X.shape[1]):
        X=np.column_stack((X,old_X[:,i]*old_X[:,j]))
        features_name.append(old_features_name[i]+'+'+old_features_name[j])
        # X=np.column_stack((X,old_X[:,i]/(old_X[:,j]+0.000001)))
        
all_index=list(range(len(X)))
# all_index=list(set(all_index).difference(set([89,7,97])))
X=X[all_index]
all_y=all_y[all_index]
all_y=all_y[:,1]
chose_featrues_name=['pH+人均地区GDP','TDS(mg/L)+44Ca（ug/L）','TDS(mg/L)+人均地区GDP','DOC (mg/L)+65Cu（ug/L）','DOC (mg/L)+森林覆盖率%','24Mg（ug/L）+65Cu（ug/L）',\
'24Mg（ug/L）+人均地区GDP','24Mg（ug/L）+森林覆盖率%',\
'44Ca（ug/L）+PM2.5年平均年浓度','55Mn（ug/L）+地区GDP','60Ni（ug/L）+PM2.5年平均年浓度',\
'66Zn（ug/L）+地区GDP','66Zn（ug/L）+人均地区GDP','66Zn（ug/L）+森林覆盖率%',\
'NH4-N(mg/L)+Longitude','地区GDP+全年公路货物运输总量（万吨）',\
    '人均地区GDP+全市全年旅游总人数(万人次)','人口密度（人/km²）+全市全年旅游总人数(万人次)']



temp_index=[]
new_features_name=[]
for i in range(len(features_name)):
    if features_name[i] in chose_featrues_name:
        temp_index.append(i)
        new_features_name.append(features_name[i])
X=X[:,temp_index]


book = xlrd.open_workbook('new_data_from_database/new_all_transformed_data_all_years.xls')
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
    for j in range(2,cols):
        if is_number(sheet.cell(i,j).value):
            temp.append(float(sheet.cell(i,j).value))
            features_name.append(sheet.cell(0,j).value)
        else:
            print(sheet.cell(i,0).value)
    data.append(temp)


select_index=[]
for i in range(len(old_features_name)):
    for j in range(len(features_name)):
        if old_features_name[i]==features_name[j]:
            select_index.append(j)
            break
data=np.array(data)
data=data[:,select_index]
features_name=copy.deepcopy(old_features_name)



data= scaler.transform(data)
old_data=copy.deepcopy(data)
old_features_name=copy.deepcopy(features_name)
for i in range(old_data.shape[1]):
    for j in range(i+1,old_data.shape[1]):
        data=np.column_stack((data,old_data[:,i]*old_data[:,j]))
        features_name.append(old_features_name[i]+'+'+old_features_name[j])
 
chose_index=[]           
for each_name in new_features_name:
    flag=0
    for i in range(len(features_name)):
        if set(features_name[i].split('+'))==set(each_name.split('+')):
            chose_index.append(i)
            flag=1
            break
    if flag==0:
        print(each_name)
data=data[:,chose_index]

train_index=len(X)
all_data=np.vstack((X,data))


from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler()
X_train = ss.fit_transform(X)
X_test= ss.transform(data)
# from sklearn.metrics import accuracy_score

# import xgboost as xgb
# from sklearn import metrics

# # pca = PCA(n_components=100)
# # pca.fit(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if np.isnan(X[i,j]):
            print(str(i)+','+str(j))


import xgboost as xgb    
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
# X_train= X_svm[0:train_index]
# X_test=X_svm[train_index:]

y_pred_rf=BaggingRegressor(RandomForestRegressor(max_depth=4, random_state=0),n_estimators=10, random_state=0).fit(X_train, all_y).predict(X_test)     
model = xgb.XGBRegressor(objective ='reg:squarederror',max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000)
y_pred_xgb = BaggingRegressor(model,n_estimators=10, random_state=0).fit(X_train, all_y).predict(X_test)   
y_pred_gbr = BaggingRegressor(GradientBoostingRegressor(n_estimators=1000),n_estimators=10, random_state=0).fit(X_train, all_y).predict(X_test) 

workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('Sheet')
worksheet = workbook.get_sheet('Sheet')



for i in range(len(y_pred_gbr)):
    
    worksheet.write(i,0,float(y_pred_gbr[i]))
    worksheet.write(i,1,float(y_pred_rf[i]))
    worksheet.write(i,2,float(y_pred_xgb[i]))
    



           
workbook.save('map_data_results/all_results_stage_2_all_years.xls')  

