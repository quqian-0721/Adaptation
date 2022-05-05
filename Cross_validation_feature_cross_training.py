import numpy as np
import pandas as pd
import numpy as np

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


# have_features_name=['pH','DO','TDS','DOC','Na','Mg','39K（ug/L）','Ca','Mn','Ni','Cu',\
#                     'Zn','Cd','NH4-N','Longitude','Latitude']+features_name[50:52]+features_name[58:59]+\
#     features_name[61:62]+features_name[75:76]+features_name[83:85]      

have_features_name=['pH','DO','TDS','DOC','Na','Mg','39K（ug/L）','Ca','Mn','Ni','Cu',\
                    'Zn','Cd','NH4-N','Longitude','Latitude']+features_name[51:52]+features_name[58:59]+\
    features_name[61:62]+features_name[75:76]

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


from sklearn.preprocessing import MinMaxScaler
# ss = MinMaxScaler()
# scaler = ss.fit(X)
# X= scaler.transform(X)

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
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import RepeatedKFold
random_states = 0
kf = RepeatedKFold(n_splits=10,n_repeats=1,random_state=random_states)

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
scaler = ss.fit(X)
X_svm= scaler.transform(X)
new_y=np.sum(all_y,1)
all_y=np.column_stack((all_y,new_y))
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
for n_features in range(2,31,1):

    print(n_features)
    for i in range(all_y.shape[1]):
        print(i)
        y=all_y[:,i]
        estimator = RandomForestRegressor(max_depth=4, random_state=0)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector = selector.fit(X_svm, y)
        temp_X_svm=selector.transform(X_svm)

        svm_r2=[]
        xgb_r2=[]
        rf_r2=[]
        gbr_r2=[]
        mlp_r2=[]
        all_y_true=[]
        all_y_train=[]
        all_y_pred_xgb=[]
        all_y_pred_rf=[]
        all_y_pred_gbr=[]
        all_y_pred_mlp=[]
        all_y_pred_svm=[]
        
        all_y_pred_train_xgb=[]
        all_y_pred_train_rf=[]
        all_y_pred_train_gbr=[]
        all_y_pred_train_mlp=[]
        all_y_pred_train_svm=[]
        
        all_y_pred_bagging_xgb=[]
        all_y_pred_bagging_rf=[]
        all_y_pred_bagging_gbr=[]
        all_y_pred_bagging_mlp=[]
        all_y_pred_bagging_svm=[]
        
        all_y_pred_train_bagging_xgb=[]
        all_y_pred_train_bagging_rf=[]
        all_y_pred_train_bagging_gbr=[]
        all_y_pred_train_bagging_mlp=[]
        all_y_pred_train_bagging_svm=[]
        for kfold in range(10):
            f=open('train_test_index/train_'+str(kfold)+'.txt','r')
            text=f.read()
            f.close()
            text=text.split('\n')
            while '' in text:
                text.remove('')
            train=[]
            for each_line in text:
                train.append(int(each_line))
            f=open('train_test_index/test_'+str(kfold)+'.txt','r')
            text=f.read()
            f.close()
            text=text.split('\n')
            while '' in text:
                text.remove('')
            test=[]
            for each_line in text:
                test.append(int(each_line))   
                
            X_train, X_test, y_train, y_test = temp_X_svm[train], temp_X_svm[test], y[train], y[test]
            model_svm=svm.SVR().fit(X_train, y_train)
            y_pred_svm=model_svm.predict(X_test)
            y_pred_train_svm=model_svm.predict(X_train)
            model_bagging_svm=BaggingRegressor(svm.SVR(),n_estimators=10, random_state=0).fit(X_train, y_train)
            y_pred_bagging_svm=model_bagging_svm.predict(X_test)
            y_pred_train_bagging_svm=model_bagging_svm.predict(X_train)
            
            model_rf=RandomForestRegressor(max_depth=4, random_state=0).fit(X_train, y_train)
            y_pred_rf=model_rf.predict(X_test)
            y_pred_train_rf=model_rf.predict(X_train)
            model_bagging_rf=BaggingRegressor(RandomForestRegressor(max_depth=4, random_state=0),n_estimators=10, random_state=0).fit(X_train, y_train)     
            y_pred_bagging_rf=model_bagging_rf.predict(X_test)
            y_pred_train_bagging_rf=model_bagging_rf.predict(X_train)
            
            model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000).fit(X_train, y_train)
            y_pred_xgb=model_xgb.predict(X_test)
            y_pred_train_xgb=model_xgb.predict(X_train)
            model_bagging_xgb = BaggingRegressor(xgb.XGBRegressor(objective ='reg:squarederror',max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000),n_estimators=10, random_state=0).fit(X_train, y_train)  
            y_pred_bagging_xgb=model_bagging_xgb.predict(X_test)
            y_pred_train_bagging_xgb=model_bagging_xgb.predict(X_train)
            
            model_gbr=GradientBoostingRegressor(n_estimators=1000).fit(X_train, y_train)
            y_pred_gbr=model_gbr.predict(X_test)
            y_pred_train_gbr=model_gbr.predict(X_train)            
            model_bagging_gbr= BaggingRegressor(GradientBoostingRegressor(n_estimators=1000),n_estimators=10, random_state=0).fit(X_train, y_train)
            y_pred_bagging_gbr=model_bagging_gbr.predict(X_test)
            y_pred_train_bagging_gbr=model_bagging_gbr.predict(X_train)
            
            model_mlp=MLPRegressor(hidden_layer_sizes=(10, 5),tol=1e-2, max_iter=500, random_state=0).fit(X_train, y_train)
            y_pred_mlp=model_mlp.predict(X_test)
            y_pred_train_mlp=model_mlp.predict(X_train)               
            model_bagging_mlp =BaggingRegressor(MLPRegressor(hidden_layer_sizes=(10, 5),tol=1e-2, max_iter=500, random_state=0),n_estimators=10, random_state=0).fit(X_train, y_train)
            y_pred_bagging_mlp=model_bagging_mlp.predict(X_test)
            y_pred_train_bagging_mlp=model_bagging_mlp.predict(X_train)
            
            xgb_r2.append(np.corrcoef(y_test, y_pred_xgb)[0,1])
            rf_r2.append(np.corrcoef(y_test, y_pred_rf)[0,1])
            gbr_r2.append(np.corrcoef(y_test, y_pred_gbr)[0,1])
            mlp_r2.append(np.corrcoef(y_test, y_pred_mlp)[0,1])
            svm_r2.append(np.corrcoef(y_test, y_pred_svm)[0,1])
            all_y_pred_gbr+=y_pred_gbr.tolist()
            all_y_pred_xgb+=y_pred_xgb.tolist()
            all_y_pred_rf+=y_pred_rf.tolist()
            all_y_pred_mlp+=y_pred_mlp.tolist()
            all_y_pred_svm+=y_pred_svm.tolist()
            all_y_true+=y_test.tolist()
            
            all_y_pred_train_gbr+=y_pred_train_gbr.tolist()
            all_y_pred_train_xgb+=y_pred_train_xgb.tolist()
            all_y_pred_train_rf+=y_pred_train_rf.tolist()
            all_y_pred_train_mlp+=y_pred_train_mlp.tolist()
            all_y_pred_train_svm+=y_pred_train_svm.tolist()
            all_y_train+=y_train.tolist()

            all_y_pred_bagging_gbr+=y_pred_bagging_gbr.tolist()
            all_y_pred_bagging_xgb+=y_pred_bagging_xgb.tolist()
            all_y_pred_bagging_rf+=y_pred_bagging_rf.tolist()
            all_y_pred_bagging_mlp+=y_pred_bagging_mlp.tolist()
            all_y_pred_bagging_svm+=y_pred_bagging_svm.tolist()


            all_y_pred_train_bagging_gbr+=y_pred_train_bagging_gbr.tolist()
            all_y_pred_train_bagging_xgb+=y_pred_train_bagging_xgb.tolist()
            all_y_pred_train_bagging_rf+=y_pred_train_bagging_rf.tolist()
            all_y_pred_train_bagging_mlp+=y_pred_train_bagging_mlp.tolist()
            all_y_pred_train_bagging_svm+=y_pred_train_bagging_svm.tolist()
 
            
        final_rf_r2=np.corrcoef(all_y_true, all_y_pred_rf)[0,1]
        final_xgb_r2=np.corrcoef(all_y_true, all_y_pred_xgb)[0,1]
        final_gbr_r2=np.corrcoef(all_y_true, all_y_pred_gbr)[0,1]
        final_mlp_r2=np.corrcoef(all_y_true, all_y_pred_mlp)[0,1]
        final_svm_r2=np.corrcoef(all_y_true, all_y_pred_svm)[0,1]
        
        final_train_rf_r2=np.corrcoef(all_y_train, all_y_pred_train_rf)[0,1]
        final_train_xgb_r2=np.corrcoef(all_y_train, all_y_pred_train_xgb)[0,1]
        final_train_gbr_r2=np.corrcoef(all_y_train, all_y_pred_train_gbr)[0,1]
        final_train_mlp_r2=np.corrcoef(all_y_train, all_y_pred_train_mlp)[0,1]
        final_train_svm_r2=np.corrcoef(all_y_train, all_y_pred_train_svm)[0,1]
        
        final_bagging_rf_r2=np.corrcoef(all_y_true, all_y_pred_bagging_rf)[0,1]
        final_bagging_xgb_r2=np.corrcoef(all_y_true, all_y_pred_bagging_xgb)[0,1]
        final_bagging_gbr_r2=np.corrcoef(all_y_true, all_y_pred_bagging_gbr)[0,1]
        final_bagging_mlp_r2=np.corrcoef(all_y_true, all_y_pred_bagging_mlp)[0,1]
        final_bagging_svm_r2=np.corrcoef(all_y_true, all_y_pred_bagging_svm)[0,1]
        
        final_train_bagging_rf_r2=np.corrcoef(all_y_train, all_y_pred_train_bagging_rf)[0,1]
        final_train_bagging_xgb_r2=np.corrcoef(all_y_train, all_y_pred_train_bagging_xgb)[0,1]
        final_train_bagging_gbr_r2=np.corrcoef(all_y_train, all_y_pred_train_bagging_gbr)[0,1]
        final_train_bagging_mlp_r2=np.corrcoef(all_y_train, all_y_pred_train_bagging_mlp)[0,1]
        final_train_bagging_svm_r2=np.corrcoef(all_y_train, all_y_pred_train_bagging_svm)[0,1]
        
        print('{:.4f}'.format(np.mean(final_train_rf_r2))+',{:.4f}'.format(np.mean(final_train_gbr_r2))+',{:.4f}'.format(np.mean(final_train_xgb_r2))+',{:.4f}'.format(np.mean(final_train_mlp_r2))+',{:.4f}'.format(np.mean(final_train_svm_r2)))
        print('{:.4f}'.format(np.mean(final_rf_r2))+',{:.4f}'.format(np.mean(final_gbr_r2))+',{:.4f}'.format(np.mean(final_xgb_r2))+',{:.4f}'.format(np.mean(final_mlp_r2))+',{:.4f}'.format(np.mean(final_svm_r2)))
        print('{:.4f}'.format(np.mean(final_train_bagging_rf_r2))+',{:.4f}'.format(np.mean(final_train_bagging_gbr_r2))+',{:.4f}'.format(np.mean(final_train_bagging_xgb_r2))+',{:.4f}'.format(np.mean(final_train_bagging_mlp_r2))+',{:.4f}'.format(np.mean(final_train_bagging_svm_r2)))
        print('{:.4f}'.format(np.mean(final_bagging_rf_r2))+',{:.4f}'.format(np.mean(final_bagging_gbr_r2))+',{:.4f}'.format(np.mean(final_bagging_xgb_r2))+',{:.4f}'.format(np.mean(final_bagging_mlp_r2))+',{:.4f}'.format(np.mean(final_bagging_svm_r2)))
        all_y_pred_rf=np.array(all_y_pred_rf)
        all_y_pred_xgb=np.array(all_y_pred_xgb)
        # for weight in range(1,10):
        #     all_y_pred_combined=weight*0.1*all_y_pred_rf+(1-weight*0.1)*all_y_pred_xgb
        #     print('{:.4f}'.format(np.mean(np.corrcoef(all_y_true, all_y_pred_combined)[0,1])))
        f=open('final_results/cross_data/FS/feature_name/'+str(n_features)+'_'+\
                str(i)+'_minmax_scaler.txt','w',encoding="utf-8")
        for k in range(len(selector.support_)):
            if selector.support_[k]:

                f.write(features_name[k]+'\n')
        f.close()
        f=open('final_results/cross_data/FS/feature_data/'+str(n_features)+'_'+\
                str(i)+'_minmax_scaler.txt','w')
        for k in range(len(temp_X_svm)):
            for j in range(len(temp_X_svm[k])):
                f.write(str(temp_X_svm[k][j])+'\t')
            f.write(str(y[k])+'\n')

        f.close()
        
        f=open('final_results/cross_data/FS/nobagging/test_'+str(n_features)+'_'+str(i)+'.txt','w')
        for n in range(len(all_y_true)):
            f.write(str(all_y_true[n])+'\t'+str(all_y_pred_svm[n])+'\t'+str(all_y_pred_rf[n])+'\t'+str(all_y_pred_gbr[n])+'\t'+str(all_y_pred_xgb[n])+'\t'+str(all_y_pred_mlp[n])+'\n')
        f.close()

        f=open('final_results/cross_data/FS/nobagging/train_'+str(n_features)+'_'+str(i)+'.txt','w')
        for n in range(len(all_y_train)):
            f.write(str(all_y_train[n])+'\t'+str(all_y_pred_train_svm[n])+'\t'+str(all_y_pred_train_rf[n])+'\t'+str(all_y_pred_train_gbr[n])+'\t'+str(all_y_pred_train_xgb[n])+'\t'+str(all_y_pred_train_mlp[n])+'\n')
        f.close()

        f=open('final_results/cross_data/FS/bagging/test_'+str(n_features)+'_'+str(i)+'.txt','w')
        for n in range(len(all_y_true)):
            f.write(str(all_y_true[n])+'\t'+str(all_y_pred_bagging_svm[n])+'\t'+str(all_y_pred_bagging_rf[n])+'\t'+str(all_y_pred_bagging_gbr[n])+'\t'+str(all_y_pred_bagging_xgb[n])+'\t'+str(all_y_pred_bagging_mlp[n])+'\n')
        f.close()

        f=open('final_results/cross_data/FS/bagging/train_'+str(n_features)+'_'+str(i)+'.txt','w')
        for n in range(len(all_y_train)):
            f.write(str(all_y_train[n])+'\t'+str(all_y_pred_train_bagging_svm[n])+'\t'+str(all_y_pred_train_bagging_rf[n])+'\t'+str(all_y_pred_train_bagging_gbr[n])+'\t'+str(all_y_pred_train_bagging_xgb[n])+'\t'+str(all_y_pred_train_bagging_mlp[n])+'\n')
        f.close()
