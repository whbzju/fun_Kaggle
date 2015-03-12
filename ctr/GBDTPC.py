import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

def trainModel(df, key):
    clf = GradientBoostingClassifier(n_estimators=100)
    if 'pc' in key:
        print "pc"
        clf.fit(df[['C1_cat', 'banner_pos_cat', 'site_id_cat',
       'site_domain_cat', 'site_category_cat', 'device_model_cat',
       'device_type_cat', 'device_conn_type_cat', 'C14_cat', 'C15_cat',
       'C16_cat', 'C17_cat', 'C18_cat', 'C19_cat', 'C20_cat', 'C21_cat']].values, df['click'].values)
    elif 'app' in key:
        print "app"
        clf.fit(df[['C1_cat', 'banner_pos_cat', 'app_id_cat',
       'app_domain_cat', 'app_category_cat', 'device_model_cat',
       'device_type_cat', 'device_conn_type_cat', 'C14_cat', 'C15_cat',
       'C16_cat', 'C17_cat', 'C18_cat', 'C19_cat', 'C20_cat', 'C21_cat']].values, df['click'].values)
    else:
        print "error data"

    return clf

def predict(clf, df):
    y = clf.predict_proba(df.values)
    return y

def save(name, df, y):
    df['click'] = y[:, 1]
    df.to_csv(name, columns=('id', 'click'), index=False)

df_train_pc = pd.read_csv('train_pc_clean.csv')

clf = trainModel(df_train_pc, 'pc')

from sklearn.externals import joblib
joblib.dump(clf, 'model/gbdt100pc.pkl') 

df_test_pc = pd.read_csv('test_pc_clean.csv')
df_test_pc_id = pd.read_csv('test_pc_id.csv')

y = predict(clf, df_test_pc)
save('gbdt_test_pc_prediction', df_test_pc_id, y)

