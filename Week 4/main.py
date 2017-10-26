import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# X_train, X_test
df_train = pd.read_csv('train.csv', encoding='ISO-8859-1')
pre_X_test = pd.read_csv('test.csv')

# dropping records found not responsible
df_train = df_train[np.isfinite(df_train['compliance'])]

# Columns containing unwanted data
cols_drop = ['payment_amount',
             'payment_date',
             'payment_status',
             'balance_due',
             'collection_status',
             'compliance',
             'compliance_detail']

# DROP 'EM
pre_X_train = df_train.drop(cols_drop, axis=1)

# target (train)
y_train = df_train['compliance']

# merge X_train and X_test for data normialization
X = pre_X_train.append(pre_X_test)
# for visualization purpose
X_back = X

# merge latlons into X
df_add = pd.read_csv('addresses.csv')
df_ll = pd.read_csv('latlons.csv')
latlons = df_add.merge(df_ll, how='outer', left_on='address', right_on='address').drop(['address'],axis=1)
# fill lat lon's NaNs
latlons['lat'] = latlons['lat'].fillna(44)
latlons['lon'] = latlons['lon'].fillna(-85)
X = X.merge(latlons, how='outer', left_on='ticket_id', right_on='ticket_id')

# turn textual data into numbers
def text_to_num(col):
    df = pd.DataFrame(data=np.array([x+1 for x in range(len(X[col].unique()))]), index=X[col].unique())
    df = df.merge(X[col].to_frame(), left_index=True, right_on=col)
    X[col] = df[0]

# do so for every column storing non-numeric data
for col in X.columns:
    if (X[col].dtype == 'object'):
        X[col] = X[col].str.upper()
        text_to_num(col)

# fill all unsightly NaNs        
X = X.fillna(0)

# select features
features = ['ticket_id', 'agency_name', 'inspector_name', 'city', 'state',
            'fine_amount', 'admin_fee', 'state_fee',
            'late_fee', 'discount_amount', 'clean_up_cost', 'disposition',
            'judgment_amount', 'lat', 'lon', 'violation_code']

# get selected features as X_select
X_select = X[features]

# Re-split X into X_train and X_test
X_train = X_select[X_select['ticket_id'].isin(pre_X_train['ticket_id'])].set_index('ticket_id')
X_test = X_select[X_select['ticket_id'].isin(pre_X_test['ticket_id'])].set_index('ticket_id')

# Scaler
scaler = MinMaxScaler()        
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


def blight_model():
    
    from sklearn.ensemble import RandomForestClassifier
    
    X_train_subset = X_train_scaled[:]
    y_train_subset = y_train[:]
    
    clf = RandomForestClassifier(max_depth=13).fit(X_train_subset, y_train_subset)
    
#     # ROC Curve, Training Set
#     from sklearn.metrics import roc_auc_score, roc_curve
#     y_pred = clf.predict_proba(X_train_subset)[:,1]
#     print(roc_auc_score(y_train_subset, y_pred))
#     fpr, tpr, _ = roc_curve(y_train_subset, y_pred)
    
#     import matplotlib.pyplot as plt
#     plt.plot(fpr, tpr)
#     plt.title('ROC Curve, Training Set')
#     plt.show()
    
    y_pred = clf.predict_proba(X_test_scaled)[:,1]
    result = pd.Series(data=y_pred, index=X_test.index)
    return result
    
# blight_model()