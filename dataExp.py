# !pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

link = 'https://drive.google.com/open?id=1wvKi-E6GJnRqAkYMpVPaRc0L0XgY0Wkq'
fluff, id = link.split('=')
file = drive.CreateFile({'id':id}) # replace the id with id of file you want to access
file.GetContentFile('churn.csv')  

# test comment

# 1.1 raw dataset
import warnings
warnings.filterwarnings('ignore') #????

import pandas as pd
import numpy as np
churn_df = pd.read_csv('churn.csv.all')
churn_df.info()
churn_df.head()
churn_df['state'].unique()

print('Rows: '+ str(churn_df.shape[0]))
print('col: '+ str(churn_df.shape[1]))

# 1.2 data cleaning
'''check categorical feature before cleaning
remove head and tril whitespace
check after cleaning'''
churn_df['state'][0]
churn_df['voice_mail_plan'] = churn_df['voice_mail_plan'].apply(lambda x: x.strip())
churn_df['voice_mail_plan']

'''check distribution, correlation among all features, show heatmap of correlations'''
%matplotlib inline 
# for jupyter, or use the ipython: In command pallette, choose 'Show Python Interactive Window'
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(churn_df['total_intl_charge'])
plt.show()
# need to type in all the headers?
corr = churn_df[["account_length",
"number_vmail_messages",
'total_day_minutes',
'total_day_calls','total_day_charge',
'total_eve_minutes',
'total_eve_calls','total_eve_charge',
'total_night_minutes','total_night_calls',
'total_night_charge','total_intl_minutes',
'total_intl_calls','total_intl_charge'
]].corr()
# mind corr() is linear, if two col has high corr, must regularize, can just reduce feature
# not suggest PCA, it's for dimension reduction
# how was string calculated in corr? ASCII not make sense -- sklearn not accept categorical
sns.heatmap(corr)
plt.show()
corr
# calculate two features correlation
from scipy.stats import pearsonr
print(pearsonr(churn_df['total_day_minutes'],churn_df['number_vmail_messages'])[0])

# 2. Feature preprocessing

# get ground truth data
y = np.where(churn_df['churned'] == 'True.', 1, 0)
# drop useless col 
# ???????? how to tell useless? corr? 

to_drop = ['area_code','phone_number','churned']
churn_feat_space = churn_df.drop(to_drop, axis=1)
# yes_and_no to covert to boolean val
# boolean is same as binary
yes_no_cols = ["intl_plan", "voice_mail_plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes' 
# sklearn.preprocessing.OneHotEncoder
# label encoding is wrong, one hot encoding is better 
# -- target encoding can handle the issue that one hot result in too many dimension, which lead to sparse
# frequency encoding? 
# for categorical feature, suggest using Tree, if insist on regression, read papers here:
# https://github.com/scikit-learn-contrib/categorical-encoding
churn_feat_space = pd.get_dummies(churn_feat_space, columns=['state'])
churn_feat_space.head()
X = churn_feat_space
# check the propotion of y = 1, imbalance data? -- use up/down sampling
print(y.sum() / y.shape * 100)


