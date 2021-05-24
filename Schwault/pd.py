
# The Main Function which is called in app.py
def prob(str1,credamt,age):
  import pandas as pd
  import numpy as np
  str2 = "./static/"
  global str3
  str3 = str2 + str1
  default_var = pd.read_csv(str3)
  default_var
  default_var = default_var.rename(columns={'LIMIT_BAL':'CRED_AMT', 'MARRIAGE': 'MARITAL STATUS',
                                          'PAY_0': 'PAYSTAT_0', 'PAY_1': 'PAYSTAT_1', 'PAY_2': 'PAYSTAT_2',
                                          'PAY_3': 'PAYSTAT_3', 'PAY_4': 'PAYSTAT_4', 'PAY_5': 'PAYSTAT_5',
                                            'PAY_6': 'PAYSTAT_6', 'default.payment.next.month':'default'})
  default_var.rename(columns=lambda x: x.lower(), inplace=True)
  default_var = default_var[['pay_amt1','age','default']]
  default_var['pay_amt1'] = (default_var['pay_amt1']/1000)
  X = default_var[['pay_amt1','age']]
  y = default_var['default']
  from sklearn.linear_model import LogisticRegression
  reg = LogisticRegression()
  reg.fit(X,y)
  print(reg.intercept_)
  print(reg.coef_)
    # creating  the formula for Z
    # According to the Least Squares Regression Analysis
    # Pr(default = 1/X) = 1 / 1 + exp(-Z)
    # Z = w0 + w1.LimitBalance + w2.Age
  dim()
  payamt1 = default_var['pay_amt1']
  agep = default_var['age']
  pdval = prob_default(cred_amt=credamt, age=age)
  return pdval



#Function to show dimensionality reduction
def dim():  
  import pandas as pd 
  import numpy as np
  from sklearn import linear_model
  from matplotlib import pyplot as plt
  from sklearn.decomposition import PCA,TruncatedSVD
  default_var = pd.read_csv(str3)
  X = X = default_var.iloc[:, 0:-1].values
  y = default_var.iloc[:, -1].values
  from sklearn.linear_model import LogisticRegression
  reg = LogisticRegression()
  from sklearn import preprocessing
  scaler = preprocessing.StandardScaler().fit(X)
  X_scaled = scaler.transform(X)
  print(reg.fit(X_scaled,y))
  print(reg.score(X_scaled,y))
  pca = PCA(n_components=15, whiten='True')
  x = pca.fit(X).transform(X)
  reg.fit(x,y)
  reg.score(x,y)
  print(pca.explained_variance_)



#Function calculates Z
def Z(pay_amt1,age):
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  import numpy as np
  str1 = "./static\inputfile"
  default_var = pd.read_csv(str3)
  X = default_var[['PAY_AMT1','AGE']]
  y = default_var['default.payment.next.month']
  from sklearn.linear_model import LogisticRegression
  reg = LogisticRegression()
  reg.fit(X,y)

  return reg.intercept_[0] + reg.coef_[0][0]*pay_amt1 + reg.coef_[0][1]*age

#Function computes the default
def prob_default(cred_amt, age):
  import numpy as np
  z = Z(cred_amt, age)
  return 1/(1 + np.exp(-z))

 
#Function which plots the boxplot 
def box():
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  import numpy as np
  data = pd.read_csv(str3)
  data = data.rename(columns={'LIMIT_BAL':'CRED_AMT', 'MARRIAGE': 'MARITAL STATUS',
                                          'PAY_0': 'PAYSTAT_0', 'PAY_1': 'PAYSTAT_1', 'PAY_2': 'PAYSTAT_2',
                                          'PAY_3': 'PAYSTAT_3', 'PAY_4': 'PAYSTAT_4', 'PAY_5': 'PAYSTAT_5',
                                            'PAY_6': 'PAYSTAT_6', 'default.payment.next.month':'default'})
  sns.boxplot(x='default', y='CRED_AMT', data=data)
  plt.savefig("./static/BoxPlot.jpg")

#Function which plots the conditional plots
def cond():
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  import numpy as np
  data = pd.read_csv(str3)
  data = data.rename(columns={'LIMIT_BAL':'CRED_AMT', 'MARRIAGE': 'MARITAL STATUS',
                                          'PAY_0': 'PAYSTAT_0', 'PAY_1': 'PAYSTAT_1', 'PAY_2': 'PAYSTAT_2',
                                          'PAY_3': 'PAYSTAT_3', 'PAY_4': 'PAYSTAT_4', 'PAY_5': 'PAYSTAT_5',
                                            'PAY_6': 'PAYSTAT_6', 'default.payment.next.month':'default'})
  conditional_plot = sns.FacetGrid(data, col="default",  row="SEX", aspect=2)
  conditional_plot.map(sns.regplot, 'BILL_AMT1', 'BILL_AMT2')
  conditional_plot.savefig("./static/ConditionalPlot.jpg")
  

#Function which plots the chart
def plot():
 import pandas as pd
 import seaborn as sns
 import matplotlib.pyplot as plt
 import numpy as np
 data = pd.read_csv(str3)
 data = data.rename(columns={'LIMIT_BAL':'CRED_AMT', 'MARRIAGE': 'MARITAL STATUS',
                                         'PAY_0': 'PAYSTAT_0', 'PAY_1': 'PAYSTAT_1', 'PAY_2': 'PAYSTAT_2',
                                         'PAY_3': 'PAYSTAT_3', 'PAY_4': 'PAYSTAT_4', 'PAY_5': 'PAYSTAT_5',
                                          'PAY_6': 'PAYSTAT_6', 'default.payment.next.month':'default'})
 data.groupby(['EDUCATION'])['default'].value_counts(normalize=True)\
 .unstack().plot(kind='bar', stacked=True)
 plt.savefig("./static/Plot.jpg")


