
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from access import repeat_action_zero_columns

training_data = pd.read_csv("./train.csv")
training_data = repeat_action_zero_columns(training_data)

Y_tr = training_data.ACTION.to_numpy()  # Y (ACTION) = access permission (0 or 1)
X_tr = training_data.drop("ACTION",axis=1)

features = ["RESOURCE","MGR_ID","ROLE_ROLLUP_1","ROLE_ROLLUP_2","ROLE_DEPTNAME","ROLE_TITLE","ROLE_FAMILY_DESC","ROLE_FAMILY","ROLE_CODE"]
#features = ["ROLE_DEPTNAME","ROLE_TITLE","ROLE_CODE"]
#
x = training_data.loc[:, features].values

y = training_data.loc[:,['ACTION']].values
x = StandardScaler().fit_transform(x)


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x[:100])
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, training_data[:100][['ACTION']]], axis = 1)
print(finalDf)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['ACTION'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()