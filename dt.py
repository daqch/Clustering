import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus

col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
formatted = pd.read_csv("processed.cleveland.data",
                        header=None, names=col_names)
print(formatted.head())

for i, row in formatted.iterrows():
    ifor_val = row[13]
    if ifor_val > 0:
        ifor_val = 1
    formatted.at[i, 'num'] = ifor_val

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = formatted[features]
y = formatted.num
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=1)

clf = DecisionTreeClassifier(max_leaf_nodes=12)

clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("accurracy: ", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=features, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('cleveland.png')
Image(graph.create_png())
