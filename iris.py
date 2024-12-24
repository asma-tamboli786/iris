import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# Load the iris dataset

iris = load_iris()

# Create a DataFrame from the iris dataset

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['target'] = iris.target

# Display the first few rows of the DataFrame

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

print(df[df['target'] == 0].head())

print(df[df['target'] == 1].head())

print(df[df['target'] == 2].head())

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

# plt.scatter(x=df0['sepal length (cm)'],y=df0['sepal width (cm)'],marker='+')
# plt.scatter(x=df1['sepal length (cm)'],y=df1['sepal width (cm)'],marker='.')
# plt.show()

