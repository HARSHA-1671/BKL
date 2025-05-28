import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
print("Dataset Preview:")
print(df.head())
df.hist(figsize=(15, 10), bins=30)
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=df[column])
    plt.title(f"Box plot of {column}")
plt.tight_layout()
plt.show()
print("\nOutlier Analysis (using IQR):")
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    print(f"{column}: {len(outliers)} outliers")
