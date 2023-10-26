import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from matplotlib.colors import LogNorm

# Read the data
df1 = pd.read_csv('/Users/tomkafer/Desktop/Q16.csv', delimiter=';')
df2 = pd.read_csv('/Users/tomkafer/Desktop/Q19.csv', delimiter=';')

# Filtering out error values
valid_range = set(range(1, 7)) # 1-5 and 6 for "no opinion"
columns_q1 = ['A', 'B', 'C']
columns_q2 = ['A', 'B', 'C']


for col in columns_q1:
        df1 = df1[df1[col].isin(valid_range)]
    
for col in columns_q2:
        df2 = df2[df2[col].isin(valid_range)]


# For Chi-square results
chi2_matrix = pd.DataFrame(index=columns_q1, columns=columns_q2)
for col1 in columns_q1:
    for col2 in columns_q2:
        contingency = pd.crosstab(df1[col1], df2[col2])
        chi2, p, _, _ = chi2_contingency(contingency, correction= True)
        chi2_matrix.at[col1, col2] = p
chi2_matrix = chi2_matrix.astype(float)

#

# Apply SymLogNorm normalization
norm = LogNorm(vmin=chi2_matrix.min().min(), vmax=chi2_matrix.max().max())

# Plot the chi2 matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(chi2_matrix, annot=True, cmap='gray', ax=ax, norm=norm)
ax.set_title('Chi-square Test Results Matrix')
ax.set_xlabel('Q19')
ax.set_ylabel('Q16', rotation=0, labelpad=40)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


plt.show()
