import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from matplotlib.colors import LogNorm

# Read the data
df1 = pd.read_csv('/Q16.csv', delimiter=';')
df2 = pd.read_csv('/Q19.csv', delimiter=';')

# Filtering out error values
valid_range = set(range(1, 7)) # 1-5 and 6 for "no opinion"
columns_q1 = ['A', 'B', 'C']
columns_q2 = ['A', 'B', 'C']

for col in columns_q1:
        df1 = df1[df1[col].isin(valid_range)]
    
for col in columns_q2:
        df2 = df2[df2[col].isin(valid_range)]


# Initialize an empty DataFrame to store the mean and median values
summary_table = pd.DataFrame(index=['Mean', 'Median'])

# Calculate mean and median values for each column and add them to the summary_table
for col1, col2 in zip(columns_q1, columns_q2):
    summary_table[f'Q16{col1}'] = [df1[col1].mean(), df1[col1].median()]
    summary_table[f'Q19{col2}'] = [df2[col2].mean(), df2[col2].median()]

# Format the cell text to have three decimal places
formatted_values = summary_table.applymap('{:.3f}'.format).values

# Plot the summary table as a plain table
fig, ax = plt.subplots(figsize=(8, 6))  
ax.axis('off') 
tbl = ax.table(cellText=formatted_values,
               colLabels=summary_table.columns,
               rowLabels=summary_table.index,
               cellLoc='center',  
               loc='center')  
tbl.auto_set_font_size(False)
tbl.set_fontsize(16)  
tbl.scale(1.2, 1.2)  
plt.title('Summary Table of Mean and Median Values', fontsize=18, pad=20) 
plt.show()

# For Chi-square results
chi2_matrix = pd.DataFrame(index=columns_q1, columns=columns_q2)


# Chi-Square results and contingency table
for col1 in columns_q1:
    for col2 in columns_q2:
        contingency = pd.crosstab(df1[col1], df2[col2])

        # Plot the contingency tables
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(contingency, annot=True, fmt='d', cmap='gray', ax=ax)
        ax.set_title(f'Contingency Table: Q16 {col1} & Q19 {col2}')
        ax.set_xlabel('Q19')
        ax.set_ylabel('Q16', rotation=0, labelpad=40)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        print(contingency)
        plt.show()

        chi2, p, _, _ = chi2_contingency(contingency, correction=True)
        chi2_matrix.at[col1, col2] = p

chi2_matrix = chi2_matrix.astype(float)

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
