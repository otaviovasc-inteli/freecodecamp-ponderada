import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Import data
df = pd.read_csv('medical_examination.csv')

# 2 - Add 'overweight' column
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# 3 - Normalize 'cholesterol' and 'gluc' data
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5 - Create DataFrame for cat plot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=[
        'active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'
    ])

    # 6 - Group and reformat data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().rename(columns={'size': 'total'})

    # 7 - Draw the catplot
    fig = sns.catplot(
        x='variable', y='total', hue='value', col='cardio',
        data=df_cat, kind='bar'
    ).fig

    # 8 - Get the figure for output
    # Assigned 'fig' already in the previous step

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11 - Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 - Calculate the correlation matrix
    corr = df_heat.corr()

    # 13 - Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15 - Draw the heatmap
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.1f', center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .5}
    )

    # 16
    fig.savefig('heatmap.png')
    return fig
