#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 1) Load your cleaned DF
df = pd.read_pickle('/home/export/hbailey/data/embedding_resonance/who_leads_who_follows/cleaned_who_leads_df.pkl')

# 2) Convert date column
df['date'] = pd.to_datetime(df['date'])

# 3) Compute total tweets by group
total_tweets = df.groupby('user_type').size().sort_values(ascending=False)

# 4) Compute average daily tweets by group
daily = df.groupby(['date', 'user_type']).size().unstack(fill_value=0)
avg_daily = daily.mean().sort_values(ascending=False)

# 5) Map codes → pretty labels
label_map = {
    'random':      'General Public',
    'democrats':   'Democratic Supporters',
    'republicans': 'Republican Supporters',
    'public':      'Attentive Public',
    'media':       'Media Outlets',
    'dem_house':   'Dem. Legislators (House)',
    'rep_house':   'Rep. Legislators (House)',
    'dem_senate':  'Dem. Legislators (Senate)',
    'rep_senate':  'Rep. Legislators (Senate)'
}
pretty_index = [label_map.get(k, k) for k in total_tweets.index]

# 6) Prep formatter to disable scientific notation
def plain_formatter(x, pos):
    return f'{int(x):,}'

# 7) Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

# Top panel: total tweets (Oxford blue)
ax = axes[0]
total_tweets.plot(
    kind='bar', 
    ax=ax, 
    color='#002147'  # Oxford blue
)
ax.set_title('Total Tweets by Group', pad=12)
ax.set_ylabel('Total Tweets')
ax.set_xlabel('')                # remove any x-axis label
ax.set_xticklabels(pretty_index, rotation=45, ha='right')
ax.yaxis.set_major_formatter(FuncFormatter(plain_formatter))

# Bottom panel: avg daily tweets (CMU red)
ax = axes[1]
avg_daily.plot(
    kind='bar', 
    ax=ax, 
    color='#BE0000'  # CMU red
)
ax.set_title('Average Daily Tweets by Group', pad=12)
ax.set_ylabel('Avg Tweets per Day')
ax.set_xlabel('')                # remove any x-axis label
ax.set_xticklabels(pretty_index, rotation=45, ha='right')
ax.yaxis.set_major_formatter(FuncFormatter(plain_formatter))

# Save to file
outpath = '/home/export/hbailey/data/embedding_resonance/who_leads_who_follows/plots/twitter_data_summary.png'
plt.savefig(outpath, dpi=300)
