

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

def p_to_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'n.s.'
    
def add_sig(ax, x1, x2, y, p, h=0.02):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='black')
    ax.text((x1+x2)/2, y+h, p_to_star(p), ha='center', va='bottom')
    
def plot_test_bars(data_sources, spacing=0, label=None, ax=None, color='grey',
                   input_types = ['Visual only', 'Vector only', 'Both']) : 
    means = []
    stds   = []
    
    for d in data_sources : 
        rates = success_rate_per_run(d)
        
        mean = np.mean(rates)
        sem = np.std(rates, ddof=1) / np.sqrt(len(rates))
        
        means.append(mean)
        stds.append(sem)
    
    # Create lists for the plot

    x_pos = np.arange(len(input_types))
    means = np.array(means)
    stds = np.array(stds)
    lower_err = np.zeros(len(stds))
    upper_err = np.minimum(stds, 1 - means)
    
    yerr = np.vstack([lower_err, upper_err])
    # Build the plot    
    ax.bar(x_pos+spacing, means, yerr=yerr, align='center',width=0.2, alpha=0.9, ecolor='#333333', linewidth=0.5,
           capsize=1.5, color=color,label=label, edgecolor = "gray")
    
    ax.set_ylabel('Proportion successful test trials')
    ax.set_xlabel('Noise')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(input_types)
    #ax.set_title('Test performance')
    ax.set_ylim(top=1.15)
    #ax.yaxis.grid(True)
    
    return means, stds

def success_rate_per_run(test_data_csv): 
    test_data = pd.read_csv(test_data_csv, sep='\t')
    
    counts = []
    groups = test_data.groupby('run')
    
    for g in groups:
        try:
            counts.append((100 - g[1].steps.value_counts()[100]) / 100)
        except:
            counts.append(1.0)
    
    return np.array(counts)


def collect_rates(data_sources):
    """Return list of per-run arrays (one per session)"""
    all_rates = []
    for d in data_sources:
        rates = success_rate_per_run(d)
        rates = rates[~np.isnan(rates)]
        all_rates.append(rates)
    return all_rates

def compute_pvals(control, cereb):
    pvals = []
    for c, v in zip(control, cereb):
        n = min(len(c), len(v))
        stat, p = wilcoxon(c[:n], v[:n])
        pvals.append(p)
    return pvals


def build_df_single_condition(control, cereb):
    rows = []
    
    for session_idx, (c_vals, v_vals) in enumerate(zip(control, cereb)):
        n = min(len(c_vals), len(v_vals))
        
        for i in range(n):
            rows.append([i, session_idx, 'control', c_vals[i]])
            rows.append([i, session_idx, 'cerebellar', v_vals[i]])
    
    return pd.DataFrame(rows, columns=['run', 'session', 'group', 'value'])


#Comparison of test performance
folders = ['../data/rochefort/control_agent',
           '../data/rochefort/cerebellar_agent']

trials = [0,100,200,300,500]
light_test = ["test_log_both_{}.csv".format(trial) for trial in trials]
dark_test = ["test_log_vector_only_{}.csv".format(trial) for trial in trials]

light_test_data = [[f+'/'+test for test in light_test] for f in folders]
dark_test_data = [[f+'/'+test for test in dark_test] for f in folders]


#Collect per-run data for statistics
light_control = collect_rates(light_test_data[0])
light_cereb   = collect_rates(light_test_data[1])

dark_control = collect_rates(dark_test_data[0])
dark_cereb   = collect_rates(dark_test_data[1])

df_light = build_df_single_condition(light_control, light_cereb)
df_dark  = build_df_single_condition(dark_control, dark_cereb)


aov_light = AnovaRM(df_light,
                    depvar='value',
                    subject='run',
                    within=['session', 'group'])

res_light = aov_light.fit()
print(res_light)

df_dark = build_df_single_condition(dark_control, dark_cereb)

aov_dark = AnovaRM(df_dark,
                   depvar='value',
                   subject='run',
                   within=['session', 'group'])

res_dark = aov_dark.fit()
print(res_dark)

pvals_light = compute_pvals(light_control, light_cereb)
pvals_dark  = compute_pvals(dark_control, dark_cereb)

reject, pvals_corr, _, _ = multipletests(pvals_light, method='fdr_bh')

#--------------------------------ENDSTATS####
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5,2.5), dpi=300)

#
plot_test_bars(light_test_data[0], ax=ax[0], input_types=trials, color='white', label='Control')
plot_test_bars(light_test_data[1], spacing=0.2, ax=ax[0], input_types=['L1','L2','L3','L4','L5'],
               color='black', label='Cerebellar')
   
plot_test_bars(dark_test_data[0], ax=ax[1], input_types=trials, color='white')
plot_test_bars(dark_test_data[1], spacing=0.2, ax=ax[1], input_types=['D1','D2','D3','D4','D5'],
               color='black')


x = np.arange(len(trials))

for i, p in enumerate(pvals_light):
    x1 = x[i]
    x2 = x[i] + 0.2
    
    # get bar heights (approximate from means)
    y = max(light_control[i].mean(), light_cereb[i].mean()) + 0.05
    
    add_sig(ax[0], x1, x2, y, p)

for i, p in enumerate(pvals_dark):
    x1 = x[i]
    x2 = x[i] + 0.2
    
    y = max(dark_control[i].mean(), dark_cereb[i].mean()) + 0.05
    
    add_sig(ax[1], x1, x2, y, p)

    
ax[0].yaxis.grid(False)
ax[1].yaxis.grid(False)
ax[0].set_xlabel('                                           Sessions')
ax[1].set_xlabel('')
ax[0].set_title(' ')
ax[1].set_title(' ')
ax[1].set_ylabel(' ')
ax[1].set_facecolor('lightgray')

#plt.tight_layout()
plt.rcParams.update({'font.size': 10})
ax[0].legend(ncols=2)
fig.savefig('figs/light_dark_test_performance.svg', format='svg', dpi=300)