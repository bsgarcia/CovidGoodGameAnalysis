import pandas as pd
import scipy.stats as sp
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
# import pymc3 as pm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


# def timeit(method):
#     def timed(*args, **kw):
#         ts = time.time()
#         result = method(*args, **kw)
#         te = time.time()
#         if 'log_time' in kw:
#             name = kw.get('log_name', method.__name__.upper())
#             kw['log_time'][name] = int((te - ts) * 1000)
#         else:
#             print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
#         return result
#     return timed


# @timeit
def phase_diagram():
    good = 2.5
    bad = .8

    vmin = 7.5
    vmax = 25

    labelx = 'Contribution x'
    labely = 'Contribution y'
    cbar_kw = {'label': 'payoff for x'}

    for t1, t2 in ((good, good), (bad, bad), (good, bad), (bad, good)):
        title = f'x={t1} - y={t2}'
        data = np.zeros((11, 11))
        for i in range(11):
            for j in range(11):
                data[j, i] = 10 - i + ((i*t1 + j*t2)/2)

        sns.heatmap(data, vmin=vmin, vmax=vmax, cbar_kws=cbar_kw, annot=True, fmt='.1f')
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.title(title)
        plt.show()


def contribution_by_group(df):
    # exclude bots
    df = df[df['rt1'] != -1]

    # plot 3
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # df_mean = df.groupby(['id', 'multiplier', 'round_id'])['c'].mean()
    dff = df[df.groupby(['group_id'])['multiplier'].transform('nunique') > 1]
    dff_gb = dff[dff['multiplier'] == 2.5]
    dff_bg = dff[dff['multiplier'] == 1.5]
    dff = df[df.groupby(['group_id'])['multiplier'].transform('nunique') == 1]
    dff_gg = dff[dff['multiplier'] == 2.5]
    dff_bb = dff[dff['multiplier'] == 1.5]

    hue_order = []
    for (i, j) in ((2.5, 2.5), (2.5, 1.5), (1.5, 1.5), (1.5, 2.5)):
        hue_order.append(f'x={i}, y={j}')

    sem1 = stats.sem(dff_gg.groupby('id_in_session')['contribution'].mean())
    sem2 = stats.sem(dff_gb.groupby('id_in_session')['contribution'].mean())
    sem3 = stats.sem(dff_bb.groupby('id_in_session')['contribution'].mean())
    sem4 = stats.sem(dff_bg.groupby('id_in_session')['contribution'].mean())

    mean1 = np.mean(dff_gg.groupby('id_in_session')['contribution'].mean())
    mean2 = np.mean(dff_gb.groupby('id_in_session')['contribution'].mean())
    mean3 = np.mean(dff_bb.groupby('id_in_session')['contribution'].mean())
    mean4 = np.mean(dff_bg.groupby('id_in_session')['contribution'].mean())

    sns.barplot(x=hue_order, y=[mean1, mean2, mean3, mean4], ci=None)
    plt.errorbar(
        [0, 1, 2, 3], y=[mean1, mean2, mean3, mean4], yerr=[sem1, sem2, sem3, sem4], lw=2.5,
        capsize=3, capthick=2.5, ecolor='black', ls='none', zorder=10)

    plt.title('contribution by matching')
    plt.ylabel('average contribution of x')
    plt.ylim([0, 10])
    plt.show()


def contribution(df):

    # exclude bots
    df = df[df['rt1'] != -1]

    fig = plt.figure(figsize=(8, 5))
    # Plot 1
    ax = fig.add_subplot(121)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    df_mean = df.groupby(['id_in_session', 'multiplier'], as_index=False)['contribution'].mean()
    df_good = df_mean[df_mean['multiplier'] == 2.5]
    df_bad = df_mean[df_mean['multiplier'] == 1.5]

    sem1 = stats.sem(df_bad['contribution'])
    sem2 = stats.sem(df_good['contribution'])

    mean1 = np.mean(df_bad['contribution'])
    mean2 = np.mean(df_good['contribution'])

    label = [1.5, 2.5]
    y = []
    for i in range(2):
        y.append(df_mean[df_mean['multiplier'] == label[i]]['contribution'].tolist())

    # sns.barplot(x=['bad', 'good'], y=[mean1, mean2], ci=None)
    ax = sns.violinplot(data=y, inner=None, alpha=.2, linewidth=0)

    for x in ax.collections:
        x.set_alpha(.5)

    sns.stripplot(data=y, linewidth=.7, edgecolor='black', alpha=.7, zorder=9)
    plt.errorbar(
        [0, 1], y=[mean1, mean2], yerr=[sem1, sem2], lw=3,
        markersize=7, marker='o', markerfacecolor='w', markeredgecolor='black',
        capsize=4, capthick=2.5, ecolor='black', ls='none', zorder=10)

    plt.title('contribution')
    # ax.get_legend().remove()
    plt.ylim([-0.08 * 10, 1.08 * 10])
    plt.xticks(ticks=[0, 1], labels=[1.5, 2.5])

    # Plot 2
    ax = plt.subplot(122)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    df_mean = df.groupby(['id_in_session', 'multiplier'], as_index=False)['disclose'].mean()
    df_good = df_mean[df_mean['multiplier'] == 2.5]
    df_bad = df_mean[df_mean['multiplier'] == 1.5]

    sem1 = stats.sem(df_bad['disclose'])
    sem2 = stats.sem(df_good['disclose'])

    mean1 = np.mean(df_bad['disclose'])
    mean2 = np.mean(df_good['disclose'])

    label = [1.5, 2.5]
    y = []
    for i in range(2):
        y.append(df_mean[df_mean['multiplier'] == label[i]]['disclose'].tolist())

    ax = sns.violinplot(data=y, inner=None, alpha=.2, linewidth=0)

    for x in ax.collections:
        x.set_alpha(.5)

    sns.stripplot(data=y, linewidth=.7, edgecolor='black', zorder=9, alpha=.7)
    plt.errorbar(
        [0, 1], y=[mean1, mean2], yerr=[sem1, sem2], lw=3, markersize=7, marker='o',
        markerfacecolor='w', markeredgecolor='black',
        capsize=4, capthick=2.5, ecolor='black', ls='none', zorder=10)

    plt.title('disclosure')
    plt.ylim([-0.08, 1.08])
    plt.xticks(ticks=[0, 1], labels=[1.5, 2.5])
    # ax.get_legend().remove()
    plt.show()


def over_time(df):

    # exclude session
    # data = data[data['session'] == sessions[1]]

    # exclude bots
    data = df[df['rt1'] != -1]

    ids = np.unique(data['prolific_id'])

    data['contribution'] = data['contribution'] / 10

    sns.lineplot(
        x='round_number',
        y='contribution',
        # hue='multiplier',
        data=data,
        label='contribution',
        ci='sem')

    sns.lineplot(
        x='round_number',
        y='disclose',
        # hue='multiplier',
        data=data,
        label='disclosure',
        ci='sem')

    plt.title(f"Over time")
    plt.ylabel('Rate')
    plt.ylim([0, 1.1])
    plt.show()


def mw(df):
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
    df = df[df['round_number'] > 30]

    means = [df[df['disclosure_group']==i].groupby('prolific_id', as_index=True)['contribution'].mean() for i in (1,2)]

    res = pg.wilcoxon(means[0], means[1][:12])
    import pdb; pdb.set_trace()


def biglm(df):
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
    df = df.groupby('prolific_id', as_index=False).mean()
    print('N=', len(df))
    # df = df['contribution'].mean()
    # df = df['disclose'].mean()

    # model = sm.OLS.from_formula('contribution ~ multiplier', data=df).fit()

    sns.barplot(df['multiplier'], df['contribution'], alpha=.4, ci='sem')
    sns.stripplot(df['multiplier'], df['contribution'], edgecolor='white', linewidth=0.6, size=8, alpha=.8)
    x = df[df['multiplier']==1.5]['contribution']
    y = df[df['multiplier']==2.5]['contribution']
    res = pg.mwu(x, y)
    print(res)
    plt.show()



def lm(df):
    data = df

    # exclude session
    # data = data[data['session'] == sessions[1]]
    # exclude bots
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    data = df[~df['prolific_id'].isin(prolific_id_to_exclude)]

    x = data.groupby(['prolific_id'], as_index=False)['disclose'].mean()
    y = data.groupby(['prolific_id'], as_index=False)['contribution'].mean()
    # to_exclude = x['prolific_id'][x['disclose']<.1]
    # x = x[~x.isin(to_exclude)].dropna()
    # y = y[~y.isin(to_exclude)].dropna()
    # to_exclude = y['prolific_id'][y['contribution']<.1]
    # x = x[~x.isin(to_exclude)].dropna()
    # y = y[~y.isin(to_exclude)].dropna()

    x = np.array(x['disclose'])
    y = np.array(y['contribution'])
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    print(model.summary())


    print(sp.spearmanr(x, y))

    y2 = model.predict(X)
    sns.scatterplot(
        x=x,
        y=y, alpha=.6, s=100)

    plt.plot(x, y2)

    plt.ylabel('Contribution rate')
    plt.xlabel('Disclosure rate')
    plt.show()


if __name__ == '__main__':
    # main()
    df = pd.read_csv('theresa_player.csv')
    # over_time(df)
    # mw(df)
    biglm(df)
    # contribution(df)
    # contribution_by_group(df)
