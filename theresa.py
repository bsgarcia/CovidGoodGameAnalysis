# import scipy.stats as stats
import itertools as it
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import numpy as np

pal = sns.color_palette("Paired", 10)
sns.set_palette(pal)

# ------------------------------------------------------------------------------------------------------------
# Simulations
# ------------------------------------------------------------------------------------------------------------
def heatmaps():
    good = 2.5
    bad = 1.5

    vmin = 7.5
    vmax = 25

    labelx = 'Contribution i'
    labely = 'Contribution j'
    cbar_kw = {'label': 'payoff for i'}
    # creating a colormap
    colormap = plt.get_cmap("Blues")

    for t1, t2 in ((good, good), (bad, bad), (good, bad), (bad, good)):
        title = '$i_{multiplier}=' + str(t1) + ',\ j_{multiplier}=' + str(t2) + '$'
        data = np.zeros((11, 11))
        for i in range(11):
            for j in range(11):
                data[j, i] = 10 - i + ((i*t1 + j*t2)/2)

        sns.heatmap(data, vmin=vmin, vmax=vmax, cbar_kws=cbar_kw, annot=True, fmt='.1f', cmap=colormap)
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.title(title)
        plt.show()


# ------------------------------------------------------------------------------------------------------------
# By Matching
# ------------------------------------------------------------------------------------------------------------
def disclosure_by_group(df1):

    for exp in (1,2):

        df  = df1[df1['exp']==exp]
        N = len(df['prolific_id'].unique())
        # plot 3
        ax = plt.subplot(int(str(12) + str(exp)))
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
            hue_order.append(f'i={i}, j={j}')

        sem1 = dff_gg.groupby('id_in_session')['disclose'].mean().sem()
        sem2 = dff_gb.groupby('id_in_session')['disclose'].mean().sem()
        sem3 = dff_bb.groupby('id_in_session')['disclose'].mean().sem()
        sem4 = dff_bg.groupby('id_in_session')['disclose'].mean().sem()

        average1 = dff_gg.groupby('id_in_session')['disclose'].mean()
        average2 = dff_gb.groupby('id_in_session')['disclose'].mean()
        average3 = dff_bb.groupby('id_in_session')['disclose'].mean()
        average4 = dff_bg.groupby('id_in_session')['disclose'].mean()

        mean1 = dff_gg.groupby('id_in_session')['disclose'].mean().mean()
        mean2 = dff_gb.groupby('id_in_session')['disclose'].mean().mean()
        mean3 = dff_bb.groupby('id_in_session')['disclose'].mean().mean()
        mean4 = dff_bg.groupby('id_in_session')['disclose'].mean().mean()

        print('N gg=', len(dff_gg))
        print('N bg=', len(dff_bg))
        print('N bb=', len(dff_bb))
        print('N gb=', len(dff_gb))

        sns.barplot(x=hue_order, y=[mean1, mean2, mean3, mean4], ci=None, alpha=.4)
        sns.stripplot(data=[average1, average2, average3, average4], alpha=.8)
        
        plt.errorbar(
            [0, 1, 2, 3], y=[mean1, mean2, mean3, mean4], yerr=[sem1, sem2, sem3, sem4], lw=2.5,
            capsize=3, capthick=2.5, ecolor='black', ls='none', zorder=10)


        plt.title(f'Exp. {exp}, N={N}')
        plt.ylabel('average disclose of i')
        plt.xticks(range(4), hue_order)
        plt.ylim([0, 1.1])
        plt.draw()
    plt.show()


def contribution_by_group(df1):

    for exp in (1,2):

        df  = df1[df1['exp']==exp]
        N = len(df['prolific_id'].unique())
        # plot 3
        ax = plt.subplot(int(str(12) + str(exp)))
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
            hue_order.append(f'i={i}, j={j}')

        sem1 = dff_gg.groupby('id_in_session')['contribution'].mean().sem()
        sem2 = dff_gb.groupby('id_in_session')['contribution'].mean().sem()
        sem3 = dff_bb.groupby('id_in_session')['contribution'].mean().sem()
        sem4 = dff_bg.groupby('id_in_session')['contribution'].mean().sem()

        average1 = dff_gg.groupby('id_in_session')['contribution'].mean()
        average2 = dff_gb.groupby('id_in_session')['contribution'].mean()
        average3 = dff_bb.groupby('id_in_session')['contribution'].mean()
        average4 = dff_bg.groupby('id_in_session')['contribution'].mean()

        mean1 = dff_gg.groupby('id_in_session')['contribution'].mean().mean()
        mean2 = dff_gb.groupby('id_in_session')['contribution'].mean().mean()
        mean3 = dff_bb.groupby('id_in_session')['contribution'].mean().mean()
        mean4 = dff_bg.groupby('id_in_session')['contribution'].mean().mean()

        print('N gg=', len(dff_gg))
        print('N bg=', len(dff_bg))
        print('N bb=', len(dff_bb))
        print('N gb=', len(dff_gb))

        sns.barplot(x=hue_order, y=[mean1, mean2, mean3, mean4], ci=None, alpha=.4)
        sns.stripplot(data=[average1, average2, average3, average4], alpha=.8)
        
        plt.errorbar(
            [0, 1, 2, 3], y=[mean1, mean2, mean3, mean4], yerr=[sem1, sem2, sem3, sem4], lw=2.5,
            capsize=3, capthick=2.5, ecolor='black', ls='none', zorder=10)


        plt.title(f'Exp. {exp}, N={N}')
        plt.ylabel('average contribution of i')
        plt.xticks(range(4), hue_order)
        plt.ylim([0, 10.13])
        plt.draw()
    plt.show()


# ------------------------------------------------------------------------------------------------------------
# Over Time
# ------------------------------------------------------------------------------------------------------------

def over_time_2(df):

    # exclude bots
    # data = df[df['rt1'] != -1]
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
    

    for exp in (1, ):

        # data['contribution'] = data['contribution'] / 10
        plt.figure(figsize=(4.2, 4.5))

        # plt.text(x=8, y=.5, s="random treatment", color="grey", alpha=.9)
        # plt.text(x=38, y=.5, s="sorting treatment", color="grey", alpha=.9)
        # plt.plot([30, 30], [0, 10], linestyle='--', color="grey")
        # ax.add_patch(rect_sorting)
        colors = ["C9", "C3"]
        data = df[df.groupby(['group_id'])['multiplier'].transform('nunique') > 1]
        data = data[data['exp']==exp]
        for color, multiplier in zip(colors, (1, 2)):
            sns.lineplot(
                x='round_number',
                y='payoff',
                # hue='disclosure_group',
                data=data[data['disclosure_group'] == multiplier],
                label=multiplier,
                ci='sem', color=color, zorder=1, alpha=1)

        # sns.lineplot(x='round_number', y='contribution', data=data, ci="sem", label="Both", color="black")
        plt.title(f'Exp. {exp}')
        plt.show()

    # plt.title("Contribution over time")
    # plt.ylabel('Contribution level')
    # plt.xlabel('Round number')
    # plt.ylim([0, 10])
    # plt.xlim([0, 60])
    # plt.legend(loc="upper left")
    # plt.show()

    # for multiplier in (1.5, 2.5):
    #     sns.lineplot(
    #         x='round_number',
    #         y='disclose',
    #         # hue='multiplier',
    #         data=data[data['multiplier'] == multiplier],
    #         label=multiplier,
    #         ci='sem')

    # plt.title(f"Disclosure over time")
    # plt.ylabel('Rate')
    # plt.ylim([0, 1.1])
    # plt.show()


def over_time(df, var):

    for exp in (1, 2):

        data = df[df['exp']==exp]
        plt.subplot(2, 1, 1)

        colors = ["C9", "C3"]
        for color, multiplier in zip(colors, (1.5, 2.5)):
            ax = sns.lineplot(
                x='round_number',
                y=var,
                data=remove_bots(data[data['multiplier'] == multiplier]),
                label=multiplier,
                ci='sem', color=color, zorder=100, alpha=1)
            
        plt.ylim([0, ax.get_ylim()[1]])

        if exp == 1:
            plt.text(x=8, y=.5, s="random treatment", color="grey", alpha=.9)
            plt.text(x=38, y=.5, s="sorting treatment", color="grey", alpha=.9)
            plt.plot([30, 30], [0, ax.get_ylim()[1]], linestyle='--', color="grey", zorder=-1)


        plt.title(f"Exp. {exp}")
        # plt.ylabel('Contribution level')
        plt.xlabel('')
        # plt.ylim([0, 10])
        plt.xlim([0, 60])
        plt.legend(loc="lower right")
        plt.draw()

        plt.subplot(2, 1, 2)

        groups = []
        for t in range(1, data['round_number'].max()+1):
            d = data['group_id'][data['round_number']==t].unique()
            count = {k: 0 for k in it.product((1.5, 2.5), repeat=2)}
            del count[2.5, 1.5]

            for i in d:
                m1 = data[data['group_id']==i]['multiplier'].iloc[0]
                m2 = data[data['group_id']==i]['multiplier'].iloc[1]
                if all(i in (m1, m2) for i in (1.5, 2.5)):
                    count[1.5, 2.5] += 1
                else:
                    count[m1, m2] += 1
            
            groups.append({'t': t,  **count})

        dd = pd.DataFrame(groups)
        for k in ((1.5, 2.5), (2.5, 2.5), (1.5, 1.5)):
            ax = sns.lineplot(
                x='t',
                y=k,
                data=dd,
                ci='sem', zorder=100, alpha=1, label=k)
            
        plt.ylabel('Count')
        plt.ylim([0, ax.get_ylim()[1]])

        if exp == 1:
            plt.text(x=8, y=.5, s="random treatment", color="grey", alpha=.9)
            plt.text(x=38, y=.5, s="sorting treatment", color="grey", alpha=.9)
            plt.plot([30, 30], [0, ax.get_ylim()[1]], linestyle='--', color="grey", zorder=-1)


        # plt.title(f"Exp. {exp}")
        # plt.ylabel('Contribution level')
        plt.xlabel('Round number')
        # plt.ylim([0, 10])
        plt.xlim([0, 60])
        plt.legend(loc="lower right")
        plt.draw()

        plt.show()


# ------------------------------------------------------------------------------------------------------------
# Barplot by multiplier
# ------------------------------------------------------------------------------------------------------------

def disclosure_according_to_multiplier(df):
    df = df.groupby('prolific_id').mean()
    print('N=', len(df))
    sns.barplot(x='multiplier', y='disclose', alpha=.4, ci=68, data=df, hue='exp')
    plt.legend('off')
    sns.stripplot(x='multiplier', y='disclose', hue='exp', data=df, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True)
    for multiplier in (1.5, 2.5):
        print('*'*20)
        print(f'Multiplier. {multiplier}')
        x = df[df['multiplier']==multiplier]['disclose'][df['exp']==1]
        y = df[df['multiplier']==multiplier]['disclose'][df['exp']==2]
        print(len(x), len(y))
        res = pg.ttest(x, y)
        print(res)
    for exp in (1, 2):
        print('*'*20)
        print(f'Exp. {exp}')
        x = df[df['multiplier']==1.5]['disclose'][df['exp']==exp]
        y = df[df['multiplier']==2.5]['disclose'][df['exp']==exp]
        print(len(x), len(y))
        res = pg.ttest(x, y)
        print(res)

    # plt.legend('')
    plt.show()


def payoff_according_to_multiplier(df):
    df = df.groupby('prolific_id').mean()
    print('N=', len(df))
    sns.barplot(x='multiplier', y='norm_payoff', alpha=.4, ci=68, data=df, hue='exp')
    plt.legend('off')
    sns.stripplot(x='multiplier', y='norm_payoff', hue='exp', data=df, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True)
    for multiplier in (1.5, 2.5):
        print('*'*20)
        print(f'Multiplier. {multiplier}')
        x = df[df['multiplier']==multiplier]['disclose'][df['exp']==1]
        y = df[df['multiplier']==multiplier]['disclose'][df['exp']==2]
        print(len(x), len(y))
        res = pg.ttest(x, y)
        print(res)
    for exp in (1, 2):
        print('*'*20)
        print(f'Exp. {exp}')
        x = df[df['multiplier']==1.5]['disclose'][df['exp']==exp]
        y = df[df['multiplier']==2.5]['disclose'][df['exp']==exp]
        print(len(x), len(y))
        res = pg.ttest(x, y)
        print(res)

    # plt.legend('')
    plt.show()


def contribution_according_to_multiplier(df):
    df = df[df['round_number']>30]
    df = df.groupby('prolific_id').mean()
    print('N=', len(df))
    sns.barplot(x='multiplier', y='contribution', alpha=.4, ci=68, data=df, hue='exp')
    plt.legend('off')
    sns.stripplot(x='multiplier', y='contribution', hue='exp', data=df, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True)
    for multiplier in (1.5, 2.5):
        print('*'*20)
        print(f'Multiplier. {multiplier}')
        x = df[df['multiplier']==multiplier]['contribution'][df['exp']==1]
        y = df[df['multiplier']==multiplier]['contribution'][df['exp']==2]
        print(len(x), len(y))
        res = pg.ttest(x, y)
        print(res)
    for exp in (1, 2):
        print('*'*20)
        print(f'Exp. {exp}')
        x = df[df['multiplier']==1.5]['contribution'][df['exp']==exp]
        y = df[df['multiplier']==2.5]['contribution'][df['exp']==exp]
        print(len(x), len(y))
        res = pg.ttest(x, y)
        print(res)
    # plt.legend('')
    plt.show()


def lm(df):

    data = data[df['round_number']<31]

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

    print(stats.spearmanr(x, y))

    y2 = model.predict(X)
    sns.scatterplot(
        x=x,
        y=y, alpha=.6, s=100)

    plt.plot(x, y2)

    plt.ylabel('Contribution rate')
    plt.xlabel('Disclosure rate')
    plt.show()


# ------------------------------------------------------------------------------------------------------------
# Cleaning functions
# ------------------------------------------------------------------------------------------------------------

def add_max_payoff(df):
    group_ids = df['group_id'].unique()
    df['max_payoff'] = np.NaN
    matching = {
        (1.5, 2.5): [22.5, 20],
        (2.5, 1.5): [20, 22.5],
        (2.5, 2.5): [25, 25],
        (1.5, 1.5): [17.5, 17.5]
    }
    
    for g in group_ids:
        d = df[df['group_id']==g]
        if len(d) == 2:
            idx = d.index.tolist()
            t1 = d.iloc[0]['multiplier']
            t2 = d.iloc[1]['multiplier']

            max_payoff = matching[t1, t2]

            df.loc[idx[0], 'max_payoff'] = max_payoff[0] 
            df.loc[idx[1], 'max_payoff'] = max_payoff[1] 
    return df

def add_norm_payoff(df):
    df['norm_payoff'] = df['payoff']/df['max_payoff']
    return df
            
def remove_bots(df):
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
    return df

def remove_trials_where_both_players_are_bots(df):
    group_ids = df['group_id'].unique()
    for g in group_ids:
        d = df[df['group_id']==g]
        idx = d.index.tolist()
        rt1 = d.iloc[0]['rt1'] == -1
        rt2 = d.iloc[1]['rt1'] == -1
        if rt1 and rt2:
            df = df.drop(idx)
    return df


def merge_exp(df1, df2):
    df1['exp'] = 1
    df2['exp'] = 2
    df = pd.concat([df1 , df2])
    df['exp'] = df['exp'].astype(int)
    return df
     


if __name__ == '__main__':

    # heatmaps()

    df2 = pd.read_csv('data/theresa_control.csv')
    df1 = pd.read_csv('data/theresa_baseline.csv')

    # add normalized payoffs to dataframes
    df1 = add_max_payoff(df1)
    df2 = add_max_payoff(df2)
    df1 = add_norm_payoff(df1)
    df2 = add_norm_payoff(df2)

    # remove bots
    df1 = remove_trials_where_both_players_are_bots(df1)
    df2 = remove_trials_where_both_players_are_bots(df2)
    
    # avoid identical group_ids among exp.1 and exp.2 
    df2['group_id'] += 20000

    # merge exp data and add an exp column which specifies the exp number (1 or 2)
    df = merge_exp(df1, df2)
    
    # provide dataframe  + variable to plot as y over round_number
    over_time(df, 'norm_payoff')
    



