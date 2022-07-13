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

def over_time_with_count(df, var):


    for exp in df['exp'].unique():
        n = len(df['prolific_id'].unique())
        print(f'Exp. {exp}, N={n}')


        data = df[df['exp']==exp]
        f1, f2 = sorted(data['multiplier'].unique())
        plt.subplot(2, 1, 1)

        colors = ["C9", "C3"]
        for color, multiplier in zip(colors, (f1, f2)):
            ax = sns.lineplot(
                x='round_number',
                y=var,
                # hue='disclosure_group',
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
            count = {k: 0 for k in it.product((f1, f2), repeat=2)}
            del count[f2, f1]

            for i in d:
                m1 = data[data['group_id']==i]['multiplier'].iloc[0]
                m2 = data[data['group_id']==i]['multiplier'].iloc[1]
                if all(i in (m1, m2) for i in (f1, f2)):
                    count[f1, f2] += 1
                else:
                    count[m1, m2] += 1
            
            groups.append({'t': t,  **count})

        dd = pd.DataFrame(groups)
        for k in ((f1, f2), (f2, f2), (f1, f1)):
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




def over_time(df, var):


    for exp in df['exp'].unique():
        n = len(df['prolific_id'].unique())
        print(f'Exp. {exp}, N={n}')


        data = df[df['exp']==exp]
        f1, f2 = sorted(data['multiplier'].unique())

        colors = ["C9", "C3"]
        for color, multiplier in zip(colors, (f1, f2)):
            ax = sns.lineplot(
                x='round_number',
                y=var,
                # hue='disclosure_group',
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


def according_to_multiplier(df, var):
    df = df[df['round_number']>30]
    df = df.groupby('prolific_id').mean()
    exp = sorted(df['exp'].unique())
    print('N=', len(df))
    colors = ['C9', 'C3']
    sns.barplot(x='exp', y=var, alpha=.4, ci=68, data=df, hue='multiplier', palette=colors)
    plt.legend('off')
    sns.stripplot(x='exp', y=var, hue='multiplier', data=df, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True, palette=colors)
    # import pdb;pdb.set_trace()
    res = pg.pairwise_ttests(within_first=True,padjust='bonf', parametric=False, dv=var, between=['multiplier', 'exp'], data=df, subject='id_in_session')
    print(pg.print_table(res))
    res.to_csv(f'{var}2.csv')

    # plt.legend('')
    plt.show()

def payoff_according_to_disclosure_group(df):
    f1, f2 = sorted(df['multiplier'].unique())
    df = df[df['round_number']>30]
    df = df.groupby('prolific_id').mean()

    print('N=', len(df))
    sns.barplot(x='disclosure_group', y='norm_payoff', alpha=.4, ci=68, data=df)
    # plt.legend('off')
    sns.stripplot(x='disclosure_group', y='norm_payoff', data=df, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True)
    # for multiplier in (f1, f2):
    #     print('*'*20)
    #     print(f'Multiplier. {multiplier}')
    x = df[df['disclosure_group']==1]['norm_payoff']
    y = df[df['disclosure_group']==2]['norm_payoff']
    print(x)
    print(y)
    print(pg.mwu(x, y))
    #     print(len(x), len(y))
    #     res = pg.ttest(x, y)
    #     print(res)
    # # for exp in (1, 2):
    #     print('*'*20)
    #     print(f'Exp. {exp}')
    #     x = df[df['multiplier']==1.5]['disclose'][df['exp']==exp]
    #     y = df[df['multiplier']==2.5]['disclose'][df['exp']==exp]
    #     print(len(x), len(y))
    #     res = pg.ttest(x, y)
    #     print(res)

    # plt.legend('')
    plt.show()


def according_to_disclosure_group(df, key):
    f1, f2 = sorted(df['multiplier'].unique())
    # df = df[df['round_number']>30]
    # df = df.groupby('prolific_id').mean()
    
    new_data = []
    # add disclosure groups when round < 30
    for idx in df['prolific_id'].unique():
        disclosure_group = int(df[df['prolific_id']==idx][df['round_number']==35]['disclosure_group'])
        multiplier = df[df['prolific_id']==idx]['multiplier'].unique()[0]
        value_before = float(df[df['prolific_id']==idx][df['round_number']<30][key].mean())
        value_after = float(df[df['prolific_id']==idx][df['round_number']>30][key].mean())
        new_data.append({
            'prolific_id': idx, 'disclosure_group': disclosure_group, 'period': 'before_sorting', key: value_before, 'multiplier': multiplier
        })
        new_data.append({
            'prolific_id': idx, 'disclosure_group': disclosure_group, 'period': 'after_sorting', key: value_after,'multiplier': multiplier
        })

    new_data = pd.DataFrame(new_data)

    plt.figure()
    sns.barplot(x='period', y=key, hue='disclosure_group', alpha=.4, ci=68, data=new_data)
    # plt.legend('off')
    sns.stripplot(x='period', y=key, hue='disclosure_group', data=new_data, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True)
    # plt.title('Before')
    # 
    # plt.figure()
    # df2 = df[df['round_number']>30]
    # df2 = df2.groupby('prolific_id').mean()
    # sns.barplot(x='disclosure_group', y='contribution', alpha=.4, ci=68, data=df2)
    # plt.legend('off')
    # sns.stripplot(x='disclosure_group', y='contribution', data=df2, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True)
    # plt.title('after')

    # for multiplier in (f1, f2):
    #     print('*'*20)
    #     print(f'Multiplier. {multiplier}')
    # x = df[df['disclosure_group']==1]['norm_payoff']
    # y = df[df['disclosure_group']==2]['norm_payoff']
    # print(x)
    # print(y)
    # print(pg.mwu(x, y))
    #     print(len(x), len(y))
    #     res = pg.ttest(x, y)
    #     print(res)
    # # for exp in (1, 2):
    #     print('*'*20)
    #     print(f'Exp. {exp}')
    #     x = df[df['multiplier']==1.5]['disclose'][df['exp']==exp]
    #     y = df[df['multiplier']==2.5]['disclose'][df['exp']==exp]
    #     print(len(x), len(y))
    #     res = pg.ttest(x, y)
    #     print(res)

    # plt.legend('')
    plt.show()




def period_comparison(df):
    # f1, f2 = sorted(df1['multiplier'].unique())

    df['period'] = np.NaN
    df.loc[df['round_number']>30, 'period'] = 2
    df.loc[df['round_number']<30, 'period'] = 1

    for period in (1, 2):
        
        
        data = df[df['period']==period].groupby('prolific_id').mean()

        sns.barplot(x='multiplier', y='disclose', alpha=.4, ci=68, data=data)
        sns.stripplot(x='multiplier', y='disclose', data=data, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True)
    # for multiplier in (f1, f2):
    #     print('*'*20)
    #     print(f'Multiplier. {multiplier}')

    x = df[df['period']==1].groupby('prolific_id').mean()['disclose']
    y = df[df['period']==2].groupby('prolific_id').mean()['disclose']
    print(x)
    print(y)
    print(pg.mwu(x, y))
    print(pg.ttest(x, y))
    plt.show()
    #



def payoff_cross_exp_disc_group(df):
    # f1, f2 = sorted(df1['multiplier'].unique())

    df = df[df['round_number']>30]

    df = df.groupby('prolific_id').mean()

    print('N=', len(df))

    sns.barplot(x='exp', y='norm_payoff', alpha=.4, ci=68, data=df)
    # plt.legend('off')
    sns.stripplot(x='exp', y='norm_payoff', data=df, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True)
    # for multiplier in (f1, f2):
    #     print('*'*20)
    #     print(f'Multiplier. {multiplier}')

    x = df[df['exp']==1]['norm_payoff']
    y = df[df['exp']==2]['norm_payoff']
    print(x)
    print(y)
    print(pg.mwu(x, y))
    print(pg.ttest(x, y))
    plt.show()
    #

def expected_payoff_bar(df):
    
    for exp in df['exp'].unique():
        plt.figure()
        data = df[df['exp']==exp]
    df = df[df['round_number']>30]
    data = df.groupby(['prolific_id']).mean()
    sns.barplot(x='exp', y='norm_payoff', alpha=.4, ci=68, data=data)
    sns.stripplot(x='exp', y='norm_payoff', data=data, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True)
    # plt.title(f'Exp. {exp}')
    plt.ylim([0, 1])

    # for multiplier in (1.5, 2.5):
        # print('*'*20)
        # print(f'Multiplier. {multiplier}')
        # x = df[df['multiplier']==multiplier]['contribution'][df['exp']==2]
        # y = df[df['multiplier']==multiplier]['contribution'][df['exp']==3]
        # print(len(x), len(y))
        # res = pg.ttest(x, y)
        # print(res)
    # for exp in (2, 3):
        # print('*'*20)
        # print(f'Exp. {exp}')
    df = df.groupby(['prolific_id']).mean()
    x = df['norm_payoff'][df['exp']==2]
    y = df['norm_payoff'][df['exp']==3]
    print(len(x), len(y))
    res = pg.ttest(x, y)
    print(res)

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
        x = df[df['multiplier']==multiplier]['contribution'][df['exp']==2]
        y = df[df['multiplier']==multiplier]['contribution'][df['exp']==3]
        print(len(x), len(y))
        res = pg.ttest(x, y)
        print(res)
    for exp in (2, 3):
        print('*'*20)
        print(f'Exp. {exp}')
        x = df[df['multiplier']==1.5]['contribution'][df['exp']==exp]
        y = df[df['multiplier']==2.5]['contribution'][df['exp']==exp]
        print(len(x), len(y))
        res = pg.ttest(x, y)
        print(res)
    plt.legend('')
    plt.show()

def payoff_according_to_multiplier(df):
    df = df[df['round_number']>30]
    df = df.groupby('prolific_id').mean()
    print('N=', len(df))
    sns.barplot(x='multiplier', y='norm_payoff', alpha=.4, ci=68, data=df, hue='exp')
    plt.legend('off')
    sns.stripplot(x='multiplier', y='norm_payoff', hue='exp', data=df, edgecolor='white', linewidth=0.6, size=8, alpha=.8, dodge=True)
    for multiplier in (1.5, 2.5):
        print('*'*20)
        print(f'Multiplier. {multiplier}')
        x = df[df['multiplier']==multiplier]['norm_payoff'][df['exp']==2]
        y = df[df['multiplier']==multiplier]['norm_payoff'][df['exp']==3]
        print(len(x), len(y))
        res = pg.ttest(x, y)
        print(res)
    for exp in (2, 3):
        print('*'*20)
        print(f'Exp. {exp}')
        x = df[df['multiplier']==1.5]['norm_payoff'][df['exp']==exp]
        y = df[df['multiplier']==2.5]['norm_payoff'][df['exp']==exp]
        print(len(x), len(y))
        res = pg.ttest(x, y)
        print(res)
    plt.legend('')
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

def get_max_payoff(f1, f2, cost1=0, cost2=0):
    data = np.zeros((11-cost2, 11-cost1))
    for i in range(11-cost1):
        for j in range(11-cost2):
            data[j, i] = 10 - i - cost1 + ((i*f1 + j*f2)/2)
    return np.amax(data)


def add_max_payoff(df, cost=None):
    group_ids = df['group_id'].unique()
    f1, f2 = sorted(df['multiplier'].unique())
    df['max_payoff'] = np.NaN
    matching = {}
    for m1, m2 in ((f1, f2), (f1, f1), (f2, f1), (f2, f2)):
        matching[(m1, m2)] = [get_max_payoff(m1, m2), get_max_payoff(m2, m1)]
        
    for g in group_ids:
        d = df[df['group_id']==g]
        if len(d) == 2:
            idx = d.index.tolist()
            t1 = d.iloc[0]['multiplier']
            t2 = d.iloc[1]['multiplier']
            c1 = d.iloc[0]['disclose']
            c2 = d.iloc[1]['disclose']

            max_payoff = matching[t1, t2]
            
            if cost:
                max_payoff = [get_max_payoff(t1, t2, cost1=cost*c1, cost2=c2*cost), 
                get_max_payoff(t2, t1, cost1=cost*c2, cost2=c1*cost)]

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

def add_subjects(df, num):
    df['id_in_session'] = df['id_in_session']+ (num*100)
    return df

def remove_bots_but_keep_previous_rows(df):
    return df[df['rt1'] != -1]


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


def add_exp_num(df, num):
    df['exp'] = num
    return df


def prepare(df, num):
    df = add_subjects(df, num)
    df = add_max_payoff(df,  cost=2*(num==3))
    df = add_norm_payoff(df)
    print(num)
    import pdb;pdb.set_trace()
    df = remove_trials_where_both_players_are_bots(df)
    df = remove_bots(df)
    df['group_id'] += num*30000
    df = add_exp_num(df, num)
    return df




if __name__ == '__main__':

    # heatmaps()
    # exit() 

    print('Start reading data...')
    df1 = pd.read_csv('data/theresa_sorting.csv')
    df2 = pd.read_csv('data/theresa_control.csv')
    df3 = pd.read_csv('data/theresa_with_cost.csv')
    # df3 = pd.read_csv('data/fernanda_baseline.csv')
    # df3 = df3[df3['session']=='85jbdx27']
    
    # import pdb; pdb.set_trace()

    df1 = prepare(df1, 1)
    df3 = prepare(df3, 3)
    df2 = prepare(df2, 2)
    # add normalized payoffs to dataframes
    # for x, i in enumerate((df1, df2, df3)):
    ndf1 = print(len(df1['id_in_session'].unique()))
    ndf2 = print(len(df2['id_in_session'].unique()))
    ndf3 = print(len(df3['id_in_session'].unique()))
    
    print(df1.head())
    print(ndf1)
    print(ndf2)
    print(ndf3)

    # df3 = add_exp_num(df3, 3)

    # merge exp data and add an exp column which specifies the exp number (1 or 2)
    df = pd.concat([df1, df2, df3])
    print('Done.')
    
    # according_to_multiplier(df, 'contribution') 
    # according_to_multiplier(df, 'disclose')
    according_to_multiplier(df, 'norm_payoff')
    



