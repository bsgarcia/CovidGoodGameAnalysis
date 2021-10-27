import scipy.stats as stats
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import numpy as np

pal = sns.color_palette("Paired", 10)
sns.set_palette(pal)

def phase_diagram():
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


def disclosure_by_group(df1):
    # exclude bots
    prolific_id_to_exclude = df1[df1['rt1'] == -1]['prolific_id'].unique()
    df1 = df1[~df1['prolific_id'].isin(prolific_id_to_exclude)]
    # df1 = df1[df1['round_number']>30]


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
    # exclude bots
    # prolific_id_to_exclude = df1[df1['rt1'] == -1]['prolific_id'].unique()
    # df1 = df1[~df1['prolific_id'].isin(prolific_id_to_exclude)]

    df2 = df1[(df1['rt1']==-1) & (df1['exp']==1)]#['prolific_id'].unique()
    df3 = df1[(df1['rt1']==-1) & (df1['exp']==2)]#['prolific_id'].unique()
    # df1 = df1[~df1['prolific_id'].isin(prolific_id_to_exclude)]

    import pdb;pdb.set_trace()

    # df1 = df1[df1['round_number']>30]

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


def over_time_2(df):

    # exclude bots
    # data = df[df['rt1'] != -1]
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]

    # data['contribution'] = data['contribution'] / 10
    plt.figure(figsize=(4.2, 4.5))
    # plt.text(x=8, y=.5, s="random treatment", color="grey", alpha=.9)
    # plt.text(x=38, y=.5, s="sorting treatment", color="grey", alpha=.9)
    # plt.plot([30, 30], [0, 10], linestyle='--', color="grey")
    # ax.add_patch(rect_sorting)
    colors = ["C9", "C3"]
    data = df[df.groupby(['group_id'])['multiplier'].transform('nunique') > 1]
    for color, multiplier in zip(colors, (1.5, 2.5)):
        sns.lineplot(
            x='round_number',
            y='contribution',
            data=data[data['multiplier'] == multiplier],
            label=multiplier,
            ci='sem', color=color, zorder=1, alpha=1)

    # sns.lineplot(x='round_number', y='contribution', data=data, ci="sem", label="Both", color="black")

    plt.title("Contribution over time")
    plt.ylabel('Contribution level')
    plt.xlabel('Round number')
    plt.ylim([0, 10])
    plt.xlim([0, 60])
    plt.legend(loc="upper left")
    plt.show()

    for multiplier in (1.5, 2.5):
        sns.lineplot(
            x='round_number',
            y='disclose',
            # hue='multiplier',
            data=data[data['multiplier'] == multiplier],
            label=multiplier,
            ci='sem')

    plt.title(f"Disclosure over time")
    plt.ylabel('Rate')
    plt.ylim([0, 1.1])
    plt.show()


def over_time(df):

    # exclude bots
    # data = df[df['rt1'] != -1]
    for exp in (1, 2):
        data = df[df['exp']==exp]
        prolific_id_to_exclude = data[data['rt1'] == -1]['prolific_id'].unique()
        data = data[~data['prolific_id'].isin(prolific_id_to_exclude)]

        # data['contribution'] = data['contribution'] / 10
        #plt.figure(figsize=(4.2, 4.5))
        plt.figure()

        if exp == 1:
            plt.text(x=8, y=.5, s="random treatment", color="grey", alpha=.9)
            plt.text(x=38, y=.5, s="sorting treatment", color="grey", alpha=.9)
            plt.plot([30, 30], [0, 10], linestyle='--', color="grey")
            # ax.add_patch(rect_sorting)
        colors = ["C9", "C3"]
        for color, multiplier in zip(colors, (1.5, 2.5)):
            sns.lineplot(
                x='round_number',
                y='contribution',
                data=data[data['multiplier'] == multiplier],
                label=multiplier,
                ci='sem', color=color, zorder=1, alpha=1)

        # sns.lineplot(x='round_number', y='contribution', data=data, ci="sem", label="Both", color="black")

        plt.title(f"Exp. {exp}")
        plt.ylabel('Contribution level')
        plt.xlabel('Round number')
        plt.ylim([0, 10])
        plt.xlim([0, 60])
        plt.legend(loc="upper left")
        plt.draw()

        plt.figure()
        if exp == 1:
            plt.text(x=8, y=.5, s="random treatment", color="grey", alpha=.9)
            plt.text(x=38, y=.5, s="sorting treatment", color="grey", alpha=.9)
            plt.plot([30, 30], [0, 10], linestyle='--', color="grey")
         
        for multiplier in (1.5, 2.5):
            sns.lineplot(
                x='round_number',
                y='disclose',
                data=data[data['multiplier'] == multiplier],
                label=multiplier,
                ci='sem')

        plt.title(f"Exp. {exp}")
        plt.ylabel('Rate')
        plt.ylim([0, 1.1])
        plt.draw()
    plt.show()


def difference_matching_disclosure(df):
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
    df_disclosure = df[df['round_number'] > 30]
    mean_disclosure = df_disclosure.groupby('prolific_id', as_index=True)['disclose'].mean()
    df_random = df[df['round_number'] <= 30]
    mean_random = df_random.groupby('prolific_id', as_index=True)['disclose'].mean()

    sns.barplot(data=[mean_random, mean_disclosure], alpha=.5)
    sns.stripplot(data=[mean_random, mean_disclosure], edgecolor='white', linewidth=0.6, size=8, alpha=.8)
    plt.xticks([0, 1], [1.5, 2.5])

    plt.ylabel('Disclosure')
    plt.show()

    res = pg.wilcoxon(np.array(mean_disclosure), np.array(mean_random))
    print(res)


def difference_matching_disclosure_with_other_criteria(df):
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
    df_sorting = df[df['round_number'] > 30]
    mean_sorting = df_sorting.groupby(
        ['prolific_id', 'disclosure_group', 'multiplier'], as_index=False)['contribution'].mean()
    df_random = df[df['round_number'] <= 30]
    mean_random = df_random.groupby(
        ['prolific_id', 'disclosure_group', 'multiplier'], as_index=False)['contribution'].mean()

    # disclosure_group
    # for d_group in (1, 2):
    #     data = [mean_random[mean_random['disclosure_group'] == d_group]['disclose'],
    #             mean_sorting[mean_sorting['disclosure_group'] == d_group]['disclose']]
    #     sns.barplot(data=data, alpha=.5)df['multiplier'], df['contribution'], 
    #     sns.stripplot(data=data, edgecolor='white', linewidth=0.6, size=8, alpha=.8)
    #     plt.xticks([0, 1], ['random', 'sorting'])
    #     plt.title(f'disclosure_group={d_group}')
    #
    #     plt.ylabel('Disclosure')
    #
    #     res = pg.wilcoxon(np.array(data[0]), np.array(data[1]))
    #     print(res)
    #     plt.show()

    colors = [["C2", "C3"], ["C8", "C9"]]
    # multiplier
    for color, multiplier in zip(colors, (2.5, 1.5)):
        data = [mean_random[mean_random['multiplier'] == multiplier]['contribution'],
                mean_sorting[mean_sorting['multiplier'] == multiplier]['contribution']]

        plt.figure(figsize=(3, 4.5))
        sns.barplot(data=data, alpha=.7, palette=color)
        sns.stripplot(data=data, edgecolor='white', linewidth=0.6, size=8, palette=color)# plot 3
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
    sns.stripplot(data=[average1, average2, average3, average4])
    plt.errorbar(
        [0, 1, 2, 3], y=[mean1, mean2, mean3, mean4], yerr=[sem1, sem2, sem3, sem4], lw=2.5,
        capsize=3, capthick=2.5, ecolor='black', ls='none', zorder=10)

    plt.title('contribution by matching')
    plt.ylabel('average contribution of i')
    plt.xticks(range(4), hue_order)
    plt.ylim([0, 10.13])
    plt.show()


    plt.xticks([0, 1], ['random', 'sorting'])
    plt.title(f'multiplier={multiplier}')

    plt.ylabel('Contribution level')

    res = pg.wilcoxon(np.array(data[0]), np.array(data[1]))
    print(res)

    plt.tight_layout()
    plt.show()


def difference_sorting_disclosure(df):
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
    df_sorting = df[df['round_number'] > 30].groupby(
        ['prolific_id', 'disclosure_group'], as_index=False)['disclose'].mean()

    data = [df_sorting[df_sorting['disclosure_group'] == 1]['disclose'],
            df_sorting[df_sorting['disclosure_group'] == 2]['disclose']]
    sns.barplot(data=data, alpha=.5)
    sns.stripplot(data=data, edgecolor='white', linewidth=0.6, size=8, alpha=.8)
    plt.xticks([0, 1], ['group 1', 'group 2'])
    plt.ylabel('Disclosure')

    res = pg.mwu(np.array(data[0]), np.array(data[1])[:18])
    print(res)
    plt.show()


# def contribution_according_to_multiplier(df):
#     prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
#     df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
#     df = df.groupby('prolific_id', as_index=False).mean()
#     print('N=', len(df))
#     plt.figure()
#     sns.barplot(df['multiplier'],df['contribution'],  hue=df['exp'], alpha=.4, ci='sem')
#     sns.stripplot(df['multiplier'], df['contribution'], hue=df['exp'], edgecolor='white', linewidth=0.6, size=8, alpha=.8)
#     # x = df[df['multiplier']==1.5]['contribution'][df['exp']==exp]
#     # y = df[df['multiplier']==2.5]['contribution'][df['exp']==exp]
#     # res = pg.ttest(x, y)
#     # print(res)
#     plt.show()


# def disclosure_according_to_multiplier(df):
#     prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
#     df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
#     df = df.groupby('prolific_id', as_index=False).mean()
#     print('N=', len(df))
#     for exp in (1, 2):
#         plt.figure()
#         sns.barplot(df['multiplier'][df['exp']==exp], df['disclose'][df['exp']==exp], alpha=.4, ci='sem')
#         sns.stripplot(df['multiplier'][df['exp']==exp], df['disclose'][df['exp']==exp], edgecolor='white', linewidth=0.6, size=8, alpha=.8)
#         x = df[df['multiplier']==1.5]['disclose'][df['exp']==exp]
#         y = df[df['multiplier']==2.5]['disclose'][df['exp']==exp]
#         res = pg.ttest(x, y)
#         print(res)
#         plt.draw()
#     plt.show()


def disclosure_according_to_multiplier(df):
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
    df = df[df['round_number']>30]
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




def contribution_according_to_multiplier(df):
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]
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
    # exclude session
    # data = data[data['session'] == sessions[1]]
    # exclude bots
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    data = df[~df['prolific_id'].isin(prolific_id_to_exclude)]

    # data = data[df['round_number']<31]

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


if __name__ == '__main__':
    # main()
    # phase_diagram()
    df1 = pd.read_csv('data/last_data_2.csv')
    df2 = pd.read_csv('data/theresa_baseline.csv')

    df1['exp'] = 2
    df2['exp'] = 1
    df = pd.concat([df1 , df2])
    df['exp'] = df['exp'].astype(int)

    # disclosure_according_to_multiplier(df)
    # contribution_according_to_multiplier(df)
    # lm(df)
    contribution_by_group(df)
    # disclosure_by_group(df)
    # difference_matching_disclosure_with_other_criteria(df)
    # over_time(df)

