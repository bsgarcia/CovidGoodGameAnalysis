import scipy.stats as stats
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import numpy as np

pal = sns.color_palette("Paired", 10)
sns.set_palette(pal)


def heatmap():
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


def over_time(df):

    # exclude bots
    # data = df[df['rt1'] != -1]
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    data = df[~df['prolific_id'].isin(prolific_id_to_exclude)]

    plt.figure(figsize=(6, 4.5))
    plt.text(x=8, y=.5, s="random treatment", color="grey", alpha=.9)
    plt.text(x=38, y=.5, s="sorting treatment", color="grey", alpha=.9)
    plt.plot([30, 30], [0, 10], linestyle='--', color="grey")

    colors = ["C9", "C3"]
    for color, multiplier in zip(colors, (1.5, 2.5)):
        sns.lineplot(
            x='round_number',
            y='contribution',
            data=data[data['multiplier'] == multiplier],
            label=multiplier,
            ci='sem', color=color, zorder=1, alpha=1)

    plt.title("Contribution over time")
    plt.ylabel('Contribution level')
    plt.xlabel('Round number')
    plt.ylim([0, 10])
    plt.xlim([0, 60])
    plt.legend(loc="upper left")
    plt.show()


def difference_contribution_over_matching_according_to_multiplier(df):
    # exclude bots
    prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]

    #select sorting treatment data
    df_sorting = df[df['round_number'] > 30]
    #compute average for each subject
    mean_sorting = df_sorting.groupby(
        ['prolific_id', 'disclosure_group', 'multiplier'], as_index=False)['contribution'].mean()
    #select random treatment data
    df_random = df[df['round_number'] <= 30]
    #compute average for each subject
    mean_random = df_random.groupby(
        ['prolific_id', 'disclosure_group', 'multiplier'], as_index=False)['contribution'].mean()

    colors = [["C2", "C3"], ["C8", "C9"]]
    # multiplier
    for color, multiplier in zip(colors, (2.5, 1.5)):
        data = [mean_random[mean_random['multiplier'] == multiplier]['contribution'],
                mean_sorting[mean_sorting['multiplier'] == multiplier]['contribution']]

        plt.figure(figsize=(3, 4.5))
        sns.barplot(data=data, alpha=.7, palette=color)
        sns.stripplot(data=data, edgecolor='white', linewidth=0.6, size=8, palette=color)
        plt.xticks([0, 1], ['random', 'sorting'])
        plt.title(f'multiplier={multiplier}')

        plt.ylabel('Contribution level')

        res = pg.wilcoxon(np.array(data[0]), np.array(data[1]))
        print(res)

        plt.tight_layout()
        plt.show()


def difference_contribution_in_sorting_according_to_disclosure_group(df):
    # exclude bots
    # prolific_id_to_exclude = df[df['rt1'] == -1]['prolific_id'].unique()
    # df = df[~df['prolific_id'].isin(prolific_id_to_exclude)]

    #select sorting treatment data
    df_sorting = df[df['round_number'] > 30]
    #compute average for each subject
    mean_sorting = df_sorting.groupby(
        ['prolific_id', 'disclosure_group', 'multiplier'], as_index=False)['disclose'].mean()

    colors = ["C4", "C5"]
    # multiplier
    data = [mean_sorting[mean_sorting['disclosure_group'] == 1]['disclose'],
            mean_sorting[mean_sorting['disclosure_group'] == 2]['disclose']]

    plt.figure(figsize=(3, 4.5))
    sns.barplot(data=data, alpha=.7, palette=colors)
    sns.stripplot(data=data, edgecolor='white', linewidth=0.6, size=8, palette=colors)
    plt.xticks([0, 1], ['Non-discloser', 'Discloser'])
    # plt.title(f'Disclosure group={d_group}')

    plt.ylabel('Contribution level')

    res = pg.mwu(np.array(data[0]), np.array(data[1]))
    print(res)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # main()
    heatmap()

    df = pd.read_csv('theresa_baseline.csv')
    # difference_contribution_over_matching_according_to_multiplier(df)
    difference_contribution_in_sorting_according_to_disclosure_group(df)
    # over_time(df)

