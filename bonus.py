import pandas as pd
import numpy as np


def compute_bonus(df, rate):
    return np.round(df[df['round_number']>3]['payoff'].sum()*rate, 2)


def exclude_bot_responses(df):
    return df[df['rt1']!=-1]


def main(df):
    sessions = df['session'].unique()

    for sess in sessions:
        d = df[df['session'] == sess]
        subjects = d['prolific_id'].unique()
        print('*'*30)
        print(f'Sess = {sess}')
        print(f'N={len(d["prolific_id"].unique())}')
        print('*'*30)
        for sub in subjects:
            dd = exclude_bot_responses(d[d['prolific_id'] == sub])
            rate = .003
            if len(dd) < 60:
                # print(sub)
                print(f'{sub},{compute_bonus(dd, rate)}, n={len(dd)}')
                # print(f'{sub},{compute_bonus(dd, rate)}')


if __name__ == '__main__':

    df = pd.read_csv('data/cost_single_final_58.csv')
    main(df)