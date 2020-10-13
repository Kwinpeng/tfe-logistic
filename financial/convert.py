import pandas as pd

def convert(prefix):
    dfx = pd.read_csv('X_train_woe.csv')
    dfx = dfx.reset_index()
    dfy = pd.read_csv('Y_train_aligned.csv')
    dfy = dfy.reset_index()

    y_cols = dfy.columns
    dfy.rename(columns={y_ori:'xy%d'%i for i, y_ori in enumerate(y_cols) if i > 0 and y_ori != 'y'}, inplace=True)

    df = pd.merge(dfx, dfy, left_on='index', right_on='index', how='inner')
    df.pop('index')

    n_feature = len(df.columns) - 1
    print(f'# of feature: {n_feature}')

    if len(dfx) != len(df) or len(dfx) != len(df):
        print('Error: mis-matched size')

    dfy = pd.DataFrame(df['y'])
    dfy.insert(1, 'x', 0.)
    dfy.to_csv(f'{prefix}_tfe_guest.csv', index=False, header=False)

    dfx = df
    dfx.pop('y')
    dfx.to_csv(f'{prefix}_tfe_host.csv', index=False, header=False)

if __name__ == '__main__':

    convert('financial')

