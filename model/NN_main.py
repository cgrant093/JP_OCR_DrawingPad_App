
import os
import pandas as pd




def main():
    kanakanji_info_file = os.path.join('kanakanji_assets', 'kanakanji_info.csv')
    kanakanji_info = pd.read_csv(kanakanji_info_file)

    # df_to_print = kanakanji_info[kanakanji_info.strokeCount == 1]
    df_to_print = kanakanji_info
    df_to_print.sort_values('unicode', inplace=True)
    for u, k, s in zip(df_to_print['unicode'].values, df_to_print['kanakanji'].values, df_to_print['strokeCount'].values):
        print(f'{u} {k} {s}')


if __name__ == '__main__':
    main()




