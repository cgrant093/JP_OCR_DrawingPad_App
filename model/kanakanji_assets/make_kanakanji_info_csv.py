
import os
import pandas as pd
from xml.dom import minidom

def create_kana_df(kanakanji_vocab):
    data = []
    with open(kanakanji_vocab, encoding='utf-8') as file:
        data = [
            [
                char.strip(), 
                hex(int(line.split()[0].strip(), 16) + shift)
            ]
            for line in file 
            for shift, char in enumerate(line.split()[1:])
        ]
    return pd.DataFrame(data, columns=['kanakanji', 'unicode'])

def get_character_stroke_count(svg_file):
    # svg files from: https://github.com/KanjiVG/kanjivg/tree/master
    loc = os.path.join('kanakanji_svg', svg_file) 
    try:
        doc = minidom.parse(loc)
        stroke_count = len(doc.getElementsByTagName('text')) # 'path'
        doc.unlink()
        return stroke_count
    except:
        return 0

def main():
    # create dataframe, make strokeCount column, and 
    #   drop any row that has strokeCount = 0 b/c we don't have stroke info for them
    df = create_kana_df('kanakanji_unicode.txt')
    df['svgFile'] = df['unicode'].apply(lambda r: f'{r[0]}{r[2:]}.svg')
    df['pngFile'] = df['unicode'].apply(lambda r: f'{r[0]}{r[2:]}.png')
    df['strokeCount'] = df['svgFile'].apply(lambda r: get_character_stroke_count(r))
    df = df[df['strokeCount'] != 0]

    # print(df.head())
    # print(df[['kanakanji', 'strokeCount']].groupby('strokeCount').count())
    # for index, row in df[df.strokeCount == 1].iterrows():
    #     print(row.kanakanji, row.unicode)

    # # send info to csv so I don't have to keep remaking it
    df.to_csv('kanakanji_info.csv', index=False)

if __name__ == '__main__':
    main()


