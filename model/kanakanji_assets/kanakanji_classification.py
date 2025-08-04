
from math import pi
import numpy as np
import os
import pandas as pd
from PySide6.QtCore import QPoint
import re
import sys
from xml.etree import ElementTree as ET

# # os.chdir('..\\..\\app')
# current_directory = os.getcwd()
# print(f"Searching for Python modules in: {current_directory}")
# quit()
#
# python_modules = []
# for filename in os.listdir(current_directory):
#     if filename.endswith(".py") and os.path.isfile(os.path.join(current_directory, filename)):
#         # Exclude __init__.py if desired, or other special files
#         if filename != "__init__.py":
#             module_name = filename[:-3]  # Remove .py extension
#             python_modules.append(module_name)
#
# if python_modules:
#     print("Found Python modules:")
#     for module in sorted(python_modules):
#         print(f"- {module}")
# else:
#     print("No Python modules found in the current working directory.")
# quit()
sys.path.append(os.getcwd())
from util import UnitCircleDivisionInfo
dir_list = ['l', 's', 'd', 't', 'r', 'p', 'u', 'q']
uc_info = UnitCircleDivisionInfo(dir_list, pi/4)

os.chdir('..\\model\\kanakanji_assets')


def find_elements(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot() 

    root_string = ET.tostring(root[0], encoding='unicode')
    root_list = root_string.split(' ')

    element_set = set()
    for element in root_list:
        if 'type' not in element: continue
        complete_type = element.split('"')[1]
        type_set = set(
            elem for elem in list(complete_type)
            if elem not in ['/', 'a', 'b', 'c', 'v']
        )
        element_set.update(type_set)

    return element_set


def svg_path_counts(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot() 

    root_string = ET.tostring(root[0], encoding='unicode')
    root_list = root_string.split(' ')

    m_count = 0
    c_count = 0
    s_count = 0

    for element in root_list:
        if 'id=' in element: continue
        if 'd=' not in element: continue
        # if 's' not in element.lower(): continue
        if 'm' in element.lower(): m_count += 1
        if 'c' in element.lower(): c_count += 1
        if 's' in element.lower(): s_count += 1
        
    return (m_count, c_count, s_count)



def find_strokes(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot() 

    root_string = ET.tostring(root[0], encoding='unicode')
    
    # root_string = re.sub('M ', 'M', root_string)
    # root_string = re.sub(' c', 'c', root_string)
    # root_string = re.sub('c ', 'c', root_string)
    # root_string = re.sub(' C', 'C', root_string)
    # root_string = re.sub('C ', 'C', root_string)
    # root_string = re.sub(' s', 's', root_string)
    # root_string = re.sub('s ', 's', root_string)
    # root_string = re.sub(' S', 'S', root_string)
    # root_string = re.sub('S ', 'S', root_string)
    # root_string = re.sub(' -', '-', root_string)
    # root_string = re.sub('- ', '-', root_string)
    root_string = re.sub(' ', ',', root_string)
    root_list = root_string.split('"')


    for i, element in enumerate(root_list):
        # if 'id=' in element: continue
        # if 'd=' not in element: continue
        if 'kvg:' in element: continue
        if 'd=' not in root_list[i-1]: continue
        # if 's' not in element.lower(): continue
        # if 'c' not in element.lower(): continue
        element = re.sub('[a-zA-Z]', '\g<0>,', element)
        element = re.sub(',,', ',', element)
        print(svg_file.split('\\')[-1])
        print('\t', element)

    return

    root_list = root_string.split('>')


    for element in root_list:
        # if 'type' in element: print('\t', element)
        # if 'element' in element: print('\t', element)
        # if 'id' in element: print('\t', element)
        # if 'd=' in element: print('\t', element)
        if element.strip() == '': continue
        if '</ns0:g' in element: continue
        print('\t', element)
        # if 'type' not in element: continue
        # if '６' not in element: continue
        # print('\t', svg_file, element)
        # # element_set.add(element.split('"')[1])

    return #element_set

    for element in root.findall(".//ns0:g"):
        print('\t', element)
        # for k, v in child.attrib.items():
        #     if k == 'id': continue
        #     print('\t', k, ': ', v)

    return

    for child in root[0]:
        # for key, value in child.items():
        #     print(key, value)
        # print('\t', 'element: ', child.attrib['element'])
        for k, v in child.attrib.items():
            if k in ['id']: continue
            print('\t', 'c', k, ': ', v)
        # print('\t', child.tag, child.attrib)
        for gc in child:
            # print('\t\t', 'element: ', gc.attrib['element'])
            # print('\t\t', 'position: ', gc.attrib['position'])
            for k, v in gc.attrib.items():
                if k in ['id']: continue
                print('\t\t', 'gc', k, ': ', v)
            # print('\t\t', gc.tag, gc.attrib)
            for ggc in gc:
                # print('\t\t\t', 'type: ', ggc.attrib['type'])
                # print('\t\t\t', 'd: ', ggc.attrib['d'])
                for k, v in ggc.attrib.items():
                    if k in ['id']: continue
                    print('\t\t\t', 'ggc', k, ': ', v)
                # print('\t\t\t', ggc.tag, ggc.attrib)t
                # for element in ggc.findall("{http://kanjivg.tagaini.net}type"):
                #     print('\t\t\t', 'type: ', element)
                for gggc in ggc:
                    # print('\t\t\t', 'type: ', ggc.attrib['type'])
                    # print('\t\t\t', 'd: ', ggc.attrib['d'])
                    for k, v in gggc.attrib.items():
                        if k in ['id']: continue
                        print('\t\t\t\t', 'gggc', k, ': ', v)


def adjust_stroke_path_info(element):
    element = re.sub('[a-zA-Z]', '\g<0>,', element)
    element = re.sub(',,', ',', element)
    element = re.sub('\d-', '\g<0>,-', element)
    element = re.sub('-,-', ',-', element)
    return element

def element_to_steps(element):
    element = re.sub('[a-zA-Z],', '||\g<0>|', element)
    steps = [
        step.split(',|')
        for step in element.split('||')
        if step
    ]
    return steps
    
def conv_string_to_number(number_as_string):
    if number_as_string[0] == '-':
        return -1*float(number_as_string[1:])
    return float(number_as_string)

def split_string_to_numbers(string_of_numbers):
    numbers = [
        conv_string_to_number(num)
        for num in string_of_numbers.split(',')
        if num
    ]
    return np.array(numbers)

def find_direction(s):
    diff = QPoint(s[0], s[1])
    main_direction = ''
    for direction, is_primary in uc_info.find_direction(diff):
        if is_primary == 1:
            main_direction = direction
            break

    return main_direction
    # '''
    # directions key:
    #     u = y+ (within approx +/- 5 deg)
    #     d = y- (within approx +/- 5 deg)
    #     r = x+ (within approx +/- 5 deg)
    #     l = x- (within approx +/- 5 deg)
    #     p ~= Q1 (everything remaining after above removed)
    #     q ~= Q2 (everything remaining after above removed)
    #     s ~= Q3 (everything remaining after above removed)
    #     t ~= Q4 (everything remaining after above removed)
    #     o = starts and ends in same place
    # '''
    # abs_s = np.absolute(s)
    # main_dir = np.max(abs_s)
    # # print('\t\ts', s, sep='\t')
    # # print('\t\tabs_s', abs_s, sep='\t')
    # # print('\t\tdir', main_dir, sep='\t')
    # # print('\t\ts_norm', abs_s / main_dir, sep='\t')
    #
    #
    # if main_dir == 0:
    #     return 'o'
    # s_norm = abs_s / main_dir
    # # if y is main direction and there is little x
    # #   (roughly +/- 5% around the y-axis of unit circle)
    # if s_norm[0] < 0.42:
    #     if s[1] < 0:
    #         return 'u'
    #     # remaining is s[1] > 0
    #     return 'd'
    # # if x is main direction and there list little y
    # #   (roughly +/- 5% around the x-axis of unit circle)
    # elif s_norm[1] < 0.42:
    #     if s[0] > 0:
    #         return 'r'
    #     # remaining is s[0] < 0
    #     return 'l'
    # # all that's remaining is the "middle" of the quadrants
    # elif s[1] < 0:
    #     # Q1
    #     if s[0] > 0:
    #         return 'p'
    #     # Q2
    #     return 'q'
    # # Q3
    # if s[0] < 0:
    #     return 's'
    # #Q4
    # return 't'

def cleanup_directions_list(directions):
    new_directions = directions[0]
    last_dir_loc = 0
    for i in range(1, len(directions)):
        if directions[i] == directions[last_dir_loc]:
            continue
        new_directions += directions[i]
        last_dir_loc = i
    return new_directions

def get_high_level_info(element, last_start_pos):
    steps = element_to_steps(element)

    if steps[0][0] not in ('M', 'm'):
        return [[pd.NA, pd.NA], pd.NA, pd.NA, pd.NA]

    s0 = split_string_to_numbers(steps[0][1])
    dsi_total = np.zeros(2)

    rel_start_dir = ''
    if last_start_pos.size > 0:
        rel_start_dir = find_direction(s0 - last_start_pos)
    # rel_start_pos = np.zeros(2)
    # if last_start_pos.size > 0:
    #     rel_start_pos = last_start_pos - s0
    # rel_start_dir = find_direction(rel_start_pos)

    approx_dist = 0
    micro_directions = ''

    for i, step in enumerate(steps):
        if i == 0: continue
        instruction = step[0]
        data = split_string_to_numbers(step[1])
        if instruction in ('C', 'S'):
            si = data[-2:]
            # dsi = si - s_last
            dsi = si - (s0 + dsi_total)
        elif instruction in ('c', 's'):
            dsi = data[-2:]
            # si = dsi + s_last
        approx_dist += np.linalg.norm(dsi)
        micro_directions += find_direction(dsi)
        # s_last = si
        dsi_total += dsi 

    macro_direction = find_direction(dsi_total)
    micro_directions = cleanup_directions_list(micro_directions)

    return [s0, macro_direction, approx_dist, micro_directions, rel_start_dir]

def get_stroke_info(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot() 
    root_string = ET.tostring(root[0], encoding='unicode')
    root_string = re.sub(' ', ',', root_string)
    root_list = root_string.split('"')
    stroke_paths_info = []
    last_start_pos = np.array([])
    n = 1
    for i, element in enumerate(root_list):
        if 'm' not in element.lower(): continue
        if 'd=' not in root_list[i-1]: continue
        element = adjust_stroke_path_info(element)
        high_level_info = get_high_level_info(element, last_start_pos)
        new_stroke_info = [svg_file, n, element] + high_level_info
        stroke_paths_info.append(new_stroke_info)
        last_start_pos = high_level_info[0]
        n += 1
    return stroke_paths_info

def find_character_class(svg_file):
    if re.search('030[4-9][a-f0-9].svg', svg_file):
        return 'hiragana'
    if re.search('030[a-f][a-f0-9].svg', svg_file):
        return 'katakana'
    return 'kanji'


def find_chars_with_specific_dirs(df_info, df_agg_dir, row, df_s):
    agg_mask = (df_agg_dir.maxStrokeID == row['maxStrokeID']) \
        & (df_agg_dir.strokeDirections == row['strokeDirections']) \
        & (df_agg_dir.positionDirections == row['positionDirections']) \
        & (df_agg_dir.charClass == row['charClass']) 
    files = [fileloc.split('\\')[-1] for fileloc in df_agg_dir[agg_mask].svgFile.values]
    info_mask = df_info.svgFile.isin(files)
    print('\t', df_info[info_mask].kanakanji.values)
    print('\t', df_info[info_mask].unicode.values)

    filelocs = [fileloc for fileloc in df_agg_dir[agg_mask].svgFile.values]
    s_mask = df_s.svgFile.isin(filelocs)
    print('\t', df_s[s_mask].microDirectionsList.values)


def add_new_agg_info_to_kanakanji_info(df_info, df_agg_dir, svg_subfolder):
    df = df_info[['kanakanji','unicode','svgFile']]
    df.loc[:, 'svgFile'] = df.svgFile.apply(lambda r: os.path.join(svg_subfolder, r))
    df_new = pd.merge(df, df_agg_dir, on='svgFile', validate='one_to_one')
    df_new.drop('svgFile', axis=1, inplace=True)

    os.chdir('..\\..\\app')
    new_df_filename = os.path.join(os.getcwd(), 'kanakanji_app_info.csv')
    df_new.to_csv(new_df_filename, header=True, index=False)
    quit()


def main():

    svg_subfolder = 'kanakanji_svg'

    new_df_filename = 'kanakanji_strokePaths.csv'
    display_names = [
        'svgFile', 'strokeID',  
        'startingLocation', 'overallStrokeDirection', 
        'approxTotalDistance', 'microDirectionsList',
        'relativeStartDirection'
    ]
    col_names = display_names[:2] + ['baseStrokePath'] + display_names[2:]

    df_info = pd.read_csv('kanakanji_info.csv')

    # try:
    #     os.remove(new_df_filename)
    # except FileNotFoundError:
    #     pass
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     quit()
    # for i, svg in enumerate(df_info['svgFile'].values):
    #     # if svg not in ['03044.svg', '03063.svg']:  continue
    #     # print(svg)
    #     svg_loc = os.path.join(svg_subfolder, svg)
    #     new_stroke_paths = pd.DataFrame(get_stroke_info(svg_loc), columns=col_names)
    #     if i > 0:
    #         new_stroke_paths.to_csv(new_df_filename, header=False, index=False, mode='a')
    #         continue
    #     new_stroke_paths.to_csv(new_df_filename, header=True, index=False, mode='a')
    # quit()
    
    df_s = pd.read_csv('kanakanji_strokePaths.csv')
    df_s = df_s[display_names]
    total_strokes = df_s[['svgFile','strokeID']].groupby('svgFile').max()
    total_strokes.reset_index(inplace=True)
    total_strokes.rename(columns={'strokeID': 'maxStrokeID'}, inplace=True)
    df = pd.merge(df_s, total_strokes, on='svgFile', how='inner').reindex(df_s.index)
    df.sort_values(by=['maxStrokeID', 'svgFile', 'strokeID'], inplace=True)
    df['relativeStartDirection'] = df['relativeStartDirection'].fillna('')

    df_agg_dir = df[['svgFile', 'maxStrokeID', 'overallStrokeDirection', 'relativeStartDirection']] \
        .groupby(['svgFile', 'maxStrokeID']) \
        .agg(lambda r: ''.join(r))
    df_agg_dir.rename(columns={'overallStrokeDirection': 'strokeDirections'}, inplace=True)
    df_agg_dir.rename(columns={'relativeStartDirection': 'positionDirections'}, inplace=True)
    df_agg_dir.reset_index(inplace=True)
    df_agg_dir['charClass'] = df_agg_dir['svgFile'].apply(lambda r: find_character_class(r))

    add_new_agg_info_to_kanakanji_info(df_info, df_agg_dir, svg_subfolder)
    quit()

    cols = ['maxStrokeID', 'strokeDirections', 'positionDirections', 'charClass'] #['maxStrokeID', 'strokeDirections', 'positionDirections', 'charClass']
    sum_agg_dir = df_agg_dir.groupby(cols).count()
    sum_agg_dir.rename(columns={'svgFile': 'charCount'}, inplace=True)
    sum_agg_dir.reset_index(inplace=True)
    sum_agg_dir.sort_values(by=cols, inplace=True)
    
    print('n', list(sum_agg_dir.columns))
    n = 1
    for index, row in sum_agg_dir.iterrows():
        # if row['strokeDirections'] != 'ttt': continue
        if row['charCount'] > 5:
            print(n, list(row))
            find_chars_with_specific_dirs(df_info, df_agg_dir, row, df_s)
            n += 1


    # print(df[(df.strokeID==1)&(df.overallStrokeDirection=='d')&(df.maxStrokeID==1)])

    # df_s1 = df[df.strokeID == 1][['svgFile', 'overallStrokeDirection']]
    # df_s1.rename(columns={'overallStrokeDirection': 'FirstDirection'}, inplace=True)
    # df_s1.reset_index(inplace=True)
    # df_s2 = df[df.strokeID == 2][['svgFile', 'overallStrokeDirection']]
    # df_s2.rename(columns={'overallStrokeDirection': 'SecondDirection'}, inplace=True)
    # df_s12 = pd.merge(df_s1, df_s2, on='svgFile', how='left').reindex(df_s1.index)
    # df_s3 = df[df.strokeID == 3][['svgFile', 'overallStrokeDirection']]
    # df_s3.rename(columns={'overallStrokeDirection': 'ThirdDirection'}, inplace=True)
    # df_r = pd.merge(df_s12, df_s3, on='svgFile', how='left').reindex(df_s1.index)
    # df_r.fillna('past max?', inplace=True)
    # direction_groups = df_r.groupby(['FirstDirection', 'SecondDirection', 'ThirdDirection']).count()
    # direction_groups.rename(columns={'svgFile': 'characterCounts'}, inplace=True)
    # print(direction_groups)
    # print(direction_groups.characterCounts.sum())
    quit()
    

        




    # only_care_about = ['04e00.svg', '04e14.svg', '04e16.svg', '04e17.svg']
    # only_care_about = ['030cb.svg', '05143.svg', '09801.svg', '09811.svg']
    # only_care_about = ['04e14.svg', '03041.svg', '03042.svg']
    # only_care_about = ['052f5.svg', '05bd3.svg', '05d4e.svg', '07658.svg','0792a.svg','079ba.svg','0842c.svg','07cf2.svg','085d5.svg','08823.svg','09081.svg',]
    # only_care_about = ['03042.svg', '03053.svg', '0307b.svg', '0307e.svg', '0307f.svg']
    # only_care_about = ['050c5.svg']

    

    # for svg in df['svgFile'].values:
    #     if svg not in only_care_about: continue
    #     # print(svg)
    #     find_strokes(os.path.join('kanakanji_svg', svg))
    


    # element_set = set()
    # for svg in df['svgFile'].values:
    #     element_set.update(find_elements(os.path.join('kanakanji_svg', svg)))
    # element_list = list(element_set)
    # element_list.sort()
    # print(len(element_list))
    # for element in element_list:
    #     print('\t', element)



    # m_counts = 0
    # c_counts = 0 
    # s_counts = 0
    # for svg in df['svgFile'].values:
    #     new_counts = svg_path_counts(os.path.join('kanakanji_svg', svg))
    #     m_counts += new_counts[0]
    #     c_counts += new_counts[1]
    #     s_counts += new_counts[2]

    # print(f'm counts: {m_counts}')
    # print(f'c counts: {c_counts}')
    # print(f's counts: {s_counts}')


if __name__ == '__main__':
    main()


# 04e00.svg
# matrix(1 0 0 1 4.25 54.13)

# 04e14.svg
# matrix(1 0 0 1 27.50 28.50)
# matrix(1 0 0 1 36.50 18.50)
# matrix(1 0 0 1 38.50 39.50)
# matrix(1 0 0 1 38.33 61.42)
# matrix(1 0 0 1 13.50 86.50)

# 04e16.svg
# matrix(1 0 0 1 5.25 48.13)
# matrix(1 0 0 1 43.50 18.50)
# matrix(1 0 0 1 67.50 15.50)
# matrix(1 0 0 1 59.50 63.50)
# matrix(1 0 0 1 21.50 23.50)

# 04e17.svg
# matrix(1 0 0 1 5.25 52.88)
# matrix(1 0 0 1 19.50 16.50)
# matrix(1 0 0 1 42.75 16.50)
# matrix(1 0 0 1 70.00 15.40)
# matrix(1 0 0 1 34.50 89.50)