

The files in this folder are to store and manipulate all of the Japanese character assets (kana and kanji). 

<br/>

# Current Progress

## A. Create kanakanji_info.csv
I wanted to store a few pieces of information into one csv that can be referenced from various files.

I downloaded the SVG files from the [KanjiVG](https://github.com/KanjiVG/kanjivg/tree/master) project. And I downloaded (with some manipulation) kanakanji_unicode.txt from the [rikai](http://www.rikai.com/library/kanjitables/kanji_codes.unicode.shtml) website.

The make_kanakanji_info_csv.py file runs through the following process:

1. Reads in the kana-kanji unicode text file, and manipulates its information to create a pandas DataFrame that has two columns: the copyable text character, and the unicode for said character.
2. Each SVG file is named after the unicode for the character, but with some minor manipulation. The next step is to create a column that mimicks the SVG file name for the character (which does already exist). Then create a column that mimicks the PNG file name for the character (which does not yet exists).
3. The advantage of the SVG files is that we can also find how many strokes it takes to create each character. The penultimate step is to use the xml minidom package to create a column that displays the stroke count for each character.
4. Then save the pandas DataFrame as kanakanji_info.csv which has the following columns:
    1. kanakanji    - the digital text character version of the kana/kanji
    2. unicode      - unicode for the character
    3. svgFile      - the current SVG filename for the character
    4. pngFile      - the future PNG filename for the character
    5. strokeCount  - number of writing strokes used to create the character


## B. Converted the SVG image assets to PNG
To properly use the image files as training data for the PyTorch models, I need to first convert all kana-kanji asset files from SVG to PNG. We can utilize the kanakanji_info.csv and the cairosvg package to make this happen.

Read in the kanakanji_info.csv as a pandas DataFrame, and iterate through each file in the svgFile. Before using cairosvg, I need "clean-up" the SVG files. Each SVG has little number labels next to each stroke. I utilzed the xml ElementTree package to remove the number labels not from the SVG files, but the SVG context before converting to PNG. Use cairosvg's svg2png function to convert to PNG.

<br/>

# Future Work

I do not know if there will be future work for this folder. We may need to add something to the kana-kanji info CSV. Additionally, I am not sure whether this folder will be the output for the PyTorch models or if I'll use a different folder for that.
