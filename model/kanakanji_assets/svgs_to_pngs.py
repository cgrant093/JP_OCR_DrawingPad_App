
import cairosvg
import os
import pandas as pd
from pathlib import Path
from xml.etree import ElementTree as ET


def cleanup_svg_content(svg_file):
    # Parse SVG file using ElementTree
    tree = ET.parse(svg_file)
    root = tree.getroot()  
    # remove stroke number labels    
    del root[1]
    return ET.tostring(root, encoding='utf-8')


def svg_to_png(svg_path):
    """
    Converts an SVG image to a PNG

    Args:
        svg_path (str): Path to the SVG file.

    """
    try:
        png_path = svg_path.replace('svg', 'png')#[:-3] + 'png'
        svg_content = cleanup_svg_content(svg_path)
        cairosvg.svg2png(svg_content, write_to=png_path)
    except Exception as e:
        print(f"An error occurred: {e}", f"for file {svg_path}")


def main():

    Path('kanakanji_png').mkdir(exist_ok=True)

    df = pd.read_csv('kanakanji_info.csv')
    for svg in df['svgFile'].values:
        svg_to_png(os.path.join('kanakanji_svg', svg))


if __name__ == '__main__':
    main()