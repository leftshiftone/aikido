import os
from typing import List, Union

"""
Returns a list of images from the given source.
The source can either be a folder or a file.
If the source is a folder then all files of the
folder will be returned. Otherwise the file itself is returned.
"""


def get_files(source: Union[List[str], str], ext: List[str] = []):
    result = []

    if isinstance(source, str):
        source = [source]

    for file in source:
        if os.path.isdir(file):
            images = (file + "/" + file for file in os.listdir(file)
                      if os.path.splitext(file)[1].lower() in ext)
            result.extend(images)
        else:
            result.append(file)

    return result


def get_images(source: Union[List[str]], ext: List[str] = [".jpg", ".jpeg", ".png", ".tiff"]):
    return get_files(source, ext)
