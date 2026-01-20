import os
import sys

def getval_cfg(path, key):
    with open(path) as f:
        lines = f.readlines()
    lines_strip = [line.strip() for line in lines]
    l_keys = [line for line in lines_strip if key in line]
    if len(l_keys) <= 0:
        print(f'No found {key} in {path}')
        return -1

    val_str = l_keys[0][l_keys[0].find('=')+1:]
    print(f'{key}={val_str}')

    return int(val_str)
