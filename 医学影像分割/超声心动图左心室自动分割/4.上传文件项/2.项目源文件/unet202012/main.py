from level1 import level1
from level2 import level2
from level3 import level3


def entrance(level = 'level3'):
    if level == 'level1':
        level1()
    elif level == 'level2':
        level2()
    elif level == 'level3':
        level3()

if __name__ == '__main__':
    entrance('level3')