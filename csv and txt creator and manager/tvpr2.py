import os
import shutil

path = "C:/Users/Daniele/Desktop/TVPR2/test/"
path_depth = "C:/Users/Daniele/Desktop/TVPR2/npy-test/"

'''for item in os.listdir(path):
    new_name = item.strip(".jpg") + "_depth.jpg"
    os.rename(path+item, path+new_name)'''

'''for i in range(1027):
    path1 = os.path.join(path, str(i))
    os.mkdir(path1)'''

files = os.listdir(path_depth)
'''for i in range(1027):
    files.remove(str(i))'''

for item in files:
    for i in range(4,-1,-1):
        if item[6:6 + i].isnumeric():
            x = int(item[6:6+i]) -1
            break
    shutil.move(path_depth+item, path+str(x))