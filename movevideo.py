import os

if __name__ == '__main__':
    path = "D:\\video\\ss31893"  # 要遍历的目录
    for root, dirs, names in os.walk(path):
        print(names)
        for name in names:
            print(name)
            ext = os.path.splitext(name)[1]
            if ext == '.mp4':
                fromdir = os.path.join(root, name)
                moveto = os.path.join("D:\\video\\video", name) 
                os.rename(fromdir, moveto)