import os
import shutil

if __name__ == '__main__':
    namelist = [7, 31, 45]
    r_list = os.listdir(r'D:\去云对比方法\WLR\kpl\1121result\result')
    s_path = r'D:\cloudromove\Xian\256\20220226'
    # sm_path = r'D:\cloudromove\dataset\test\5'
    t_path = r'C:\Users\53110\Desktop\mountain\20220226'
    # m_path = r'C:\Users\53110\Desktop\农田\5'
    for index in namelist:
        for name in range(index, 192, 48):
            name = str(name) + '.tif'
            if not os.path.exists(t_path):
                # os.mkdir(m_path)
                os.mkdir(t_path)
            # for name in r_list:
            # source = os.path.join(sm_path, name[7:-4] + '.bmp')
            # target = os.path.join(m_path, name[7:-4] + '.bmp')
            # shutil.copy(source, target)
            st = os.path.join(s_path, name)
            tt = os.path.join(t_path, name)
            shutil.copy(st, tt)
