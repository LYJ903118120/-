
#这是计算图像融合质量，具体公式可看这篇论文http://kns.cnki.net/kcms/detail/11.5602.TP.20180307.1106.002.html
import math
import cv2 as cv
import numpy as np



#信息熵EN
def xinxi_shang(IMG):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    for i in range(len(IMG)):
        for j in range(len(IMG[i])):
            val = IMG[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if (tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res

#空间频率SF
def kongjian_pinlv(IMG):

    i = 0
    RF = 0
    for i in range(len(IMG)):
        x = 1
        for j in range(len(IMG[i])-1):
            a = pow((IMG[i][x] - IMG[i][x - 1]), 2)
            x += 1
            RF += a
    RF = pow(RF / (len(IMG[1]) * len(IMG)), 0.5)
    x = 1
    i = 0
    CF = 0
    for i in range(len(IMG)-1):
        for j in range(len(IMG[i])):
            a = pow((IMG[x][j] - IMG[x - 1][j]), 2)
            CF += a
        x += 1
    CF = pow(CF / (len(IMG[1]) * len(IMG)), 0.5)
    SF = pow(pow(RF, 2) + pow(CF, 2), 0.5)
    return SF


#基于改进的空间频率SF1.0
def kongjian_pinlv1(IMG):

    i = 0
    RF = 0
    for i in range(len(IMG)):
        x = 1
        for j in range(len(IMG[i]) - 1):
            a = pow((IMG[i][x] - IMG[i][x - 1]), 2)
            x += 1
            RF += a
    RF = pow(RF / (len(IMG[1]) * len(IMG)), 0.5)
    x = 1
    i = 0
    CF = 0
    for i in range(len(IMG) - 1):
        for j in range(len(IMG[i])):
            a = pow((IMG[x][j] - IMG[x - 1][j]), 2)
            CF += a
        x += 1
    CF = pow(CF / (len(IMG[1]) * len(IMG)), 0.5)

    MDF = 0
    x = 1
    for i in range(len(IMG)-1):
        y = 1
        for j in range(len(IMG[i])-1):
            a = pow((IMG[x][y] - IMG[x - 1][y - 1]), 2)
            y += 1
            MDF += a
        x += 1
    MDF = pow((1 / pow(2, 0.5)) * (MDF / (len(IMG[1]) * len(IMG))), 0.5)

    SDF = 0
    x=1
    for i in range(len(IMG)-1):
        for j in range(len(IMG[i])-1):
            a = pow((IMG[x][j] - IMG[x - 1][j + 1]), 2)
            SDF += a
        x += 1
    SDF = pow((1 / pow(2, 0.5)) * (SDF / (len(IMG[1]) * len(IMG))), 0.5)
    SF = pow(pow(RF,2)+pow(CF,2)+pow(MDF,2)+pow(SDF,2),0.5)

    return SF

#平均梯度AG
def pinjun_tidu(IMG):
    AG=0

    for i in range(len(IMG)-1):
        for j in range(len(IMG[i])-1):
            a = pow((IMG[i + 1][j] - IMG[i][j]), 2)
            b = pow((IMG[i][j + 1] - IMG[i][j]), 2)
            AG += pow((a+b)*0.5,0.5)
    AG = AG/((len(IMG)-1)*(len(IMG[1])-1))
    return AG

#均值
def junzhi(IMG):

    return np.mean(IMG)

#标准差
def biaozhun_cha(IMG):

    return np.std(IMG,ddof=1)

def ji_he(Img):
    #没有输出均值和标准差
    print("信息熵:", xinxi_shang(Img), "空间频率:", kongjian_pinlv(Img), "基于改进的空间频率:",kongjian_pinlv1(Img), "平均梯度:", pinjun_tidu(Img))


if __name__ == '__main__':

    image = cv.imread("X/3.PNG", 0)#图片路径
    img = np.array(image)          #转成数组
    img = np.int64(img)            #图片默认是uint8，要强制转成int64，不然后面运算会溢出
    ji_he(img)                     #把几个指标集合起来展示
