# -*- coding: utf-8 -*-
import random
import copy
import time
import sys
import math
import tkinter
import threading
from functools import reduce

# 参数
'''
ALPHA: 信息启发因子
BETA: 随机性因子
'''
(ALPHA, BETA, RHO, Q) = (1.0, 5.0, 0.5, 100.0)
# 城市数，蚁群
(city_num, ant_num) = (17, 17)
transports = ["公路", "水路", "铁路"]
cities = ["重庆", "宜昌", "常德", "襄樊", "荆门", "荆州", "岳阳", "信阳", "孝感", "武汉", "合肥", "安庆", "九江", "滁州", "芜湖",
          "马鞍山", "南京"]
distance_x = [
    100, 200, 200, 300, 300, 300, 300, 400, 400, 400, 500, 500, 500, 600, 600, 600, 700]
# distance_y = [
#    350, 600, 300, 400, 800, 450, 650, 250, 450, 310, 250, 850, 350, 460, 350]
distance_y = [
    350, 400, 300, 500, 400, 300, 200, 450, 350, 250, 450, 350, 250, 450, 350, 250, 350]
destination = len(distance_x) - 1
# 城市分级
connectivity = {0: [1, 2], 1: [3, 4, 5], 2: [3, 4, 5, 6], 3: [7, 8, 9], 4: [7, 8, 9], 5: [7, 8, 9], 6: [7, 8, 9],
                7: [10, 11, 12], 8: [10, 11, 12], 9: [10, 11, 12], 10: [13, 14, 15], 11: [13, 14, 15], 12: [13, 14, 15],
                13: [16], 14: [16], 15: [16]}
distances = [[[] for i in range(len(distance_x) * 3)] for i in range(len(distance_x) * 3)]
distances[0][1] = [560, 0, 0]
distances[0][2] = [690, 0, 0]
distances[1][3] = [240, 0, 0]
distances[1][4] = [114, 0, 0]
distances[1][5] = [110, 0, 0]
distances[1][6] = [268, 0, 0]
distances[2][3] = [379, 0, 0]
distances[2][4] = [245, 0, 0]
distances[2][5] = [170, 0, 0]
distances[2][6] = [178, 0, 0]
distances[3][7] = [228, 0, 0]
distances[3][8] = [372, 0, 0]
distances[3][9] = [421, 0, 0]
distances[4][7] = [275, 0, 0]
distances[4][8] = [182, 0, 0]
distances[4][9] = [224, 0, 0]
distances[5][7] = [315, 0, 0]
distances[5][8] = [204, 0, 0]
distances[5][9] = [218, 0, 0]
distances[6][7] = [362, 0, 0]
distances[6][8] = [236, 0, 0]
distances[6][9] = [196, 0, 0]
distances[7][10] = [327, 0, 0]
distances[7][11] = [425, 0, 0]
distances[7][12] = [422, 2, 0]
distances[8][10] = [392, 0, 0]
distances[8][11] = [362, 5, 0]
distances[8][12] = [316, 0, 0]
distances[9][10] = [371, 0, 0]
distances[9][11] = [321, 0, 0]
distances[9][12] = [258, 0, 0]
distances[10][13] = [119, 66, 0]
distances[10][14] = [132, 0, 0]
distances[10][15] = [145, 0, 0]
distances[11][13] = [272, 0, 0]
distances[11][14] = [182, 0, 0]
distances[11][15] = [232, 0, 0]
distances[12][13] = [454, 0, 0]
distances[12][14] = [352, 0, 0]
distances[12][15] = [400, 0, 0]
distances[13][16] = [64, 0, 0]
distances[14][16] = [97, 0, 0]
distances[15][16] = [52, 0, 0]

# 城市距离和信息素
distance_graph = [[0.0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph = [[1.0 for col in range(city_num)] for raw in range(city_num)]


# ----------- 蚂蚁 -----------
class Ant(object):

    # 初始化
    def __init__(self, ID):

        self.ID = ID  # ID
        self.__clean_data()  # 随机初始化出生点

    # 初始数据
    def __clean_data(self):

        self.path = []  # 当前蚂蚁的路径
        self.trans = []  # 运输方式
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态

        city_index = 0  # 初始出生点0
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city_and_transport(self):

        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in connectivity[self.current_city]:
            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / min(filter(lambda x: x > 0, distances[self.current_city][i]))), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID=self.ID,
                                                                                                current=self.current_city,
                                                                                                target=i))
                    sys.exit(1)

        # 轮盘选择城市及运输方式
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in connectivity[self.current_city]:
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break

        if (next_city == -1):
            next_city = random.randint(0, city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)

        transport = -1
        while transport == -1 or distances[self.current_city][next_city][transport] == 0:
            transport = random.choice([0, 1, 2])

        # 返回下一个城市序号
        return next_city, transport

    # 计算路径总距离
    def __cal_total_distance(self):

        temp_distance = 0.0

        for i in range(1, len(self.path)):
            start, end = self.path[i - 1], self.path[i]
            temp_distance += distances[start][end][self.trans[i - 1]]

        # # 回路
        # end = self.path[0]
        # temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city, type_transport):

        self.path.append(next_city)
        self.trans.append(type_transport)
        self.open_table_city[next_city] = False
        self.total_distance += distances[self.current_city][next_city][type_transport]
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self):

        # 初始化数据
        self.__clean_data()

        # 搜素路径，到达终点为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city, type_transport = self.__choice_next_city_and_transport()

            self.__move(next_city, type_transport)
            if destination in self.path:
                break

        # 计算路径总长度
        self.__cal_total_distance()


# ----------- TSP问题 -----------

class TSP(object):

    def __init__(self, root, width=800, height=600, n=city_num):

        # 创建画布
        self.root = root
        self.width = width
        self.height = height
        # 城市数目初始化为city_num
        self.n = n
        # tkinter.Canvas
        self.canvas = tkinter.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#EBEBEB",  # 背景白色
            xscrollincrement=1,
            yscrollincrement=1
        )
        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.title("TSP蚁群算法(n:初始化 e:开始搜索 s:停止搜索 q:退出程序)")
        self.__r = 5
        self.__lock = threading.RLock()  # 线程锁

        self.__bindEvents()
        self.new()

        # 城市间公路距离
        # for i in range(city_num):
        #     for j in range(city_num):
        #         temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
        #         temp_distance = pow(temp_distance, 0.5)
        #         distance_graph[i][j] = float(int(temp_distance + 0.5))

    # 按键响应程序
    def __bindEvents(self):

        self.root.bind("q", self.quite)  # 退出程序
        self.root.bind("n", self.new)  # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)  # 停止搜索

    # 更改标题
    def title(self, s):

        self.root.title(s)

    # 初始化
    def new(self, evt=None):

        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        self.clear()  # 清除信息
        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点对象

        # 初始化城市节点
        for i in range(len(distance_x)):
            # 在画布上随机初始坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x, y))
            # 生成节点椭圆，半径为self.__r
            node = self.canvas.create_oval(x - self.__r,
                                           y - self.__r, x + self.__r, y + self.__r,
                                           fill="#ff0000",  # 填充红色
                                           outline="#000000",  # 轮廓白色
                                           tags="node",
                                           )
            self.nodes2.append(node)
            # 显示坐标
            self.canvas.create_text(x, y - 10,  # 使用create_text方法在坐标（302，77）处绘制文字
                                    text=str(i) + ' ' + str(cities[i]),  # 所绘制文字的内容
                                    fill='black'  # 所绘制文字的颜色为灰色
                                    )

        # 顺序连接城市
        # self.line(range(city_num))

        # 初始城市之间的距离和信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 1.0

        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)  # 初始最优解
        self.best_ant.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

    # 将节点按order顺序连线
    def line(self, order):
        # 删除原线
        self.canvas.delete("line")

        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill="#000000", tags="line")
            return i2

        # order[-1]为初始值
        reduce(line2, order)

    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    # 退出程序
    def quite(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()
        print(u"\n程序已退出...")
        sys.exit()

    # 停止搜索
    def stop(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

    # 开始搜索
    def search_path(self, evt=None):

        # 开启线程
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()

        while self.__running:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path()
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            # 更新信息素
            self.__update_pheromone_gragh()
            path_print = ""
            for i in range(len(self.best_ant.path)):
                if i != len(self.best_ant.path) - 1:
                    path_print += str(cities[self.best_ant.path[i]]) + "-" + str(transports[self.best_ant.trans[i]]) + "->"
                else:
                    path_print += str(cities[self.best_ant.path[i]])
            print(path_print)
            print(u"迭代次数：", self.iter, u"最佳路径总距离：", int(self.best_ant.total_distance))
            # 连线
            self.line(self.best_ant.path)
            # 设置标题
            self.title("LXR多式联运ACO(n:随机初始 e:开始搜索 s:停止搜索 q:退出程序) 迭代次数: %d" % self.iter)
            # 更新画布
            self.canvas.update()
            self.iter += 1

    # 更新信息素
    def __update_pheromone_gragh(self):

        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in self.ants:
            for i in range(1, len(ant.path)):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]

    # 主循环
    def mainloop(self):
        self.root.mainloop()


# ----------- 程序的入口处 -----------

if __name__ == '__main__':
    print(u""" 
--------------------------------------------------------
    程序：冲冲冲
-------------------------------------------------------- 
    """)
    TSP(tkinter.Tk()).mainloop()
