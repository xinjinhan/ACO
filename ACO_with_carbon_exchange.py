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
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
     加快，但是随机性不高，容易得到局部的相对最优
'''
# 最大迭代次数
max_iteration = 1500
# ACO参数
(ALPHA, BETA, RHO, Q) = (10.0, 1.0, 0.5, 10.0)
# 单位人工碳排放数
one_man_carbon = 13.55
# 服务人工数
server_man_number = 10
# 碳排放额度上限
carbon_max = 40000
# 碳税
carbon_tax = 50
# 运输碳排放系数
carbon_transports = [0.1716, 0.0154, 0.0331]
# 转运碳排放系数
carbon_change = [4.86, 6.42, 8.24]
# 碳交易与碳补尝
carbon_remedy_amount = 20
# 发车时间
start_time = 16.0
# 城市数，蚁群
(city_num, ant_num) = (22, 50)
# 运输汽车数量
num_of_cars = 229
# 可选运输方式
transports = ["公路", "水路", "铁路"]
# 运输速度
transports_speed = [70, 18, 120]
# 安全成本系数
safety_cost = [0.042, 0.018, 0.022]
# 到达时间窗
delivering_time_window = [0, 100]
# 城市
cities = ["重庆主机厂", "团结村", "果园港", "唐家沱", "宜昌", "常德", "襄樊", "荆门", "荆州", "岳阳", "信阳", "孝感", "武汉", "合肥", "安庆", "九江", "滁州", "芜湖",
          "马鞍山", "中华门", "南京港", "南京各经销商"]
# 转换铁路运输成本
change_cost_time_railway = [0.06, 0.05, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.03, 0.06, 0.06, 0.04, 0.06, 0.05,
                            0.06, 0.06, 0.06, 0.06]
# 转换船运输成本
change_cost_time_ship = [0.1, 0.06, 0.09, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.1, 0.09, 0.08, 0.1, 0.1, 0.09, 0.1, 0.09, 0.1]
# 坐标（绘图）
distance_x = [
    75, 150, 150, 150, 200, 200, 300, 300, 300, 300, 400, 400, 400, 500, 500, 500, 600, 600, 600, 675, 675, 752]
distance_y = [
    350, 275, 350, 425, 300, 400, 200, 300, 400, 500, 250, 350, 450, 250, 350, 450, 250, 350, 450, 325, 375, 350]
destination = len(distance_x) - 1
# 运输成本（距离影响）
cost_low = [1.42, 0.77, 1.13]
cost_middle = [1.21, 0.58, 0.96]
cost_high = [1.07, 0.46, 0.67]
# 运输距离，0表示不通
distances = [[[] for i in range(len(distance_x) * 3)] for i in range(len(distance_x) * 3)]
distances[0][1] = [41.9, 0, 0]
distances[0][2] = [36.9, 0, 0]
distances[0][3] = [24.3, 0, 0]
distances[0][4] = [560, 0, 0]
distances[0][5] = [690, 0, 0]
distances[1][4] = [0, 0, 443]
distances[1][5] = [0, 0, 626]
distances[2][4] = [0, 630, 455]
distances[2][5] = [0, 630, 638]
distances[3][4] = [0, 648, 473]
distances[3][5] = [0, 648, 656]
distances[4][6] = [246, 0, 141]
distances[4][7] = [114, 0, 215]
distances[4][8] = [110, 148, 432]
distances[4][9] = [268, 395, 211]
distances[5][6] = [371, 0, 239]
distances[5][7] = [245, 0, 314]
distances[5][8] = [170, 0, 239]
distances[5][9] = [178, 0, 310]
distances[6][10] = [235, 0, 206]
distances[6][11] = [231, 0, 285]
distances[6][12] = [293, 0, 262]
distances[7][10] = [275, 0, 276]
distances[7][11] = [182, 0, 355]
distances[7][12] = [224, 0, 226]
distances[8][10] = [315, 0, 358]
distances[8][11] = [204, 0, 437]
distances[8][12] = [218, 478, 308]
distances[9][10] = [362, 0, 325]
distances[9][11] = [236, 0, 246]
distances[9][12] = [196, 231, 221]
distances[10][13] = [327, 0, 267]
distances[10][14] = [425, 0, 351]
distances[10][15] = [422, 0, 309]
distances[11][13] = [392, 0, 346]
distances[11][14] = [362, 0, 340]
distances[11][15] = [316, 0, 239]
distances[12][13] = [371, 0, 321]
distances[12][14] = [321, 433, 314]
distances[12][15] = [258, 269, 213]
distances[13][16] = [119, 0, 221]
distances[13][17] = [132, 0, 179]
distances[13][18] = [145, 0, 189]
distances[14][16] = [272, 0, 306]
distances[14][17] = [185, 204, 264]
distances[14][18] = [232, 252, 273]
distances[15][16] = [454, 0, 383]
distances[15][17] = [352, 368, 236]
distances[15][18] = [400, 416, 260]
distances[16][19] = [64, 0, 108]
distances[17][19] = [97, 0, 141]
distances[17][20] = [0, 96, 0]
distances[18][19] = [52, 0, 116]
distances[18][20] = [0, 48, 0]
distances[19][21] = [18, 0, 0]
distances[20][21] = [50, 0, 0]

# 城市分级
connectivity = {}
for i in range(city_num):
    tmp = []
    for j in range(i + 1, city_num):
        if j == len(distances):
            break
        if distances[i][j] != []:
            tmp.append(j)
            connectivity[i] = tmp
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
        self.time_sequence = [start_time]  # 运输时间节点
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态
        city_index = 0  # 初始出生点0
        self.current_city = city_index  # 当前所在城市
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.total_cost = 0  # 总花费资金
        self.total_time = start_time  # 总花费时间
        self.total_change_cost = 0  # 总转运花费资金
        self.total_connecting_cost = 0  # 总衔接成本
        self.total_carbon_cost = 0  # 总碳排放成本
        self.total_change_time = 0  # 总转运时间
        self.total_transport_cost = 0  # 总运输花费资金
        self.total_transport_time = 0  # 总运输时间
        self.total_safety_cost = 0  # 总安全花费资金
        self.total_punishment_cost = 0  # 总惩罚成本
        self.total_carbon = 0  # 总碳排放量
        self.total_transport_carbon = 0  # 运输碳排放量
        self.total_transport_change_carbon = 0  # 转换运输碳排放
        self.total_man_carbon = 0  # 人力总碳排放

    # 计算距离对应陈本
    def __cal_cost(self, distance, transport):
        cost = 0
        if distance / 500 < 1:
            cost = distance * cost_low[transport] * num_of_cars
        elif distance / 500 <= 2:
            cost = distance * cost_middle[transport] * num_of_cars
        elif distance / 500 > 2:
            cost = distance * cost_high[transport] * num_of_cars
        return int(cost)

    # 选择下一个城市
    def __choice_next_city_and_transport_only_ship(self):

        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in connectivity[self.current_city]:
            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    possible_choice_of_path_with_transport = [0, 0, 0]
                    # 不同方式间，转运成本不同
                    for j in [1]:
                        if distances[self.current_city][i][j] == 0:
                            continue
                        # 考虑不同运输时间窗
                        # if j == 1:
                        #     if self.total_time not in [31, 32, 33, 34, 77, 78, 79, 80]:
                        #         continue
                        if j == 2:
                            if self.total_time not in [6, 7, 8, 54, 55, 56, 102, 103, 104]:
                                continue
                        if len(self.trans) != 0 and j != self.trans[-1]:
                            change_cost = 0
                            if j in [0, 1] and self.trans[-1] in [0, 1]:
                                change_cost = 245
                            elif j in [0, 2] and self.trans[-1] in [0, 2]:
                                change_cost = 80
                            elif j in [1, 2] and self.trans[-1] in [1, 2]:
                                change_cost = 182
                            possible_choice_of_path_with_transport[j] = change_cost * num_of_cars
                        possible_choice_of_path_with_transport[j] += \
                            self.__cal_cost(distances[self.current_city][i][j], j)
                    if possible_choice_of_path_with_transport[0] == 0 and possible_choice_of_path_with_transport[1] == 0 \
                            and possible_choice_of_path_with_transport[2] == 0:
                        select_citys_prob[i] = 0
                        total_prob += select_citys_prob[i]
                        continue

                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / min(filter(lambda x: x > 0, possible_choice_of_path_with_transport))), BETA)
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

        while next_city == -1 and select_citys_prob[next_city] == 0:
            next_city = random.randint(0, city_num - 1)
            while not (self.open_table_city[next_city]):  # if==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)

        transport = 1
        while distances[self.current_city][next_city][transport] == 0:
            transport = random.choice([0, 1, 2])

        # 返回下一个城市序号
        return next_city, transport

    # 选择下一个城市
    def __choice_next_city_and_transport(self):

        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in connectivity[self.current_city]:
            connecting_cost = [0, 0, 0]
            # 考虑不同运输时间窗
            if self.total_time < 31:
                connecting_cost[1] = math.ceil(31 - self.total_time) * 7
            elif 34 < self.total_time < 77:
                connecting_cost[1] = math.ceil(77 - self.total_time) * 7
            elif self.total_time > 80:
                connecting_cost[1] = math.ceil(7 * 24 - self.total_time + 31) * 7

            if self.total_time < 6:
                connecting_cost[2] = math.ceil(6 - self.total_time) * 7
            elif 8 < self.total_time < 54:
                connecting_cost[2] = math.ceil(54 - self.total_time) * 7
            elif 56 < self.total_time < 102:
                connecting_cost[2] = math.ceil(102 - self.total_time) * 7
            elif self.total_time > 104:
                connecting_cost[2] = math.ceil(7 * 24 - self.total_time + 6) * 7

            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与成本成反比
                    possible_choice_of_path_with_transport = [0, 0, 0]
                    # 不同方式间，转运成本不同
                    for j in [0, 1, 2]:
                        if distances[self.current_city][i][j] == 0:
                            continue
                        if len(self.trans) != 0 and j != self.trans[-1]:
                            change_cost = 0
                            if j in [0, 1] and self.trans[-1] in [0, 1]:
                                change_cost = 245
                            elif j in [0, 2] and self.trans[-1] in [0, 2]:
                                change_cost = 80
                            elif j in [1, 2] and self.trans[-1] in [1, 2]:
                                change_cost = 182
                            possible_choice_of_path_with_transport[j] = change_cost * num_of_cars
                            possible_choice_of_path_with_transport[j] += connecting_cost[j] * num_of_cars
                        possible_choice_of_path_with_transport[j] += \
                            self.__cal_cost(distances[self.current_city][i][j], j)
                    if possible_choice_of_path_with_transport[0] == 0 and possible_choice_of_path_with_transport[1] == 0 \
                            and possible_choice_of_path_with_transport[2] == 0:
                        select_citys_prob[i] = 0
                        total_prob += select_citys_prob[i]
                        continue

                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / min(filter(lambda x: x > 0, possible_choice_of_path_with_transport))), BETA)
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

        while next_city == -1 and select_citys_prob[next_city] == 0:
            next_city = random.randint(0, city_num - 1)
            while not (self.open_table_city[next_city]):  # if==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)

        transport = random.choice([0, 1, 2])
        while distances[self.current_city][next_city][transport] == 0:
            transport = random.choice([0, 1, 2])

        # 返回下一个城市序号
        return next_city, transport

    # 计算路径总距离
    def __cal_total_distance(self):

        temp_distance = 0.0

        for i in range(1, len(self.path)):
            start, end = self.path[i - 1], self.path[i]
            temp_distance += distances[start][end][self.trans[i - 1]]

        if self.total_time > delivering_time_window[1]:
            self.total_punishment_cost += (self.total_time - delivering_time_window[1]) * 1000 * num_of_cars
            self.total_cost += self.total_punishment_cost
        elif self.total_time < delivering_time_window[0]:
            self.total_punishment_cost += (delivering_time_window[0] - self.total_time) * 100 * num_of_cars
            self.total_cost += self.total_punishment_cost
        # # 回路
        # end = self.path[0]
        # temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city, type_transport):
        change_cost = 0
        change_carbon = 0
        change_cost_time = 0
        current_distance = distances[self.current_city][next_city][type_transport]

        connecting_cost = [0, 0, 0]
        if type_transport == 1:
            if self.total_time < 31:
                connecting_cost[type_transport] = math.ceil(31 - self.total_time) * 7
            elif 34 < self.total_time < 77:
                connecting_cost[type_transport] = math.ceil(77 - self.total_time) * 7
            elif self.total_time > 80:
                connecting_cost[type_transport] = math.ceil(7 * 24 - self.total_time + 31) * 7
        elif type_transport == 2:
            if self.total_time < 6:
                connecting_cost[type_transport] = math.ceil(6 - self.total_time) * 7
            elif 8 < self.total_time < 54:
                connecting_cost[type_transport] = math.ceil(54 - self.total_time) * 7
            elif 56 < self.total_time < 102:
                connecting_cost[type_transport] = math.ceil(102 - self.total_time) * 7
            elif self.total_time > 104:
                connecting_cost[type_transport] = math.ceil(7 * 24 - self.total_time + 6) * 7

        if len(self.trans) != 0 and type_transport != self.trans[-1]:
            if type_transport in [0, 1] and self.trans[-1] in [0, 1]:
                change_cost = 182
                change_carbon = carbon_transports[type_transport]
                change_cost_time = change_cost_time_ship[next_city]
            elif type_transport in [0, 2] and self.trans[-1] in [0, 2]:
                change_cost = 80
                change_carbon = carbon_transports[type_transport]
                change_cost_time = change_cost_time_railway[next_city]
            elif type_transport in [1, 2] and self.trans[-1] in [1, 2]:
                change_cost = 245
                change_carbon = carbon_transports[type_transport]
                change_cost_time = change_cost_time_ship[next_city]
        # 当前转换运输方式时间成本
        self.total_time += change_cost_time * num_of_cars
        self.total_change_time += change_cost_time * num_of_cars
        # 当前运输时间成本
        self.total_time += current_distance / transports_speed[type_transport]
        self.total_transport_time += current_distance / transports_speed[type_transport]
        # 当前转换运输方式成本
        self.total_cost += change_cost * num_of_cars
        self.total_change_cost += change_cost * num_of_cars
        # 当前运输成本
        self.total_cost += self.__cal_cost(current_distance, type_transport)
        self.total_transport_cost += self.__cal_cost(current_distance, type_transport)
        # 当前安全成本
        self.total_cost += safety_cost[type_transport] * num_of_cars * current_distance
        self.total_safety_cost += safety_cost[type_transport] * num_of_cars * current_distance
        # 当前衔接等待成本
        self.total_cost += connecting_cost[type_transport] * num_of_cars
        self.total_connecting_cost += connecting_cost[type_transport] * num_of_cars
        # 当前运输碳排放
        self.total_carbon += current_distance * carbon_transports[type_transport] * num_of_cars
        self.total_transport_carbon += current_distance * carbon_transports[type_transport] * num_of_cars
        # 当前转运碳排放
        self.total_carbon += change_carbon * num_of_cars
        self.total_transport_change_carbon += change_carbon * num_of_cars
        self.path.append(next_city)
        self.time_sequence.append(self.total_time)
        self.trans.append(type_transport)
        self.open_table_city[next_city] = False
        self.total_distance += current_distance
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
        self.time_sequence.append(self.total_time)
        # 加入人力碳排放
        self.total_carbon += self.total_time * one_man_carbon * server_man_number
        self.total_man_carbon += self.total_time * one_man_carbon * server_man_number
        # 加入碳排放成本
        self.total_cost -= (carbon_max - self.total_carbon) * carbon_remedy_amount
        self.total_carbon_cost = (carbon_max - self.total_carbon) * carbon_remedy_amount


# ----------- TSP问题 -----------

class TSP(object):

    def __init__(self, root, width=800, height=600, n=city_num):

        # 创建画布
        self.max_iter = max_iteration / 20
        self.total_sampling_times = 20
        self.current_sampling_times = 0
        self.best_ant_after_rerunning = Ant(-1)  # 初始多重实验后的最优解
        self.best_ant_after_rerunning.total_cost = 1 << 31
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
        self.title("ACO-多式联运-碳交易  (n:初始化 e:开始搜索 s:停止搜索 q:退出程序)")
        self.__r = 5
        self.__lock = threading.RLock()  # 线程锁

        self.__bindEvents()
        self.new()

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
    def new_with_thread_alive(self, evt=None):
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
        self.best_ant.total_cost = 1 << 31  # 初始最大成本
        self.iter = 1  # 初始化迭代次数

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
        self.best_ant.total_cost = 1 << 31  # 初始最大成本
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
            while self.current_sampling_times != self.total_sampling_times:
                # 遍历每一只蚂蚁
                for ant in self.ants:
                    # 搜索一条路径
                    ant.search_path()
                    # 与当前最优蚂蚁比较
                    if ant.total_cost < self.best_ant.total_cost:
                        # 更新最优解
                        self.best_ant = copy.deepcopy(ant)
                # 更新信息素
                self.__update_pheromone_gragh()
                path_print = ""
                for i in range(len(self.best_ant.path)):
                    if i != len(self.best_ant.path) - 1:
                        path_print += str(cities[self.best_ant.path[i]]) + "(到达时间：" \
                                      + str(round(self.best_ant.time_sequence[i], 1)) \
                                      + ")-" + str(transports[self.best_ant.trans[i]]) + "->"
                    else:
                        path_print += str(cities[self.best_ant.path[i]])
                path_print += "（到达时间：{}）".format(str(round(self.best_ant.time_sequence[-1], 1)))
                print(path_print)

                result_print = " 迭代次数：{}\n" \
                               " 最佳路径总距离：{}\n" \
                               " 总资金成本：{} CNY\n" \
                               " 总转运资金成本：{} CNY\n" \
                               " 总运输（路上）资金成本：{} CNY\n" \
                               " 总安全资金成本：{} CNY\n" \
                               " 总衔接成本：{} CNY\n" \
                               " 总惩罚成本：{} CNY\n" \
                               " 总碳交易收益：{} CNY\n" \
                               " 总时间成本：{} h\n" \
                               " 总转运时间成本：{} h\n" \
                               " 总运输（路上）时间成本:{} h\n" \
                               " 总碳排放:{} kg \n" \
                               " 总运输碳排放:{} kg \n" \
                               " 总转运碳排放:{} kg \n" \
                               " 总人力碳排放:{} kg \n"\
                    .format((self.iter + self.max_iter * self.current_sampling_times),
                            str(int(self.best_ant.total_distance)),
                            str(int(self.best_ant.total_cost)),
                            str(int(self.best_ant.total_change_cost)),
                            str(int(self.best_ant.total_transport_cost)),
                            str(int(self.best_ant.total_safety_cost)),
                            str(int(self.best_ant.total_connecting_cost)),
                            str(int(self.best_ant.total_punishment_cost)),
                            str(int(self.best_ant.total_carbon_cost)),
                            str(round(self.best_ant.total_time  - start_time, 2)),
                            str(round(self.best_ant.total_change_time, 2)),
                            str(round(self.best_ant.total_transport_time, 2)),
                            str(round(self.best_ant.total_carbon, 2)),
                            str(round(self.best_ant.total_transport_carbon, 2)),
                            str(round(self.best_ant.total_transport_change_carbon, 2)),
                            str(round(self.best_ant.total_man_carbon, 2)))
                print(result_print)
                # 连线
                self.line(self.best_ant.path)
                # 设置标题
                self.title("ACO-多式联运-碳交易  (n:随机初始 e:开始搜索 s:停止搜索 q:退出程序) 迭代次数: %d"
                           % (self.iter + self.max_iter * self.current_sampling_times))
                # 更新画布
                self.canvas.update()
                self.iter += 1
                if self.iter > self.max_iter:
                    if self.best_ant_after_rerunning.total_cost > self.best_ant.total_cost:
                        self.best_ant_after_rerunning = copy.deepcopy(self.best_ant)
                    self.new(evt)
                    self.current_sampling_times += 1

        # 打印最终最优方案
        print("=======交易方案=======")
        path_print = ""
        for i in range(len(self.best_ant_after_rerunning.path)):
            if i != len(self.best_ant_after_rerunning.path) - 1:
                path_print += str(cities[self.best_ant_after_rerunning.path[i]]) + "(到达时间：" \
                              + str(round(self.best_ant_after_rerunning.time_sequence[i], 2)) \
                              + ")-" + str(transports[self.best_ant_after_rerunning.trans[i]]) + "->"
            else:
                path_print += str(cities[self.best_ant_after_rerunning.path[i]])
        path_print += "（到达时间：{}）".format(str(round(self.best_ant_after_rerunning.time_sequence[-1], 2)))
        print(path_print)

        path_print = ""
        for i in range(len(self.best_ant_after_rerunning.path)):
            if i != len(self.best_ant_after_rerunning.path) - 1:
                path_print += str(self.best_ant_after_rerunning.path[i]) + "(到达时间：" \
                              + str(round(self.best_ant_after_rerunning.time_sequence[i], 2)) \
                              + ")>" + str(transports[self.best_ant_after_rerunning.trans[i]]) + ">"
            else:
                path_print += str(self.best_ant_after_rerunning.path[i])
        path_print += "（到达时间：{}）".format(str(round(self.best_ant_after_rerunning.time_sequence[-1], 2)))
        print(path_print)

        path_print = ""
        for i in range(len(self.best_ant_after_rerunning.path)):
            if i != len(self.best_ant_after_rerunning.path) - 1:
                path_print += str(self.best_ant_after_rerunning.path[i]) +">"\
                              + str(transports[self.best_ant_after_rerunning.trans[i]]) + ">"
            else:
                path_print += str(self.best_ant_after_rerunning.path[i])
        print(path_print)

        result_print = " 迭代次数：{}\n" \
                       " 最佳路径总距离：{}\n" \
                       " 总资金成本：{} CNY\n" \
                       " 总转运资金成本：{} CNY\n" \
                       " 总运输（路上）资金成本：{} CNY\n" \
                       " 总安全资金成本：{} CNY\n" \
                       " 总衔接成本：{} CNY\n" \
                       " 总惩罚成本：{} CNY\n" \
                       " 总碳交易收益：{} CNY\n" \
                       " 总时间成本：{} h\n" \
                       " 总转运时间成本：{} h\n" \
                       " 总运输（路上）时间成本:{} h\n" \
                       " 总碳排放:{} kg \n" \
                       " 总运输碳排放:{} kg \n" \
                       " 总转运碳排放:{} kg \n" \
                       " 总人力碳排放:{} kg \n" \
            .format((self.iter + self.max_iter * self.current_sampling_times - 1),
                    str(int(self.best_ant_after_rerunning.total_distance)),
                    str(int(self.best_ant_after_rerunning.total_cost)),
                    str(int(self.best_ant_after_rerunning.total_change_cost)),
                    str(int(self.best_ant_after_rerunning.total_transport_cost)),
                    str(int(self.best_ant_after_rerunning.total_safety_cost)),
                    str(int(self.best_ant_after_rerunning.total_connecting_cost)),
                    str(int(self.best_ant_after_rerunning.total_punishment_cost)),
                    str(int(self.best_ant_after_rerunning.total_carbon_cost)),
                    str(round(self.best_ant_after_rerunning.total_time  - start_time, 2)),
                    str(round(self.best_ant_after_rerunning.total_change_time, 2)),
                    str(round(self.best_ant_after_rerunning.total_transport_time, 2)),
                    str(round(self.best_ant_after_rerunning.total_carbon, 2)),
                    str(round(self.best_ant_after_rerunning.total_transport_carbon, 2)),
                    str(round(self.best_ant_after_rerunning.total_transport_change_carbon, 2)),
                    str(round(self.best_ant_after_rerunning.total_man_carbon, 2)))
        print(result_print)
        self.line(self.best_ant_after_rerunning.path)
        self.stop(evt)

    # 更新信息素
    def __update_pheromone_gragh(self):

        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in self.ants:
            for i in range(1, len(ant.path)):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与总成本成反比
                temp_pheromone[start][end] += Q / ant.total_cost
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
    程序：ACO-多式联运-碳补偿/交易
-------------------------------------------------------- 
    """)
    TSP(tkinter.Tk()).mainloop()
