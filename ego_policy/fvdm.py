import time
import traci
import numpy as np
import random
from sumolib import checkBinary
import optparse
import sys


class fvdm_model(object):
    def __init__(self, carID, dt=0.04):
        self.carID = carID
        self.buslen = 10
        self.carlen = 5
        self.minGap = 1.5  # 前后车最小间距，相当于文章中的db-carlen
        self.follow_distance = 50
        self.lanechange_time = 5
        self.p = 0  # 驾驶员反应时间
        self.dt = dt
        self.type = traci.vehicle.getTypeID(self.carID)  # 车辆的类型
        self.vmax = traci.vehicle.getMaxSpeed(self.carID)  # 最大速度
        self.maxacc = traci.vehicle.getAccel(self.carID)  # 最大加速度
        self.maxdec = traci.vehicle.getDecel(self.carID)  # 最大减速度
        self.length = traci.vehicle.getLength(self.carID)  # 返回车辆的长度

        self.speed = traci.vehicle.getSpeed(self.carID)
        self.lane = traci.vehicle.getLaneID(self.carID)  # 返回所在的车道
        self.lanePosition = traci.vehicle.getLanePosition(self.carID)  # 返回车辆已通行距离


    # 得到前车id
    def frontCar(self):
        m = float('inf')
        vehicle_frontCarID = ""
        for carID in traci.vehicle.getIDList():  # 找到车辆的前车
            lanePosition = traci.vehicle.getLanePosition(carID)
            if traci.vehicle.getLaneID(carID) == self.lane \
                    and self.lanePosition < lanePosition \
                    and lanePosition - self.lanePosition < self.follow_distance:
                if lanePosition - self.lanePosition < m:
                    m = lanePosition - self.lanePosition
                    vehicle_frontCarID = carID
        return vehicle_frontCarID

    # 相邻车道前车id
    def nearFrontCar(self):
        m = float('inf')
        vehicle_nearFrontCarID_0 = ""  # 相邻前车在车道0
        vehicle_nearFrontCarID_1 = ""  # 相邻前车在车道1
        vehicle_nearFrontCarID_2 = ""  # 相邻前车在车道2
        for carID in traci.vehicle.getIDList():
            lanePosition = traci.vehicle.getLanePosition(carID)
            if (self.lane == "lane0" or self.lane == "lane2") and traci.vehicle.getLaneID(carID) == "lane1" \
                    and self.lanePosition < lanePosition \
                    and lanePosition - self.lanePosition < self.follow_distance:  # 找到最右侧或最左侧车道的相邻车道车辆id
                if lanePosition - self.lanePosition < m:
                    m = lanePosition - self.lanePosition
                    vehicle_nearFrontCarID_1 = carID
            elif self.lane == "lane1" and self.lanePosition < lanePosition \
                    and lanePosition - self.lanePosition < self.follow_distance:  # 找到中间车道的相邻车道车辆id
                if traci.vehicle.getLaneID(carID) == "lane0":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearFrontCarID_0 = carID
                if traci.vehicle.getLaneID(carID) == "lane2":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearFrontCarID_2 = carID
        return vehicle_nearFrontCarID_0, vehicle_nearFrontCarID_1, vehicle_nearFrontCarID_2

    # 相邻车道后车id
    def nearRearCar(self):
        m = float('inf')
        vehicle_nearRearCarID_0 = ""  # 相邻后车在车道0
        vehicle_nearRearCarID_1 = ""  # 相邻后车在车道1
        vehicle_nearRearCarID_2 = ""  # 相邻后车在车道2
        for carID in traci.vehicle.getIDList():
            lanePosition = traci.vehicle.getLanePosition(carID)
            if (self.lane == "lane0" or self.lane == "lane2") and traci.vehicle.getLaneID(carID) == "lane1" \
                    and lanePosition < self.lanePosition:  # 找到最右侧车道或最左侧的相邻车道车辆id
                if lanePosition - self.lanePosition < m:
                    m = lanePosition - self.lanePosition
                    vehicle_nearRearCarID_1 = carID
            elif self.lane == "lane1" and lanePosition < self.lanePosition:  # 找到最右侧车道的相邻车道车辆id
                if traci.vehicle.getLaneID(carID) == "lane0":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearRearCarID_0 = carID
                if traci.vehicle.getLaneID(carID) == "lane2":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearRearCarID_2 = carID
        # print(vehicle_nearRearCarID_0, vehicle_nearRearCarID_1, vehicle_nearRearCarID_2)
        return vehicle_nearRearCarID_0, vehicle_nearRearCarID_1, vehicle_nearRearCarID_2

    # 生成下一时刻速度
    def speed_generate(self):
        v_next = self.speed
        Pslow, Ps = 0, 0  # 慢启动和随机慢化概率
        # Pslow, Ps = 0.5, 0.3  # 慢启动和随机慢化概率
        frontCar = self.frontCar()
        if frontCar != "":
            frontCarSpeed = traci.vehicle.getSpeed(frontCar)  # 前车速度
            frontCarDistance = traci.vehicle.getLanePosition(frontCar)  # 前车行驶通过距离
            minAccSpeed = min(self.speed + self.maxacc, self.vmax)
            if self.speed == 0 and random.uniform(0, 1) < Pslow:  # 慢启动现象
                v_next = 0  # 下一时刻速度为0
            elif frontCarSpeed + frontCarDistance - (
                    minAccSpeed + self.speed) / 2 - self.lanePosition > self.minGap + self.length:  # 加速情况
                v_next = minAccSpeed
                if random.uniform(0, 1) < Ps:  # 随机慢化现象
                    v_next = max(v_next - self.maxdec, 0)
            elif frontCarSpeed + frontCarDistance - (
                    minAccSpeed + self.speed) / 2 - self.lanePosition == self.minGap + self.length:  # 匀速情况
                if random.uniform(0, 1) < Ps:  # 随机慢化现象
                    v_next = max(v_next - self.maxdec, 0)
            else:  # 减速情况
                v_next = max(self.speed - self.maxdec, 0)
        else:
            v_next = min(self.speed + self.maxacc, self.vmax)
        return v_next

    # 判断是否换道
    def changeLane(self):
        ifChangeLane = False  # 是否变道
        leftChangeLane = False  # 向左变道
        rightChangeLane = False  # 向右变道
        Prc, Plc = 0.6, 0.9  # 向右、向左的换道概率
        nearFrontCar_0 = self.nearFrontCar()[0]
        nearFrontCar_1 = self.nearFrontCar()[1]
        nearFrontCar_2 = self.nearFrontCar()[2]
        nearRearCar_0 = self.nearRearCar()[0]
        nearRearCar_1 = self.nearRearCar()[1]
        nearRearCar_2 = self.nearRearCar()[2]
        frontCar = self.frontCar()
        minAccSpeed = min(self.speed + self.maxacc, self.vmax)
        # 0. 没有前车，或与前车距离过近，放弃换道
        if frontCar == "" or traci.vehicle.getLanePosition(frontCar) - self.lanePosition < self.minGap + self.length:
            ...
        # 1. 左车道向中间变道
        elif self.lane == "lane2":
            # 如果存在左后车，且距离不满足换道要求，则放弃换道
            if nearRearCar_1 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_1) < self.minGap + self.length:
                ...
            # 如果存在左前车，且距离不满足换道要求，或换道不是更好的选择，则放弃换道
            elif nearFrontCar_1 != "" \
                    and (traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition < self.minGap + self.length
                         or traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition <
                         traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            # 满足换道要求，计算换道概率
            elif random.uniform(0, 1) <= Prc:
                ifChangeLane = True
        # 2. 右车道向中间变道
        elif self.lane == "lane0":
            # 如果存在右后车，且距离不满足换道要求，则放弃换道
            if nearRearCar_1 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_1) < self.minGap + self.length:
                ...
            # 如果存在右前车，且距离不满足换道要求，或换道不是更好的选择，则放弃换道
            elif nearFrontCar_1 != "" \
                    and (traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition < self.minGap + self.length
                         or traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition <
                         traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            # 满足换道要求，计算换道概率
            elif random.uniform(0, 1) <= Plc:
                ifChangeLane = True
        # 3. 中间车道变道
        elif self.lane == "lane1":
            # 3.1. 中间车道向左变道
            # 如果存在左后车，且距离不满足换道要求，则放弃换道
            if nearRearCar_2 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_2) < self.minGap + self.length:
                ...
            # 如果存在左前车，且距离不满足换道要求，或换道不是更好的选择，则放弃换道
            elif nearFrontCar_2 != "" \
                    and (traci.vehicle.getLanePosition(nearFrontCar_2) - self.lanePosition < self.minGap + self.length
                         or traci.vehicle.getLanePosition(nearFrontCar_2) - self.lanePosition <
                         traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            # 满足换道要求，计算换道概率
            elif random.uniform(0, 1) <= Plc:
                ifChangeLane = True
                leftChangeLane = True
            # 3.2. 中间车道向右变道
            if ifChangeLane:
                ...
            # 如果存在右后车，且距离不满足换道要求，则放弃换道
            elif nearRearCar_0 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(
                nearRearCar_0) < self.minGap + self.length:
                ...
            # 如果存在右前车，且距离不满足换道要求，或换道不是更好的选择，则放弃换道
            elif nearFrontCar_0 != "" \
                    and (
                    traci.vehicle.getLanePosition(nearFrontCar_0) - self.lanePosition < self.minGap + self.length
                    or traci.vehicle.getLanePosition(nearFrontCar_0) - self.lanePosition <
                    traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            # 满足换道要求，计算换道概率
            elif random.uniform(0, 1) <= Prc:
                ifChangeLane = True
                rightChangeLane = True
        # if self.lane == "lane2" and nearFrontCar_1 != "" and frontCar != "" and nearRearCar_1 != "":
        #     if traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition > \
        #             traci.vehicle.getLanePosition(frontCar) - self.lanePosition \
        #             and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_1) > self.minGap + self.length \
        #             and random.uniform(0, 1) <= Prc:
        #         ifChangeLane = True
        # elif self.lane == "lane0" and nearFrontCar_1 != "" and frontCar != "" and nearRearCar_1 != "":
        #     # 相邻车道前车大于本车道前车的距离，相邻车道后车满足安全距离，当前车道受阻，公式16
        #     if traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition > \
        #             traci.vehicle.getLanePosition(frontCar) - self.lanePosition \
        #             and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_1) > self.minGap + self.length \
        #             and random.uniform(0, 1) <= Plc:
        #         ifChangeLane = True
        # elif self.lane == "lane1" and frontCar != "":
        #     if nearFrontCar_2 != "" and nearRearCar_2 != "" \
        #             and traci.vehicle.getLanePosition(nearFrontCar_2) - self.lanePosition > \
        #             traci.vehicle.getLanePosition(frontCar) - self.lanePosition \
        #             and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_2) > \
        #             self.minGap + self.length and random.uniform(0, 1) <= Plc:  # 左侧换道
        #         ifChangeLane = True
        #         leftChangeLane = True
        # elif nearFrontCar_0 != "" and nearRearCar_0 != "" \
        #         and traci.vehicle.getLanePosition(nearFrontCar_0) - self.lanePosition > \
        #         traci.vehicle.getLanePosition(frontCar) - self.lanePosition \
        #         and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_0) > \
        #         self.minGap + self.length and random.uniform(0, 1) <= Prc:  # 右侧换道
        #     ifChangeLane = True
        #     rightChangeLane = True
        return [ifChangeLane, leftChangeLane, rightChangeLane]

    def run(self):
        self.speed = traci.vehicle.getSpeed(self.carID)
        self.lane = traci.vehicle.getLaneID(self.carID)  # 返回所在的车道
        self.lanePosition = traci.vehicle.getLanePosition(self.carID)  # 返回车辆已通行距离
        # if traci.vehicle.getLanePosition(carID):  # 车辆在车道上进行控制
        changeLane = self.changeLane()
        if self.lane == "lane0" or self.lane == "lane2":  # 如果在最右侧车道或最左侧车道
            if changeLane[0]:
                traci.vehicle.changeLane(self.carID, 1, self.lanechange_time)
            else:  # 如果不进行换道，则更新速度与位置，下同
                speed_next = self.speed_generate()
                traci.vehicle.setSpeed(self.carID, speed_next)
                traci.vehicle.changeLane(self.carID, traci.vehicle.getLaneIndex(self.carID), 0)
                # traci.vehicle.moveTo(self.carID, self.lane, speed_next * (1 + self.p) * self.dt + self.lanePosition)

        elif self.lane == "lane1":  # 如果在中间车道
            if changeLane[0]:
                if changeLane[1]:
                    traci.vehicle.changeLane(self.carID, 2, self.lanechange_time)
                elif changeLane[2]:
                    traci.vehicle.changeLane(self.carID, 0, self.lanechange_time)
            else:
                speed_next = self.speed_generate()
                traci.vehicle.setSpeed(self.carID, speed_next)
                traci.vehicle.changeLane(self.carID, traci.vehicle.getLaneIndex(self.carID), 0)
                # traci.vehicle.moveTo(self.carID, self.lane, speed_next * (1 + self.p) * self.dt + self.lanePosition)


if __name__ == "__main__":
    cfg_sumo = 'config\\lane.sumocfg'
    sim_seed = 42
    app = "sumo-gui"
    command = [checkBinary(app), '-c', cfg_sumo]
    command += ['--routing-algorithm', 'dijkstra']
    # command += ['--collision.action', 'remove']
    command += ['--seed', str(sim_seed)]
    command += ['--no-step-log', 'True']
    command += ['--time-to-teleport', '300']
    command += ['--no-warnings', 'True']
    command += ['--duration-log.disable', 'True']
    command += ['--waiting-time-memory', '1000']
    command += ['--eager-insert', 'True']
    command += ['--lanechange.duration', '2']
    command += ['--lateral-resolution', '0.0']
    traci.start(command)

    cur_time = float(traci.simulation.getTime())
    traci.vehicle.add(vehID="veh0", routeID="straight", typeID="AV",
                      depart=cur_time, departLane=1, arrivalLane=0, departPos=0.0, arrivalPos=float('inf'),
                      departSpeed=5)
    traci.vehicle.add(vehID="veh1", routeID="straight", typeID="BV", arrivalLane=2,
                      depart=cur_time, departLane=1, departPos=40.0,
                      departSpeed=5)
    # traci.vehicle.add(vehID="veh2", routeID="straight", typeID="BV",
    #                   depart=cur_time, departLane=0, departPos=70.0,
    #                   departSpeed=5)
    # traci.vehicle.add(vehID="veh3", routeID="straight", typeID="BV",
    #                   depart=cur_time, departLane=2, departPos=30.0,
    #                   departSpeed=5)
    traci.simulationStep()
    car0 = fvdm_model("veh0")
    car1 = fvdm_model("veh1")
    for t in range(500):
        time.sleep(0.04)
        car0.run()
        car1.run()
        # traci.vehicle.setSpeed('veh1', 5)
        # traci.vehicle.setSpeed('veh2', 5)
        # traci.vehicle.setSpeed('veh3', 5)
        # car1.run()
        traci.simulationStep()
    traci.close()

    # for step in range(0, 3600):
    #     while traci.simulation.getMinExpectedNumber() > 0:
    #         traci.simulationStep()
    #         time.sleep(0.04)
    #         run()
    # traci.close()
    # sys.stdout.flush()