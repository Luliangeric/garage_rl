from env.carlist import CarList
import matplotlib.pyplot as plt
import json


class Garage:
    def __init__(self, filename):
        with open(filename, 'r') as garage:
            data = json.load(garage)
            self.parking_dis = [x[0] for x in data.values()]
        self.parking_number = len(self.parking_dis)
        self.state = [0 for _ in range(self.parking_number)]

        self.norm_t = None
        self.norm_mass = None
        self.baseline = None

        self.car_list = None

        self.car_num = 0

    def check_state(self):
        self.car_num = 0
        for i in range(self.parking_number):
            if self.state[i] > 0:
                self.state[i] -= 1
                if self.state[i] > 0:
                    self.car_num += 1

        car = self.car_list.get_car(full=self.car_num == self.parking_number)

        if car:
            state = [x * self.norm_t for x in self.state]
            state.extend((car.wait_time * self.norm_t, car.mass * self.norm_mass))
        else:
            state = None
        return state, car

    def step(self, parking_num, car):
        self.state[parking_num] = car.wait_time

        reward = self.baseline - car.mass * self.norm_mass * self.parking_dis[parking_num] / self.parking_dis[-1]
        return reward, self.car_list.done

    def reset(self):
        t_range, wt_range, m_range, max_num = [1000, 7000], [10, 20], [5, 10], 100
        self.norm_t, self.norm_mass = 1 / t_range[1], 1 / m_range[1]
        self.baseline = (m_range[0] + m_range[1]) * (self.parking_dis[0] + self.parking_dis[-1]) / \
                        self.parking_dis[-1] / 4 * self.norm_mass

        self.car_list = CarList(t_range, wt_range, m_range, max_num)

        self.car_num = 0


if __name__ == '__main__':
    temp = Garage("../data/garage.json")
    temp.reset()
    num = list()
    time = list()
    sum = 0
    while True:
        s_, car_ = temp.check_state()
        if car_:
            done = False
            for i in range(temp.parking_number):
                if temp.state[i] == 0:
                    reward_, done = temp.step(i, car_)
                    sum += reward_
                    print(reward_)
                    break
            if done:
                break

        num.append(temp.car_num)
        time.append(temp.car_list.t)

    print('sum = ', sum)
    plt.plot(time, num)
    plt.show()
    pass
