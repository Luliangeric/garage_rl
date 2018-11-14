import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt


T = 4000
Car = namedtuple('car', ['arrive_time', 'wait_time', 'wait_max', 'mass'])


def step(t):
    alf = 0.2
    rate_fast = 1
    rate_slow = 30

    if (t % T) < alf * T:
        return rate_fast
    else:
        return rate_slow


class CarList:
    def __init__(self, t_range, wt_range, m_range, max_num):
        self.t = 0

        self.m_r = m_range
        self.t_r = t_range
        self.wt_r = wt_range
        self.arrive_t = np.random.randint(1, 20) * step(self.t)

        self.max = max_num
        self.index = 1
        self.done = False

        self.wait_car_list = deque()

    def get_car(self, full=False):
        self.t += 1

        if self.t < self.arrive_t:
            temp = None
        else:
            temp = Car(self.arrive_t, np.random.randint(self.t_r[0], self.t_r[1]),
                       np.random.randint(self.wt_r[0], self.wt_r[1]),
                       np.random.randint(self.m_r[0], self.m_r[1]))

            self.index += 1
            if self.index > self.max:
                self.done = True

            self.arrive_t += np.random.randint(1, 20) * step(self.t)

        if full:
            if temp:
                self.wait_car_list.append(temp)
            return None
        else:
            if self.wait_car_list:
                if temp:
                    self.wait_car_list.append(temp)
                temp = self.wait_car_list.popleft()
                temp = Car(self.t, temp.wait_time, temp.wait_max, temp.mass)
            return temp


if __name__ == '__main__':
    t_range, wt_range, m_range, max_num = [1000, 7000], [10, 20], [5, 10], 800
    carlist = CarList(t_range, wt_range, m_range, max_num)
    number = 0
    time = list()
    car_number = list()

    time_out = set()

    while not carlist.done:
        car = carlist.get_car(full=False)
        print(car)
        if car:
            time_out.add(car.arrive_time + car.wait_time)
            number = number + 1
        while carlist.t in time_out:
            number -= 1
            time_out.remove(carlist.t)

        time.append(carlist.t)
        car_number.append(number)

    plt.plot(time, car_number)
    plt.show()