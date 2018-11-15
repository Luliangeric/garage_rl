import tkinter as tk
from time import sleep
import json

UNIT = 30


class Vision(tk.Tk, object):
    def __init__(self, max_time):
        super(Vision, self).__init__()
        self.rate = 255 / max_time

        self.map_data = []
        self.parking = dict()

        self.resizable(width=False, height=True)

        self.Width = None
        self.Height = None
        self.plotform = None

        self.parking_index = dict()

    def initwindows(self, dataname, mapname):
        with open(dataname, 'r') as data:
            map_data = json.load(data)
            for key, value in map_data.items():
                self.parking_index[int(key)] = tuple(value[1])

        with open(mapname, 'r') as Map:
            while True:
                linestr = Map.readline()
                if linestr == "":
                    break
                else:
                    self.map_data.append(list(linestr.split()))

        self.Height = len(self.map_data)
        self.Width = len(self.map_data[0])

        self.plotform = tk.Canvas(self, height=self.Height * UNIT, width=self.Width * UNIT, bg='#888888')
        self._initparkings()

        self.plotform.pack()

    def _initparkings(self):
        for i in range(self.Height):
            for j in range(self.Width):
                if self.map_data[i][j] == 'P':
                    self.parking[(j, i)] = self.plotform.create_rectangle(j * UNIT, i * UNIT, (j + 1) * UNIT,
                                                                              (i + 1) * UNIT, fill='#0000ff')
                elif self.map_data[i][j] == 'B':
                    self.plotform.create_rectangle(j * UNIT, i * UNIT, (j + 1) * UNIT, (i + 1) * UNIT, fill='black')
                elif self.map_data[i][j] == 'I':
                    self.plotform.create_rectangle(j * UNIT, i * UNIT, (j + 1) * UNIT, (i + 1) * UNIT, fill='yellow')
                    self.imports = (j, i)
                elif self.map_data[i][j] == 'E':
                    self.plotform.create_rectangle(j * UNIT, i * UNIT, (j + 1) * UNIT, (i + 1) * UNIT, fill='green')
                    self.exit = (j, i)

    def updatesim(self, state, time=0.1):
        for i, item in enumerate(state):
            # r = int(item * self.rate)
            # b = 255 - r
            # sr = hex(r)[2:]
            # sb = hex(b)[2:]
            # if r < 16:
            #     sr = '0' + hex(r)[2:]
            # elif r > 239:
            #     sb = '0' + hex(b)[2:]
            #
            # color = '#' + sr + '00' + sb
            if item > 0:
                color = 'red'
            else:
                color = 'blue'
            self.plotform.itemconfig(self.parking[self.parking_index[i]], fill=color)
        self.update()
        sleep(time)


if __name__ == '__main__':
    temp = Vision(100)
    temp.initwindows("../data/garage.json", "../data/map1.txt")
    state = [i for i in range(94)]
    temp.updatesim(state)
    temp.mainloop()
