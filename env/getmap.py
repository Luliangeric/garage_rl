import json
import numpy as np

EXPAND = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class MapDate:
    def __init__(self):

        self._MapData = list()
        self.height = 0
        self.width = 0
        self.init_map()

    def init_map(self, map_file="../data/map1.txt"):

        with open(map_file, 'r') as map_:
            while 1:
                line_str = map_.readline()
                if line_str == "":
                    break
                else:
                    self._MapData.append(list(line_str.split()))

        self.height = len(self._MapData)
        self.width = len(self._MapData[0])

    def get_parking_dis(self):
        res = list()
        parking = list()
        for i in range(self.height):
            for j in range(self.width):
                if self._MapData[i][j] == 'I':
                    imports = (j, i)
                elif self._MapData[i][j] == 'E':
                    exits = (j, i)
                elif self._MapData[i][j] == 'P':
                    parking.append((j, i))

        for pos in parking:
            res.append((len(self.astar(imports, pos)) + len(self.astar(pos, exits)), tuple(pos)))
        res.sort(key=lambda x: x[0])
        Map = dict()
        for i, item in enumerate(res):
            Map[i] = res[i]
        return Map

    def astar(self, spos, gpos):

        temp = self._MapData[gpos[1]][gpos[0]]
        self._MapData[gpos[1]][gpos[0]] = 'X'
        openlist = dict()
        closelist = dict()

        openlist[spos] = [(0, 0), spos, np.inf, 0]

        while not self._expend(openlist, closelist, gpos) and len(openlist):
            pass

        node = gpos
        path = list()
        path.append(gpos)
        while 1:
            try:
                node = closelist[node]
                if node == spos:
                    break
                path.append(node)
            except KeyError:
                pass
        path.reverse()

        self._MapData[gpos[1]][gpos[0]] = temp
        return path

    def _expend(self, openlist, closelist, gpos):
        node = sorted(openlist.values(), key=lambda x: x[2], reverse=True).pop()
        openlist.pop(node[1])
        if node[1] == gpos:
            closelist[gpos] = node[0]
            return 1
        for i in range(4):
            tempnode = np.array(node[1]) + np.array(EXPAND[i])
            try:
                closelist[tuple(tempnode)]
            except KeyError:
                try:
                    if self._MapData[tempnode[1]][tempnode[0]] == 'X':
                        g = node[-1] + 1
                        heuris = abs(tempnode[0] - gpos[0]) + abs(tempnode[1] - gpos[1])

                        try:
                            existnode = openlist[tuple(tempnode)]
                            if heuris + g < existnode[2]:
                                openlist[tuple(tempnode)][2] = heuris + g
                        except KeyError:
                            nextnode = [node[1], tuple(tempnode), heuris + g, g]
                            openlist[tuple(tempnode)] = nextnode
                except:
                    pass
        closelist[node[1]] = node[0]
        return 0


if __name__ == '__main__':
    with open("../data/garage.json", 'w', newline='') as garage:
        temp = MapDate()
        data = temp.get_parking_dis()
        json.dump(data, garage)
        print('parking number:', len(data))