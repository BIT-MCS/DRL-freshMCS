import os
import matplotlib.pyplot as plt
import numpy as np


def circle(x, y, r, color='red', count=100):
    xarr = []
    yarr = []
    for i in range(count):
        j = float(i) / count * 2 * np.pi
        xarr.append(x + r * np.cos(j))
        yarr.append(y + r * np.sin(j))
    plt.plot(xarr, yarr, c=color, linewidth=2)

def draw_trajectory(path, id, mapx, mapy, num_uav, trace, energytrace, crange, fills, datas, mapob, data_collection,
                    efficiency,DATAs):
    full_path = os.path.join(path, 'trajectories')
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)
    xxx = []
    color_list = ["red", "purple", "green", "grey", "blue"]
    colors = []

    for x in range(mapx):
        xxx.append((x, 1))
    for y in range(mapy):
        c = []
        for x in range(mapx):
            # 1 represents obstacle,0 is blank
            if mapob[x][y] == 1:
                c.append((0, 0, 0, 1))
            else:
                c.append((1, 1, 1, 1))
        colors.append(c)

    Fig = plt.figure(figsize=(6, 6))
    PATH = np.array(trace)
    ENERGY_PATH = np.array(energytrace)

    for i1 in range(mapy):
        plt.broken_barh(xxx, (i1, 1), facecolors=colors[i1])

    plt.scatter(datas[:, 0], datas[:, 1], c=DATAs[:, 2], marker="s")

    for i in range(num_uav):
        # M = Fig.add_subplot(1, 1, i + 1)
        plt.ylim(bottom=0, top=mapy)
        plt.xlim(left=0, right=mapx)
        color = color_list[i]
        plt.plot(PATH[i, :, 1], PATH[i, :, 2], color=color)
        for j in range(len(PATH[i])):
            if PATH[i, j, 0] >= 0:
                plt.scatter(PATH[i, j, 1], PATH[i, j, 2], color=color, marker=".", norm=ENERGY_PATH[i])
            else:
                plt.scatter(PATH[i, j, 1], PATH[i, j, 2], color=color, marker="+", norm=ENERGY_PATH[i])
        # grid line
        plt.grid(True, linestyle='-.', color='black')
        # title
        plt.title('Collection=' + str(round(data_collection, 3)) + ',Efficiency=' + str(
            round(efficiency, 3)))

    for (x, y) in zip(fills[:, 0], fills[:, 1]):
        circle(x, y, crange)
    plt.scatter(fills[:, 0], fills[:, 1], c='red', marker="*")

    Fig.savefig(full_path + '/%d.png'%id)
    plt.close()
