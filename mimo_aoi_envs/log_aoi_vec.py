import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
# from mimo_aoi_envs.plot3d import plot_cube,plot_circle
from mimo_aoi_envs.real_aoi.real_aoi_setting import Setting
class Log_DEBUG():
    def __init__(self, path, id, num_timeslots, detail_and_gif, mapx, mapy, num_uav, uav_trace, data_position, map_ob,
                 map_aoi, mean_aoi, mean_use_energy, aoi_list, energy_list, map_aoi_list,episodic_uav_snr_list,update_aoi_list,update_snr_list):
        self.path = path
        self.id = id
        self.num_timeslots = num_timeslots
        self.mapx = mapx
        self.mapy = mapy
        self.num_uav = num_uav
        self.uav_trace = uav_trace
        self.data_position = data_position
        self.map_ob = map_ob
        self.map_aoi = map_aoi
        self.mean_aoi = mean_aoi
        self.mean_use_energy = mean_use_energy
        self.aoi_list = aoi_list
        self.energy_list = energy_list
        self.map_aoi_list = map_aoi_list
        self.episodic_uav_snr_list = episodic_uav_snr_list
        self.update_aoi_list = update_aoi_list
        self.update_snr_list = update_snr_list
        # print(update_aoi_list)
        self.color_list = ["red", "purple", "green", "gray", "blue"]
        self.color_list_rgba=["#ff000012", "#9467bd20", "#2ca02c20", "#7f7f7f20", "#1f77b420"]
        self.xxx = []
        self.colors = []
        for x in range(self.mapx):
            self.xxx.append((x, 1))
        for y in range(self.mapy):
            c = []
            for x in range(self.mapx):
                # 1 represents obstacle,0 is blank
                if self.map_ob[x][y] == 1:
                    c.append((0, 0, 0, 1))
                else:
                    c.append((1, 1, 1, 1))
            self.colors.append(c)

        self.full_path = os.path.join(path, 'trajectories')
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path, exist_ok=True)
        self.detail = detail_and_gif
        if self.detail:
            self.detail_path = os.path.join(self.full_path, str(id))
            if not os.path.exists(self.detail_path):
                os.makedirs(self.detail_path, exist_ok=True)

    def draw_trajectory(self):
        if self.detail:
            for step_id in range(self.num_timeslots):
                self.draw_each_step(step_id)
            self.draw_gif()
            self.draw_result()
        else:
            self.draw_result()

    def save_to_file(self, method,iteration):
        aoi_list = np.array(self.aoi_list)
        ppath=self.full_path + "/pdf_infocom_%d/"%iteration
        os.makedirs(ppath,exist_ok=True)
        np.save(ppath+'aoi_' + method + '.npy', aoi_list)
        update_k_list = []
        for step in self.update_aoi_list:
            uav_k = []
            for uav in step:
                uav_k.append(len(uav))
            update_k_list.append(uav_k)

        k_list = np.array(update_k_list)
        snr_list = np.log10(np.array(self.update_snr_list)) * 10
        np.save(ppath+'k_list_' + method + '.npy', k_list)
        np.save(ppath+'snr_list_' + method + '.npy', snr_list)


    def draw_aoi_compare(self):
        if os.path.exists('./pdf_infocom') is False:
            os.makedirs('./pdf_infocom',exist_ok=True)
        pdf = PdfPages('./pdf_infocom/aoi_list.pdf')

        step = np.linspace(0,100,101)
        plt.figure(figsize=(20,13))
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.plot(step,self.aoi_list,label='uav_2',color='brown',linewidth=2)

        plt.xlabel("Timeslot",fontsize=40)
        plt.ylabel("Current Mean AoI",fontsize=40)
        plt.grid(True)
        plt.grid(linestyle='--')
        plt.legend(fontsize=40)
        ax=plt.gca()
        # ax.xaxis.get_major_formatter().set_powerlimits((0,1))
        #pdf.savefig()
        plt.show()
        plt.close()
        pdf.close()

    def draw_snr_list(self):
        update_k_list = []
        for step in self.update_aoi_list:
            uav_k = []
            for uav in step:
                uav_k.append(len(uav))
            update_k_list.append(uav_k)

        k_list = np.array(update_k_list)
        snr_list = np.log10(np.array(self.update_snr_list))*10
        if os.path.exists('./pdf_infocom') is False:
            os.makedirs('./pdf_infocom',exist_ok=True)
        pdf = PdfPages('./pdf_infocom/k_snr_list.pdf')

        step = np.linspace(1,100,100)


        fig = plt.figure(figsize=(20, 17))
        plt.plot(step, snr_list[:, 0], label='SNR', color='green',
                 markersize=28, markeredgewidth=5, markerfacecolor='none', linewidth=4)
        plt.xlabel("Timeslots", fontsize=32)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.ylabel("SNR", fontsize=40,color='green')
        plt.yticks(color='green')
        plt.legend(loc='upper left', fontsize=28)
        #plt.grid(True)
        #plt.grid(linestyle='--', color="black")
        ax = plt.gca()
        ax.spines['left'].set_color('green')

        plt.twinx()
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.plot(step, k_list[:, 0], label='K', color='blue',
                 markersize=28, markeredgewidth=5, markerfacecolor='none', linewidth=4)
        plt.ylim(ymax=50, ymin=0)
        plt.ylabel("K", fontsize=40,color='blue')
        plt.yticks(color='blue')
        plt.legend(loc='upper right', fontsize=28)
        plt.grid(True)
        plt.grid(linestyle='--')

        ax = plt.gca()
        ax.spines['left'].set_color('green')
        ax.spines['right'].set_color('blue')
        #pdf.savefig()
        plt.show()
        plt.close()
        pdf.close()

    def draw_3d_result(self):
        fig = plt.figure(figsize=(12, 8))
        ax = Axes3D(fig)
        ax.set_zlim(top=200,bottom=0)
        ax.view_init(azim=45, elev=70)
        PATH = np.array(self.uav_trace)

        # 渲染障碍物
        ## 1.渲染障碍物
        obstacle = Setting().V['OBSTACLE']
        for o in obstacle:
            plot_cube(ax, x=o[0], y=o[1], dx=o[2], dy=o[3],dz=50)
        ## 2.渲染POI
        color_list = ['']
        ax.scatter(self.data_position[:, 0], self.data_position[:, 1],0, c=self.get_PoI_color(), marker="o",
                    s=50, alpha=0.7, edgecolors="black")

        # 图表标题设置
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        group_labels = np.linspace(0, 1600, 9).astype(np.int)
        x = np.linspace(0, self.mapx, 9).astype(np.int)
        y = np.linspace(0, self.mapy, 9).astype(np.int)
        plt.xlim(left=self.mapx, right=0)
        plt.ylim(bottom=self.mapy, top=0)
        plt.xticks(x, group_labels, rotation=0)
        plt.yticks(y, group_labels, rotation=0)
        ax.set_zticks(np.linspace(0,200,5))

        # 渲染无人机轨迹
        line_tall = 130
        tall = np.linspace(0, line_tall, 3)


        uav_center = []
        uav_radius = []
        for num in range(self.num_uav):
            uav_center.append([])
            uav_radius.append([])

        for step in self.update_aoi_list:
            # 每一步更新
            for index, uav in enumerate(step):
                #每个uav的更新poi列表
                center, radius = cv2.minEnclosingCircle(self.data_position[uav])
                uav_center[index].append([center[0],center[1]])
                uav_radius[index].append(radius)


        uav_center = np.array(uav_center)
        uav_radius = np.array(uav_radius)
        uav_center[uav_center == 0] = np.inf
        none_mask = np.isnan(uav_center)


        for i in range(self.num_uav):
            color = self.color_list[i]
            color_rgba = self.color_list_rgba[i]
            for j in range(len(uav_center[i])):
                # 先画空中的轨迹
                ax.scatter(uav_center[i, j, 0], uav_center[i, j, 1], line_tall, color=color, marker="^")
                ax.plot(uav_center[i, :, 0], uav_center[i, :, 1], line_tall, color=color)
                # 地面上的圆
                if uav_radius[i,j] == 0:
                    continue
                plot_circle(ax = ax,
                            x=uav_center[i, j, 0],
                            y=uav_center[i, j, 1],
                            radius=uav_radius[i,j],
                            color=color_rgba)
        fig.savefig(self.full_path + '/%d.png' % self.id)
        plt.close()

        plt.show()

    def draw_result(self):
        Fig = plt.figure(figsize=(6, 6))
        PATH = np.array(self.uav_trace)

        for i1 in range(self.mapy):
            plt.broken_barh(self.xxx, (i1, 1), facecolors=self.colors[i1])

        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        group_labels = np.linspace(0, 1600, 9).astype(np.int32)
        x = np.linspace(0, self.mapx, 9).astype(np.int32)
        y = np.linspace(0, self.mapy, 9).astype(np.int32)
        plt.xlim(left=0, right=self.mapx)
        plt.ylim(bottom=0, top=self.mapy)
        plt.xticks(x, group_labels, rotation=0)
        plt.yticks(y, group_labels, rotation=0)

        plt.xticks()
        plt.grid(True, linestyle='-.', color='black')
        plt.title('Mean aoi=%.2f s' % self.mean_aoi + ',Energy=%.2f kJ' % self.mean_use_energy)
        plt.scatter(self.data_position[:, 0], self.data_position[:, 1], c=self.num_timeslots - self.map_aoi, marker="o",
                    s=50, alpha=0.7, edgecolors="black")

        for i in range(self.num_uav):
            # M = Fig.add_subplot(1, 1, i + 1)
            color = self.color_list[i]
            plt.plot(PATH[i, :, 0], PATH[i, :, 1], color=color)
            for j in range(len(PATH[i])):
                plt.scatter(PATH[i, j, 0], PATH[i, j, 1], color=color, marker="^")


        # uav_center = []
        # uav_radius = []
        # for num in range(self.num_uav):
        #     uav_center.append([])
        #     uav_radius.append([])
        #
        # for step in self.update_aoi_list:
        #     # 每一步更新
        #     for index, uav in enumerate(step):
        #         # 每个uav的更新poi列表
        #         center, radius = cv2.minEnclosingCircle(self.data_position[uav])
        #         uav_center[index].append([center[0], center[1]])
        #         uav_radius[index].append(radius)
        #
        # uav_center = np.array(uav_center)
        # uav_radius = np.array(uav_radius)
        # uav_center[uav_center == 0] = np.inf
        # none_mask = np.isnan(uav_center)
        #
        # for i in range(self.num_uav):
        #     color = self.color_list[i]
        #     color_rgba = self.color_list_rgba[i]
        #     for j in range(len(uav_center[i])):
        #         # 先画空中的轨迹
        #         plt.scatter(uav_center[i, j, 0], uav_center[i, j, 1], color=color, marker="^")
        #         plt.plot(uav_center[i, :, 0], uav_center[i, :, 1], color=color)
        #         # 地面上的圆
        #         if uav_radius[i, j] == 0:
        #             continue
        #
        #         u = np.linspace(0, 2 * np.pi, 100)
        #         v = np.linspace(0, np.pi, 100)
        #         x = (uav_radius[i, j]+0.1) * np.outer(np.cos(u), np.sin(v)) + uav_center[i, j, 0]
        #         y = (uav_radius[i, j]+0.1) * np.outer(np.sin(u), np.sin(v)) + uav_center[i, j, 1]
        #
        #         plt.plot(x,y,color=color_rgba,linewidth=0.5)


        Fig.savefig(self.full_path + '/%d.png' % self.id)
        plt.close()


    def draw_each_step(self, step2):
        Fig = plt.figure(figsize=(6, 6))
        PATH = np.array(self.uav_trace)

        for i1 in range(self.mapy):
            plt.broken_barh(self.xxx, (i1, 1), facecolors=self.colors[i1])

        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        group_labels = np.linspace(0, 1600, 9).astype(np.int)
        x = np.linspace(0, self.mapx, 9).astype(np.int)
        y = np.linspace(0, self.mapy, 9).astype(np.int)
        plt.xlim(left=0, right=self.mapx)
        plt.ylim(bottom=0, top=self.mapy)
        plt.xticks(x, group_labels, rotation=0)
        plt.yticks(y, group_labels, rotation=0)

        plt.xticks()
        plt.grid(True, linestyle='-.', color='black')
        plt.title('Mean aoi=%.2f s' % self.aoi_list[step2 + 1] + ',Energy=%.2f kJ' % self.energy_list[step2 + 1])
        plt.scatter(self.data_position[:, 0], self.data_position[:, 1],
                    c=self.num_timeslots - self.map_aoi_list[step2 + 1], marker="o",
                    s=50, alpha=0.7, edgecolors="black")

        # for i in range(self.num_uav):
        #     color = self.color_list[i]
        #     plt.plot(PATH[i, step:step + 2, 0], PATH[i, step:step + 2, 1], color=color)
        #     plt.scatter(PATH[i, step + 1, 0], PATH[i, step + 1, 1], color=color, marker="^")

        uav_center = []
        uav_radius = []
        for num in range(self.num_uav):
            uav_center.append([])
            uav_radius.append([])

        for step in self.update_aoi_list:
            # 每一步更新
            for index, uav in enumerate(step):
                # 每个uav的更新poi列表
                center, radius = cv2.minEnclosingCircle(self.data_position[uav])
                uav_center[index].append([center[0], center[1]])
                uav_radius[index].append(radius)

        uav_center = np.array(uav_center)
        uav_radius = np.array(uav_radius)
        uav_center[uav_center == 0] = np.inf
        none_mask = np.isnan(uav_center)

        for i in range(self.num_uav):
            color = self.color_list[i]
            color_rgba = self.color_list_rgba[i]
            for j in range(len(uav_center[i])):
                # 先画空中的轨迹
                plt.scatter(uav_center[i, j, 0], uav_center[i, j, 1], color=color, marker="^")
                plt.plot(uav_center[i, :, 0], uav_center[i, :, 1], color=color)
                # 地面上的圆
                if uav_radius[i, j] == 0:
                    continue

                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = (uav_radius[i, j] + 0.1) * np.outer(np.cos(u), np.sin(v)) + uav_center[i, j, 0]
                y = (uav_radius[i, j] + 0.1) * np.outer(np.sin(u), np.sin(v)) + uav_center[i, j, 1]

                plt.plot(x, y, color=color_rgba)

        Fig.savefig(self.detail_path + '/%d-%d.png' % (self.id, step2))
        plt.close()
        if step2 % 50 == 0:
            print("Finished %d-%d.png" % (self.id, step2))

    def draw_gif(self):
        fig = plt.figure()
        ppt = []
        for i in range(self.num_timeslots):
            img = cv2.imread(self.detail_path + '/%d-%d.png' % (self.id, i))
            (r, g, b) = cv2.split(img)
            img = cv2.merge([b, g, r])
            im = plt.imshow(img, animated=True)
            ppt.append([im])

        ani = animation.ArtistAnimation(fig, ppt, interval=200, blit=True, repeat_delay=1000)
        ani.save(self.full_path + '/a%d.gif' % self.id, writer='imagemagick', fps=10)
        print("Finished %d.gif" % self.id)

    def get_PoI_color(self):
        color_list = ['#001449','#321449','#511849', '#3d3d6b', '#2a7b96', '#00baad', '#57c785', '#add45c','#ffc300','#eddd53']
        color_list.reverse()
        color = []
        for i in self.map_aoi:
            color.append(color_list[int(i//(250//len(color_list)-1))])

        return color


  # def draw_3d_result(self):
  #
  #       fig = plt.figure(figsize=(12, 8))
  #       ax = Axes3D(fig)
  #       ax.view_init(azim=45, elev=60)
  #       PATH = np.array(self.uav_trace)
  #
  #       # 渲染障碍物
  #       ## 1.渲染障碍物
  #       obstacle = Setting().V['OBSTACLE']
  #       for o in obstacle:
  #           plot_cube(ax, x=o[0], y=o[1], dx=o[2], dy=o[3])
  #       ## 2.渲染POI
  #       ax.scatter(self.data_position[:, 0], self.data_position[:, 1],0, c=self.num_timeslots - self.map_aoi, marker="o",
  #                   s=50, alpha=0.7, edgecolors="black")
  #
  #       # 图表标题设置
  #       plt.xlabel("X (m)")
  #       plt.ylabel("Y (m)")
  #       group_labels = np.linspace(0, 1600, 9).astype(np.int)
  #       x = np.linspace(0, self.mapx, 9).astype(np.int)
  #       y = np.linspace(0, self.mapy, 9).astype(np.int)
  #       plt.xlim(left=self.mapx, right=0)
  #       plt.ylim(bottom=self.mapy, top=0)
  #       plt.xticks(x, group_labels, rotation=0)
  #       plt.yticks(y, group_labels, rotation=0)
  #       ax.set_zlim(bottom=0,top=3)
  #
  #       # 渲染无人机轨迹
  #       line_tall = 2
  #       tall = np.linspace(0, line_tall, 3)
  #       for i in range(self.num_uav):
  #           # M = Fig.add_subplot(1, 1, i + 1)
  #           color = self.color_list[i]
  #           ax.plot(PATH[i, :, 0], PATH[i, :, 1], line_tall,color=color)
  #           ax.plot(PATH[i, :, 0], PATH[i, :, 1], 0, color=color, linestyle='-.')
  #           for j in range(len(PATH[i])):
  #               ax.scatter(PATH[i, j, 0], PATH[i, j, 1], line_tall, color=color, marker="^")
  #               ax.plot(np.full_like(tall,PATH[i, j, 0]), np.full_like(tall,PATH[i, j, 1]),tall , color=color,linestyle='--',linewidth=0.5)
  #
  #       fig.savefig(self.full_path + '/%d.png' % self.id)
  #       plt.close()