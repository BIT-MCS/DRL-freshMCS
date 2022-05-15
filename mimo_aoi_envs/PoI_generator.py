import numpy as np

num = 1024

poi_data = np.random.random(size=(num, 2))

for i, poi_i in enumerate(poi_data):
    if i == 0:
        print("[[%.10e,%.10e]," % (poi_i[0], poi_i[1]))
    elif i == num - 1:
        print("[%.10e,%.10e]]" % (poi_i[0], poi_i[1]))
    else:
        print("[%.10e,%.10e]," % (poi_i[0], poi_i[1]))
