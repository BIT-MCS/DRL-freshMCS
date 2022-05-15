class Setting(object):
    def __init__(self, log):
        self.V = {
            'MAP_X': 16,
            'MAP_Y': 16,
            'MAX_VALUE': 1.,
            'MIN_VALUE': 0.,
            'DATA': [[8.3549332619e-01, 9.4903570414e-01, 1.4485265128e-02],
                     [7.2130966187e-01, 1.7438349128e-01, 7.7217406034e-01],
                     [1.7968750000e-01, 1.4575195312e-01, 9.3847656250e-01],
                     [5.9241038561e-01, 5.8389687538e-01, 9.0609413385e-01],
                     [2.5722867250e-01, 7.5673032552e-03, 8.0785298347e-01],
                     [4.1423335671e-01, 4.6635714173e-01, 2.3157824576e-01],
                     [1.0672294348e-01, 6.2129569054e-01, 3.0989482999e-01],
                     [5.1904296875e-01, 7.5976562500e-01, 4.8291015625e-01],
                     [8.3140838146e-01, 7.0169627666e-02, 2.4170003831e-01],
                     [5.5175781250e-01, 8.5302734375e-01, 3.3959960938e-01],
                     [9.5707613230e-01, 1.6726264358e-01, 6.1919361353e-01],
                     [9.5888471603e-01, 4.5630550385e-01, 8.7089371681e-01],
                     [3.9599609375e-01, 1.3391113281e-01, 8.5644531250e-01],
                     [7.5441598892e-01, 1.1350642145e-01, 5.9975004196e-01],
                     [7.9367744923e-01, 6.3090234995e-01, 6.5466493368e-01],
                     [6.8066406250e-01, 7.0947265625e-01, 7.9931640625e-01],
                     [8.3784568310e-01, 3.0604279041e-01, 2.2133958340e-01],
                     [9.4845280051e-02, 9.2455798388e-01, 4.3918646872e-02],
                     [2.2278437018e-01, 7.7790427022e-03, 4.0608885884e-01],
                     [5.6192117929e-01, 1.6118277609e-01, 9.6003049612e-01],
                     [3.5304900259e-02, 7.7715098858e-02, 2.9024279118e-01],
                     [5.9664958715e-01, 8.3100549877e-02, 1.7508490384e-01],
                     [9.2007553577e-01, 1.3995824754e-01, 4.0349757671e-01],
                     [9.6916091442e-01, 2.3865221441e-01, 7.2321671247e-01],
                     [7.2004568577e-01, 1.1379265040e-01, 1.3811156154e-01],
                     [4.3336719275e-01, 6.2521293759e-02, 4.5298880339e-01],
                     [8.4907568991e-02, 3.9443579316e-01, 5.9256303310e-01],
                     [8.3184188604e-01, 5.1894861460e-01, 9.7502315044e-01],
                     [1.0198974609e-01, 5.3027343750e-01, 8.2168579102e-03],
                     [8.1818187237e-01, 2.1586267650e-01, 6.9178700447e-01],
                     [7.8221589327e-01, 9.4052962959e-02, 9.9049305916e-01],
                     [8.1347656250e-01, 1.1962890625e-01, 9.8095703125e-01],
                     [7.8662043810e-01, 3.9115092158e-01, 4.2293024063e-01],
                     [7.2735446692e-01, 1.7736417055e-01, 8.9794403315e-01],
                     [1.9113677740e-01, 8.9124190807e-01, 1.3489100337e-01],
                     [8.8126695156e-01, 3.5160303116e-01, 3.5368844867e-01],
                     [6.9399052858e-01, 9.6150028706e-01, 1.3233169913e-01],
                     [9.4517523050e-01, 1.7947681248e-01, 8.8766735792e-01],
                     [8.5698544979e-01, 1.7686622217e-02, 2.3212364316e-01],
                     [7.8553456068e-01, 2.6661336422e-02, 8.0103144050e-02],
                     [4.0668609738e-01, 1.7458504438e-01, 6.0972887278e-01],
                     [4.4861286879e-01, 4.1550755501e-01, 9.9082231522e-01],
                     [5.5468750000e-01, 7.5634765625e-01, 9.3505859375e-01],
                     [3.4179499745e-01, 6.5522867441e-01, 3.0777776241e-01],
                     [2.4697451293e-01, 1.7116887495e-02, 2.7037641406e-01],
                     [3.1030273438e-01, 9.8419189453e-03, 5.8691406250e-01],
                     [7.4186258018e-02, 6.2428802252e-01, 2.0738197863e-01],
                     [8.1763699651e-02, 4.8560926318e-01, 8.2438874245e-01],
                     [3.0224609375e-01, 2.7392578125e-01, 8.6083984375e-01],
                     [5.8343660831e-01, 7.9691636562e-01, 4.1798821092e-01],
                     [8.3277755976e-01, 9.1173088551e-01, 3.8258484006e-01],
                     [3.2369497418e-01, 2.0247344673e-01, 4.9962576479e-02],
                     [7.7681225538e-01, 9.1748046875e-01, 4.9604067206e-01],
                     [8.8536912203e-01, 9.1364866495e-01, 9.1757118702e-01],
                     [1.8793259561e-01, 6.4230054617e-01, 9.2787854373e-02],
                     [5.9570312500e-01, 4.4482421875e-01, 7.0019531250e-01],
                     [7.3322974145e-02, 6.1659944057e-01, 2.1614573896e-01],
                     [5.0277817249e-01, 6.7377817631e-01, 1.6596293449e-01],
                     [8.7054735422e-01, 7.0574343204e-01, 1.6473831236e-01],
                     [6.1191070080e-01, 7.6535023749e-02, 4.5660355687e-01],
                     [7.9340857267e-01, 1.4460492134e-01, 2.3495869339e-01],
                     [2.0439709723e-01, 8.4496356547e-02, 7.8994351625e-01],
                     [5.4150390625e-01, 2.0056152344e-01, 2.1740722656e-01],
                     [2.7954101562e-01, 8.0371093750e-01, 9.8681640625e-01],
                     [5.0698913634e-02, 3.8461935520e-01, 1.9253575802e-01],
                     [3.5693359375e-01, 2.1984863281e-01, 4.7509765625e-01],
                     [2.7880859375e-01, 2.3522949219e-01, 4.8413085938e-01],
                     [6.7658764124e-01, 9.8871475458e-01, 4.6548274159e-01],
                     [9.6045005322e-01, 8.5414981842e-01, 6.9561630487e-01],
                     [1.5496706963e-01, 1.3950237632e-01, 3.5531812906e-01],
                     [4.6191406250e-01, 9.3701171875e-01, 7.0947265625e-01],
                     [9.8095703125e-01, 5.3417968750e-01, 6.9824218750e-01],
                     [4.5622742176e-01, 1.5933208168e-01, 1.1969146132e-01],
                     [9.2236328125e-01, 1.0913085938e-01, 6.9238281250e-01],
                     [5.7806760073e-01, 6.9359081984e-01, 2.7218174934e-01],
                     [1.0012771934e-01, 7.2071874142e-01, 5.5361521244e-01],
                     [9.0676522255e-01, 8.7288480997e-01, 4.2120260000e-01],
                     [2.8734460473e-01, 5.6828832626e-01, 6.3633501530e-01],
                     [4.6602591872e-01, 7.2329938412e-01, 5.4176056385e-01],
                     [5.8860814571e-01, 6.1920189857e-01, 5.2764666080e-01],
                     [1.4558754861e-01, 4.2142999172e-01, 2.2998242080e-01],
                     [1.9481472671e-01, 6.9435030222e-02, 6.9937288761e-01],
                     [2.8173828125e-01, 7.1337890625e-01, 5.9356689453e-02],
                     [5.7361769676e-01, 4.5054298639e-01, 1.2839442492e-01],
                     [6.8671923876e-01, 1.3762013614e-01, 2.9044938087e-01],
                     [9.5556640625e-01, 7.1972656250e-01, 9.2285156250e-02],
                     [5.9701746702e-01, 3.8837137818e-01, 8.1242620945e-01],
                     [5.6485193968e-01, 9.0640997887e-01, 3.1604447961e-01],
                     [4.5660910010e-01, 8.8930815458e-01, 7.6832813025e-01],
                     [1.2238394469e-02, 3.3213269711e-01, 2.3362794518e-01],
                     [9.8146039248e-01, 9.8467588425e-01, 7.8813260794e-01],
                     [9.8896944523e-01, 1.9912499189e-01, 2.6315009594e-01],
                     [3.1744053960e-01, 3.3596986532e-01, 8.5733813047e-01],
                     [2.2204589844e-01, 3.4313201904e-03, 5.2978515625e-01],
                     [5.2131283283e-01, 5.2113354206e-01, 4.8748165369e-01],
                     [7.5192189217e-01, 9.3182629347e-01, 3.8956251740e-01],
                     [9.4888585806e-01, 1.9638332725e-01, 8.6754120886e-02],
                     [2.1908760071e-01, 2.1533429623e-01, 3.9748489857e-01],
                     [9.2979764938e-01, 6.8710303307e-01, 3.4099566936e-01],
                     [1.5286271274e-01, 9.1518115997e-01, 3.1389033794e-01],
                     [1.0217271000e-01, 1.2362764031e-01, 6.7666488886e-01],
                     [8.0932617188e-02, 1.6589355469e-01, 9.3945312500e-01],
                     [4.2724609375e-01, 4.0820312500e-01, 9.4970703125e-01],
                     [3.2612469792e-01, 3.2795542479e-01, 7.7227699757e-01],
                     [9.7851562500e-01, 6.0791015625e-01, 6.5576171875e-01],
                     [6.6868680716e-01, 1.7547428608e-02, 2.4743552506e-01],
                     [6.8014836311e-01, 4.6294069290e-01, 4.7206601501e-01],
                     [6.9486689568e-01, 7.1947133541e-01, 1.5760490298e-01],
                     [9.4067323208e-01, 1.2168993801e-01, 5.8045852184e-01],
                     [5.3711068630e-01, 8.7587279081e-01, 2.2412669659e-01],
                     [5.6129592657e-01, 8.3974355459e-01, 5.1013982296e-01],
                     [2.8433018923e-01, 7.7719414234e-01, 9.8128594458e-02],
                     [4.6557617188e-01, 2.3400878906e-01, 3.8037109375e-01],
                     [6.0010796785e-01, 8.1898760796e-01, 4.7725719213e-01],
                     [5.1108165644e-03, 2.3073831201e-01, 2.1475683153e-01],
                     [2.9878291488e-01, 3.7278565764e-01, 7.1430101991e-02],
                     [4.4540992379e-01, 2.1831369400e-01, 5.7106798887e-01],
                     [1.5514038503e-01, 1.1449321359e-01, 6.1000299454e-01],
                     [8.8860577345e-01, 2.5895839930e-01, 3.4598100185e-01],
                     [3.3524450660e-01, 3.2277691364e-01, 9.9260181189e-01],
                     [2.8222656250e-01, 4.9731445312e-01, 2.9125976562e-01],
                     [1.9954775274e-01, 6.9961398840e-03, 2.7525794506e-01],
                     [6.6327202320e-01, 1.1489849538e-01, 3.8592416048e-01],
                     [3.1449726224e-01, 6.8597882986e-01, 6.8875247240e-01],
                     [3.9503943920e-01, 9.6803337336e-02, 8.1368899345e-01],
                     [7.7084161341e-02, 8.4046828747e-01, 5.2031099796e-01],
                     [3.3740234375e-01, 4.3676757812e-01, 1.7553710938e-01],
                     [7.6647621393e-01, 4.4073814154e-01, 5.6574976444e-01],
                     [6.6801559925e-01, 9.4462889433e-01, 2.6367962360e-01],
                     [5.7698851824e-01, 3.8633239269e-01, 3.3411407471e-01],
                     [5.4417455196e-01, 6.4251795411e-02, 2.4689210951e-01],
                     [7.4464380741e-02, 8.9093619585e-01, 4.7468653321e-01],
                     [1.0071595013e-01, 8.9697086811e-01, 4.4318181276e-01],
                     [9.9118775129e-01, 1.2860167027e-01, 1.7370793968e-02],
                     [7.8808593750e-01, 6.5185546875e-01, 6.4013671875e-01],
                     [9.6883124113e-01, 9.9406421185e-01, 4.1870480776e-01],
                     [4.6270641685e-01, 7.1045798063e-01, 8.8926535845e-01],
                     [2.4168247357e-02, 3.7225973606e-01, 6.3025748730e-01],
                     [8.0377167463e-01, 5.6650352478e-01, 8.0114901066e-02],
                     [8.4465402365e-01, 6.5805417299e-01, 6.3631886244e-01],
                     [5.9608536959e-01, 4.4215866923e-01, 9.9310833216e-01],
                     [9.7149753571e-01, 1.6166780889e-01, 4.9212074280e-01],
                     [9.5944166183e-01, 8.4046614170e-01, 7.2132086754e-01],
                     [5.8691406250e-01, 5.6182861328e-02, 5.2246093750e-01],
                     [8.2579880953e-02, 9.4084471464e-01, 2.0549263060e-01],
                     [1.1553517729e-01, 2.1210658550e-01, 1.1788304150e-01],
                     [4.6980851889e-01, 2.2281603515e-01, 1.0867613368e-02],
                     [9.7266381979e-01, 5.3244304657e-01, 5.9391903877e-01],
                     [8.0224609375e-01, 4.3334960938e-01, 6.8408203125e-01],
                     [2.1447753906e-01, 7.0947265625e-01, 2.8955078125e-01],
                     [5.2406495810e-01, 8.5504561663e-01, 1.6205923259e-01],
                     [4.1086502373e-02, 7.4663269520e-01, 3.9624564350e-02],
                     [8.5914999247e-01, 7.4591505527e-01, 8.6612296104e-01],
                     [6.8027511239e-02, 8.9719355106e-01, 9.7205889225e-01],
                     [8.2723546028e-01, 6.8450033665e-01, 2.9845061898e-01],
                     [4.4763185084e-02, 2.6976650953e-01, 2.7353847027e-01],
                     [6.4281487465e-01, 6.0297679156e-02, 6.2723553181e-01],
                     [9.1254723072e-01, 2.1385987103e-01, 7.3106712103e-01],
                     [2.7845403552e-01, 3.8024261594e-01, 1.6402004659e-01],
                     [9.6289062500e-01, 3.1372070312e-02, 5.5633544922e-02],
                     [9.8642903566e-01, 7.5861579180e-01, 7.7188807726e-01],
                     [4.8168945312e-01, 2.0532226562e-01, 4.1412353516e-02],
                     [3.5842752457e-01, 6.5620416403e-01, 5.9875172377e-01],
                     [5.8537209034e-01, 5.7362645864e-01, 2.3602442443e-01],
                     [5.3301304579e-01, 6.6364187002e-01, 4.2711904645e-01],
                     [2.1086220443e-01, 4.0333861113e-01, 5.3939002752e-01],
                     [3.1201171875e-01, 8.5644531250e-01, 3.8256835938e-01],
                     [4.5736637712e-01, 1.5913459659e-01, 1.7827452719e-01],
                     [8.0712890625e-01, 7.8759765625e-01, 7.1337890625e-01],
                     [1.9676418602e-01, 7.2809958458e-01, 1.1132143438e-01],
                     [7.2064656019e-01, 1.0750342906e-01, 6.1546295881e-01],
                     [4.4703534245e-01, 4.8267862201e-01, 9.7889667749e-01],
                     [2.7992996573e-01, 4.7136551142e-01, 7.0781284571e-01],
                     [7.6162773371e-01, 6.8140399456e-01, 8.9368093014e-01],
                     [8.4739392996e-01, 1.2361865491e-01, 1.8504954875e-01],
                     [8.4270191193e-01, 8.5421806574e-01, 4.5936396718e-01],
                     [2.4153715372e-01, 7.0795261860e-01, 3.2144334912e-01],
                     [1.9189886749e-01, 9.2384427786e-01, 3.2126605511e-01],
                     [9.5177054405e-01, 8.6955451965e-01, 2.2407647967e-01],
                     [3.7109375000e-01, 4.7949218750e-01, 1.7028808594e-01],
                     [8.3189778030e-02, 7.7819359303e-01, 1.3956315815e-01],
                     [2.9945096374e-01, 3.1623309851e-01, 1.3950139284e-01],
                     [3.3096640836e-03, 7.4997943640e-01, 1.9066387415e-01],
                     [9.8478186131e-01, 5.3015261889e-01, 3.2882216573e-01],
                     [3.6712646484e-02, 2.0141601562e-01, 8.1152343750e-01],
                     [2.9652997851e-01, 8.9324671030e-01, 9.7335726023e-01],
                     [5.8455336094e-01, 8.9113235474e-01, 2.4147939682e-01],
                     [5.9059154987e-01, 4.0072762966e-01, 8.2471191883e-01],
                     [4.2799562216e-01, 4.4075375795e-01, 4.5453503728e-01],
                     [5.6755518913e-01, 2.1146789193e-01, 4.2760309577e-01],
                     [1.7620272934e-02, 6.1894172430e-01, 5.3478997946e-01],
                     [8.2359939814e-01, 1.1132505536e-01, 4.6311455965e-01],
                     [4.3184870481e-01, 4.3125128746e-01, 1.9358025491e-01],
                     [8.3642387390e-01, 5.6369924545e-01, 2.1757785976e-01],
                     [4.7691112757e-01, 4.0925717354e-01, 7.5537300110e-01],
                     [8.5478454828e-01, 3.9195250720e-03, 7.3266834021e-02],
                     [4.5068359375e-01, 9.7998046875e-01, 7.9101562500e-01],
                     [4.3992498517e-01, 6.1334166676e-03, 4.5538577437e-01],
                     [5.3211569786e-01, 3.6125785112e-01, 2.6356112957e-01],
                     [9.9817019701e-01, 7.9196661711e-01, 2.3766398430e-01]],

            'OBSTACLE': [
                [2, 4, 2, 2],
                [2, 8, 2, 2],
                [2, 12, 2, 2],
                [6, 4, 2, 2],
                [6, 8, 2, 2],
                [6, 12, 2, 2],
                [10, 4, 2, 2],
                [10, 8, 2, 2],
                [10, 12, 2, 2]
            ],

            'STATION': [
                [14 / 16, 2 / 16],
                [1 / 2, 1 / 8],
                [7 / 8, 1 / 2],
                [1 / 8, 1 / 8],
                [7 / 8, 7 / 8]
            ],
            'CHANNEL': 3,

            'NUM_UAV': 2,
            'INIT_POSITION': (0, 14, 2),  # todo
            'MAX_ENERGY': 30.,  # must face the time of lack
            'NUM_ACTION': 2,  # 2
            'SAFE_ENERGY_RATE': 0.2,
            'RANGE': 1.0,
            'MAXDISTANCE': 1.,
            'COLLECTION_PROPORTION': 0.2,  # c speed
            'FILL_PROPORTION': 0.2,  # fill speed

            'WALL_REWARD': -1.,
            'VISIT': 1. / 1000.,
            'DATA_REWARD': 1.,
            'FILL_REWARD': 1.,
            'ALPHA': 1.,
            'BETA': 0.1,
            'EPSILON': 1e-4,
            'NORMALIZE': .1,
            'FACTOR': 0.1,
            'DiscreteToContinuous': [
                [-1.0, -1.0],
                [-1.0, -0.5],
                [-1.0, 0.0],
                [-1.0, 0.5],
                [-1.0, 1.0],
                [-0.5, -1.0],
                [-0.5, -0.5],
                [-0.5, 0.0],
                [-0.5, 0.5],
                [-0.5, 1.0],
                [0.0, -1.0],
                [0.0, -0.5],
                [0.0, 0.0],  # 重点关注no-op
                [0.0, 0.5],
                [0.0, 1.0],
                [0.5, -1.0],
                [0.5, -0.5],
                [0.5, 0.0],
                [0.5, 0.5],
                [0.5, 1.0],
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],

            ]
        }
        self.LOG = log
        if self.LOG is not None:
            self.time = log.time

    def log(self):
        if self.LOG is not None:
            self.LOG.log(self.V)
        else:
            pass