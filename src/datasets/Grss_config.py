INV_OBJECT_LABEL ={
    0: "Stadium seats",
    1: "Healthy grass",
    2: "Stressed grass",
    3: "Artificial turf",
    4: "Evergreen trees",
    5: "Deciduous trees",
    6: "Bare earth",
    7: "Water",
    8: "Residential buildings",
    9: "Non-residential buildings",
    10: "Roads",
    11: "Sidewalks",
    12: "Crosswalks",
    13: "Major thoroughfares",
    14: "Highways",
    15: "Railways",
    16: "Paved parking lots",
    17: "Unpaved parking lots",
    18: "Cars",
    19: "Trains"
}

NEW_OBJ_LABEL={
    0: "grass",
    1: "buildings",
    2: "trees",
    3: "Roads",
    4: "Bare earth",
    5: "parking",
    6: "Cars"
}
SWITH_TABLE={
    0: -1,
    1: 0,
    2: 0,
    3: -1,
    4: 2,
    5: 2,
    6:4,
    7:-1,
    8:1,
    9:1,
    10:3,
    11:3,
    12:3,
    13:3,
    14:3,
    15:3,
    16:5,
    17:5,
    18:6,
    19:-1,
    20:-1,
}

def swith(points):
    for i, point in enumerate(points):
        if point[-1] == -1:
            continue
        points[i,-1] = SWITH_TABLE[int(point[-1])]
    return points

NEW_CLASS_NAME = [NEW_OBJ_LABEL[i] for i in range(len(NEW_OBJ_LABEL))] + ['ignored']

CLASS_NAME = [INV_OBJECT_LABEL[i] for i in range(len(INV_OBJECT_LABEL))] + ['ignored']
NUM_BLOCKS = 16