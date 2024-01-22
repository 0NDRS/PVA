import math

# Validace bodů ve velké blízkosti stěny
def validate_point_close_to_wall(x, y, z, side):
    if 0 < x <= 20 or 0 < y <= 20 or 0 < z <= 20 or side - 20 <= x < side or side - 20 <= y < side or side - 20 <= z < side:
        print(f"Chyba: Bod s souřadnicemi ({x}, {y}, {z}) je příliš blízko stěně.")
        exit(1)

# Validace dat bodů
def validate_point_out_of_wall(x, y, z, side):
    if x < 0 or y < 0 or z < 0 or x > side or y > side or z > side or (x != side and y != side and z != side and x != 0 and y != 0 and z != 0):
        print(f"Chyba: Bod s souřadnicemi ({x}, {y}, {z}) má chybné údaje.")
        exit(1)


# Validace délky strany
try:
    side = int(input("Zadejte délku strany místnosti(a): "))
    if side <= 0:
        print("Chyba: Délka strany musí být kladné číslo.")
        exit(1)
except ValueError:
    print("Chyba: Zadejte prosím platné číslo.")
    exit(1)

try:
    point1_input = input("Zadejte koordinace prvního bodu (x y z): ")
    if not point1_input:
        print("Chyba: Neplatné hodnoty pro první bod.")
        exit(1)
    point1 = list(map(int, point1_input.split()))
    if len(point1) != 3:
        print("Chyba: Zadejte všechny tři souřadnice pro první bod.")
        exit(1)
    validate_point_close_to_wall(*point1, side)
    validate_point_out_of_wall(*point1, side)
except ValueError:
    print("Chyba: Neplatné hodnoty pro první bod.")
    exit(1)

# Validace druhého bodu
try:
    point2_input = input("Zadejte koordinace druhého bodu (x y z): ")
    if not point2_input:
        print("Chyba: Neplatné hodnoty pro druhý bod.")
        exit(1)
    point2 = list(map(int, point2_input.split()))
    if len(point2) != 3:
        print("Chyba: Zadejte všechny tři souřadnice pro druhý bod.")
        exit(1)
    validate_point_close_to_wall(*point2, side)
    validate_point_out_of_wall(*point2, side)
except ValueError:
    print("Chyba: Neplatné hodnoty pro druhý bod.")
    exit(1)
distances = [abs(a - b) for a, b in zip(point1, point2)]

def distance_to_edge(point, side):
    return [point[0], point[1], side - point[0], side - point[1]]

if side in distances:
    point1 = [i for i in point1 if (i != 0) and (i != side)]
    point2 = [i for i in point2 if (i != 0) and (i != side)]
    point1_to_edges = distance_to_edge(point1, side)
    point2_to_edges = distance_to_edge(point2, side)

    distances_pipes = []
    distances_hose = []

    for i in range(4):
        point1_to_edge = point1_to_edges[i]
        point2_to_edge = point2_to_edges[i]

        dist_between_points = [abs(a - b) for a, b in zip(point1, point2)]

        if i % 2 == 0:
            distances_pipes.append(point1_to_edge + point2_to_edge + side + dist_between_points[1])
            distances_hose.append(((point1_to_edge + point2_to_edge + side)**2 + dist_between_points[1]**2)**0.5)
        else:
            distances_pipes.append(point1_to_edge + point2_to_edge + side + dist_between_points[0])
            distances_hose.append(((point1_to_edge + point2_to_edge + side)**2 + dist_between_points[0]**2)**0.5)

    pipe_length = min(distances_pipes)
    hose_length = min(distances_hose)

else:
    pipe_length = sum(distances)

    axis = [i for i in range(3) if (point1[i] not in [0, side]) and (point2[i] not in [0, side])][0]
    remain = [distances[i] for i in range(3) if i != axis]

    hose_length = (sum(remain)**2 + distances[axis]**2)**0.5

print(f"Délka potrubí je: {pipe_length}")
print(f"Délka hadice je: {hose_length}")