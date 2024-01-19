import math

hRoom = int(input("Zadejte délku strany místnosti(a): "))
wRoom = hRoom
room = wRoom * hRoom

class fPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class sPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

try:
    x1, y1, z1 = map(int, input("Enter coordinates of the first point (x y z): ").split())
    x2, y2, z2 = map(int, input("Enter coordinates of the second point (x y z): ").split())
except ValueError:
    print("Invalid values for inputs.")
    exit(1)

if z1 == hRoom or z2 == hRoom:
    print("Point are in the opposite sides of the room. I can't calculate that yet.")
    exit(1)


def calculate_pipe_length():
    return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)

pipe_length = calculate_pipe_length()
print(f"Delka potrubi: {pipe_length}")
