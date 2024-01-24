from decimal import Decimal, getcontext, InvalidOperation

getcontext().prec = 10 
TOLERANCE = 1e-10 

def validate_input(value):
    try:
        return Decimal(value)
    except (ValueError, InvalidOperation):
        raise ValueError("Neplatný input. Zadejte prosím platná desetinná čísla.")

def validate_coordinates_input(input_str):
    coordinates = input_str.split()
    if len(coordinates) != 2:
        raise ValueError("Neplatné formátování. Zadejte prosím dvě čísla oddělená mezerou.")
    return tuple(map(validate_input, coordinates))

def are_congruent(a, b, c):
    return a == b or b == c or c == a

def orientation(x1, y1, x2, y2, x3, y3):
    val = (y2 - y1) * (x3 - x2) - (x2 - x1) * (y3 - y2)
    if round(val, 10) == 0:
        return 0
    return 1 if val > 0 else 2

def check_collinear(x1, y1, x2, y2, x3, y3):
    return orientation(x1, y1, x2, y2, x3, y3) == 0

def find_middle_point_label(x1, y1, x2, y2, x3, y3):
    if check_collinear(x1, y1, x2, y2, x3, y3):
        middle_x = (x1 + x2 + x3) / 3
        middle_y = (y1 + y2 + y3) / 3
        middle_point = middle_x, middle_y

        if middle_point == (x1, y1):
            return 'A'
        elif middle_point == (x2, y2):
            return 'B'
        elif middle_point == (x3, y3):
            return 'C'
        else:
            distances = {
                'A': (middle_x - x1)**2 + (middle_y - y1)**2,
                'B': (middle_x - x2)**2 + (middle_y - y2)**2,
                'C': (middle_x - x3)**2 + (middle_y - y3)**2
            }
            return min(distances, key=distances.get)

    return None

def main():
    try:
        points = [validate_coordinates_input(input(f"Zadejte koordnace bodu {chr(65 + i)} (X Y): ")) for i in range(3)]

        if are_congruent(*points):
            print("Překrývají se body.")
            return

        if check_collinear(*points[0], *points[1], *points[2]):
            print("Body leží na stejné přímce.")

            middle_point_label = find_middle_point_label(*points[0], *points[1], *points[2])
            if middle_point_label:
                print(f"Prostřední bod je: {middle_point_label}")
            else:
                print("Není prostřední bod.")

        else:
            print("Body neleží na stejné přímce.")

    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
