import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_closest_airplanes(airplanes):
    closest_distance = float('inf')
    closest_pairs = []

    for i in range(len(airplanes)):
        for j in range(i + 1, len(airplanes)):
            x1, y1, name1 = airplanes[i]
            x2, y2, name2 = airplanes[j]
            
            dist = distance(x1, y1, x2, y2)
            
            if dist < closest_distance:
                closest_distance = dist
                closest_pairs = [(name1, name2)]
            elif dist == closest_distance:
                closest_pairs.append((name1, name2))

    return closest_distance, closest_pairs

def parse_input():
    airplanes = []
    
    try:
        while True:
            user_input = input("Zadejte souřadnice a název letadla (x,y:Jméno) nebo napište 'dost' pro ukončení: ")

            if user_input.lower() == 'dost':
                break

            try:
                coords, name = user_input.split(':')
                x, y = map(float, coords.split(','))
                airplanes.append((x, y, name.strip()))
            except ValueError:
                print("Nesprávný formát vstupních dat.")
    except EOFError:
        pass

    if len(airplanes) < 2:
        print("Je potřeba zadat alespoň dvě letadla.")
        exit(1)

    return airplanes

def main():
    airplanes = parse_input()

    if not airplanes:
        print("Nebyly zadány žádné letadla.")
        exit(1)

    closest_distance, closest_pairs = find_closest_airplanes(airplanes)

    print(f"Vzdálenost nejbližších letadel: {closest_distance}")
    print(f"Nalezených dvojic: {len(closest_pairs)}")

    for pair in closest_pairs:
        print(f"{pair[0]} - {pair[1]}")

if __name__ == "__main__":
    main()
