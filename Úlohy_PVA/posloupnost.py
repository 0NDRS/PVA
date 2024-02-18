def find_interval_pairs(sequence):
    n = len(sequence)
    if n < 2 or n > 2000:
        print("Chybný vstup")
        return

    interval_sums = {}

    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            try:
                current_sum += int(sequence[j])
            except ValueError:
                continue

            if j > i:
                interval = (i, j)
                if current_sum in interval_sums:
                    interval_sums[current_sum].append(interval)
                else:
                    interval_sums[current_sum] = [interval]

    pair_count = 0
    for key in interval_sums:
        intervals = interval_sums[key]
        if len(intervals) > 1:
            pair_count += len(intervals) * (len(intervals) - 1) // 2

    print(f"Počet dvojic intervalů se stejným součtem: {pair_count}")


if __name__ == "__main__":
    try:
        sequence = input("Zadejte číselnou posloupnost oddělenou mezerami: ").split()
        find_interval_pairs(sequence)
    except ValueError:
        print("Chyba při zpracování vstupu. Zadejte platná celá čísla.")
