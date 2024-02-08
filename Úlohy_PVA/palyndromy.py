from_num = int(input("Zadejte číslo: "))
radix = int(input("Zadejte číselnou řadu: "))
next_num = [0]

def is_palindrome(num, radix):
    original = num
    reversed_num = 0

    while num > 0:
        reversed_num = reversed_num * radix + num % radix
        num //= radix

    return original == reversed_num


def next_palindrome(from_num, radix, next_num):
    if radix < 2 or radix > 36:
        return 0 

    max_value = 2**64 - 1 
    if from_num == max_value:
        return 0 

    current_num = from_num + 1
    while current_num <= max_value:
        if is_palindrome(current_num, radix):
            next_num[0] = current_num
            return 1

        current_num += 1

    return 0 


success = next_palindrome(from_num, radix, next_num)

if success:
    print(f"Nejmenší větší palindrom než {from_num} je {next_num[0]}.")
else:
    print("Chyba při hledání palindromu.")
