import math
import random

print("vítejte u hry kámen, nůžky, papír. Vaším úkolem je porazit počítač.")

n = False
y = True
play = y


while play:
    go = 0
    go2 = 0
    while go != 1:
        player = input("Zadejte svůj výběr: kámen, nůžky, papír: ")
        computer = random.randint(1, 3)

        if player == "kámen" or player == "nůžky" or player == "papír":
            print("Zadali jste: ", player)
            go = 1
        else:
            print("Špatně zadaná hodnota. Zadejte prosím znovu.")

    #konverze1
    if player == "kámen":
        player = 1
    elif player == "nůžky":
        player = 2
    elif player == "papír":
        player = 3

    #konverze2
    def conversion2 (computer):
        if computer == 1:
            computer_choice = "kámen"
        elif computer == 2:
            computer_choice = "nůžky"
        elif computer == 3:
            computer_choice = "papír"

        print("Počítač zadal:", computer_choice)

    conversion2(computer)
    
    if player == computer:
        print("Remíza")
    elif player == 1 and computer == 2:
        print("Vyhrál jste")
    elif player == 2 and computer == 3:
        print("Vyhrál jste")
    elif player == 3 and computer == 1:
        print("Vyhrál jste")
    else:
        print("Prohrál jste")

    while go2 != 1:
        play = input("Chcete hrát znovu? y/n: ")
        if play == "y":
            play = y
            go2 = 1
        elif play == "n":
            play = n
            go2 = 1
        else:
            print("Špatně zadaná hodnota. Zadejte prosím y nebo n.")
else:
    print("Děkujeme za hru.")
    exit(1)
