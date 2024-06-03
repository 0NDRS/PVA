using System;
using System.Collections.Generic;

public class Player
{
    public string Name { get; private set; }
    public int Score { get; private set; }

    public Player(string name)
    {
        Name = name;
        Score = 0;
    }

    public void RollDice(Random random)
    {
        int dice1 = random.Next(1, 7);
        int dice2 = random.Next(1, 7);

        Console.WriteLine($"{Name} rolled: {dice1} and {dice2}");

        if (dice1 == 1 && dice2 == 1)
        {
            Score = 0;
            Console.WriteLine($"{Name} rolled double ones and lost all points!");
        }
        else if (dice1 == 1 || dice2 == 1)
        {
            Console.WriteLine($"{Name} rolled a one. Turn ends with no points.");
        }
        else
        {
            Score += dice1 + dice2;
            Console.WriteLine($"{Name} scored {dice1 + dice2} points this turn.");
        }
    }

    public void UpdateScore(int points)
    {
        Score += points;
    }
}

public class Dice
{
    public int Roll(Random random)
    {
        return random.Next(1, 7);
    }
}

public class Game
{
    private List<Player> players = new List<Player>();
    private Dice dice = new Dice();
    private Random random = new Random();

    public void AddPlayer(string name)
    {
        players.Add(new Player(name));
    }

    public void Play()
    {
        bool gameOver = false;

        while (!gameOver)
        {
            foreach (var player in players)
            {
                Console.WriteLine($"It's {player.Name}'s turn. Current score: {player.Score}");
                Console.WriteLine("Press Enter to roll the dice...");
                Console.ReadLine();

                player.RollDice(random);

                Console.WriteLine("Scores after this round:");
                foreach (var p in players)
                {
                    Console.WriteLine($"{p.Name}: {p.Score}");
                }
                Console.WriteLine();

                if (player.Score >= 100)
                {
                    gameOver = true;
                    Console.WriteLine($"{player.Name} wins with {player.Score} points!");
                    break;
                }
            }
        }
    }
}

    class Program
{
    static void Main(string[] args)
    {
        Game game = new Game();

        Console.WriteLine("Welcome to Dice Game!");

        Console.Write("Enter number of players (2-5): ");
        int numPlayers = int.Parse(Console.ReadLine());

        for (int i = 1; i <= numPlayers; i++)
        {
            Console.Write($"Enter name for Player {i}: ");
            string playerName = Console.ReadLine();
            game.AddPlayer(playerName);
        }

        Console.WriteLine("Let's start the game!");
        game.Play();

        Console.WriteLine("Press any key to exit...");
        Console.ReadKey();
    }
}
