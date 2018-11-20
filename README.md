# PacmanQLearning

How to run -

x = number of training episodes

n = total number of episodes including training and testing


Epsilon Greedy :

    python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

    python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l mediumGrid


Metropolis :

    python pacman.py -p PacmanQAgentMetro -x 2000 -n 2010 -l smallGrid

    python pacman.py -p PacmanQAgentMetro -x 2000 -n 2010 -l mediumGrid

Feature Based :
    
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid

python pacman.py -p ApproximateQAgentMetro -x 2000 -n 2010 -l smallGrid

python pacman.py -p ApproximateQAgentMetro -x 2000 -n 2010 -l mediumGrid