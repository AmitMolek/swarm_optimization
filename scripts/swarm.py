import numpy as np
import sys
import matplotlib.pyplot as plt
from dataclasses import dataclass

class swarm_opt:
    @dataclass
    class init_params:
        SWARM_SIZE: np.uint32 = 100
        MAX_POSITION: np.uint32 = 1
        MAX_VELOCITY: np.uint32 = 1


    @dataclass
    class swarm_params:
        W: float = 1.0
        C1: float = 0.5
        C2: float = 0.5


    def __init__(self, init_p: init_params, swarm_p: swarm_params, interest_point: np.array) -> None:
        self.__init_p = init_p
        self.__swarm_p = swarm_p

        # The particles position, initializes uniformly between [0, MAX_POSITION)
        self.__positions = np.random.uniform(low=0, high=init_p.MAX_POSITION, size=(init_p.SWARM_SIZE, 2))
        # The particles velocity, initializes uniformly between [0, MAX_VELOCITY)
        self.__velocities = np.random.uniform(low=0, high=init_p.MAX_VELOCITY, size=(init_p.SWARM_SIZE, 2))
        # Holds the best personal fitness for each particle
        self.__personal_best_fitness = np.full((init_p.SWARM_SIZE, ), sys.maxsize, dtype=np.float64)
        # Holds the best personal position for each particle
        self.__personal_best = np.full((init_p.SWARM_SIZE, 2), sys.maxsize, dtype=np.float64)
        # Holds the global best position
        self.__global_best = np.full((2,), sys.maxsize, dtype=np.float64)
        # Holds the global best fitness
        self.__global_best_fitness = sys.float_info.max
        # The point we want the particles to go to
        self.__interest_point = interest_point
        # Generation counter
        self.__generation = 0


    def __particles_fitness(self):
        #dist_arr = np.full((SWARM_SIZE, 2), dst)
        #return np.array([np.hypot(p[0] - dst[0], p[1] - dst[1]) for p in positions])
        return np.array([np.linalg.norm(p - self.__interest_point) for p in self.__positions])


    def iterate(self) -> np.array:
        self.__generation += 1

        # Calculate fitness
        fitness = self.__particles_fitness()

        # Update global fitness
        min_fitness_index = np.argmin(fitness)
        min_fitness = fitness[min_fitness_index]

        if min_fitness < self.__global_best_fitness:
            self.__global_best_fitness = min_fitness
            self.__global_best = self.__positions[min_fitness_index]

        #print(f"Global best {global_best} Global best fitness = {global_best_fitness}")

        # Update personal fitness
        personal_to_update = (fitness < self.__personal_best_fitness).nonzero()[0]
        self.__personal_best_fitness[personal_to_update] = fitness[personal_to_update]
        self.__personal_best[personal_to_update] = self.__positions[personal_to_update]

        # v(t+1) = W*v(t) + c1*rand(0,1)*(p_best-x(t)) + c2*rand(0,1)*(g_best-x(t))
        new_velocities = np.array([ \
            self.__swarm_p.W * self.__velocities[i] + \
            self.__swarm_p.C1 * np.random.rand() * (self.__personal_best[i] - x) + \
            self.__swarm_p.C2 * np.random.rand() * (self.__global_best - x) \
            for i, x in enumerate(self.__positions) \
            ])

        self.__velocities = new_velocities
        self.__positions = self.__positions + new_velocities

        return self.__positions, self.__generation


init_params = swarm_opt.init_params(100, 5, 5)
swarm_params = swarm_opt.swarm_params(0.7, 0.4, 0.6)
interest_point = np.array([3, 3])
swarm = swarm_opt(init_params, swarm_params, interest_point)

while True:
    positions, generation = swarm.iterate()
    
    plt.scatter(positions[:,0], positions[:,1])
    plt.scatter(interest_point[0], interest_point[1])
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.pause(0.05)
    plt.clf()