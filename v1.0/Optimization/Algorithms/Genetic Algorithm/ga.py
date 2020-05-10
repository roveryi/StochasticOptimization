import numpy as np

def cal_pop_fitness(pop):
    # Calculating the fitness value of each individual in the current population.
    fitness = np.zeros([pop.shape[0],1])
    for i in range(pop.shape[0]):
        fitness[i,0] = -np.sum(pop[i]**2)
    return fitness

class GA(object):
    def __init__(self,num_individual, x_bound, cal_pop_fitness, top_percentage, num_offspring, mutation_prob, num_iteration):
        self.ini_individual = num_individual
        self.x_bound = x_bound
        self.eval_fitness = cal_pop_fitness
        self.offspring_size = [num_offspring, x_bound.shape[1]]
        self.mutation_prob = mutation_prob
        self.num_iter = num_iteration
        self.top_percentage = top_percentage

    def Optimizer(self):
        pop = self.create_pop(self.ini_individual, self.x_bound)
        hist = []
        best_individual = []
        for iter in range(self.num_iter):
            current_fitness = self.eval_fitness(pop)
            mating_pool = self.select_mating_pool(pop, current_fitness, self.top_percentage)
            offspring_crossover = self.crossover(mating_pool, self.offspring_size)
            offspring_mutation = self.mutation(offspring_crossover, self.x_bound, self.mutation_prob)
            pop = offspring_mutation
            hist.append(np.max(current_fitness))

            print('It is %i th iteration, the current optimal fitness is %.4f'%(iter+1, np.max(current_fitness)))

        max_fitness_idx = np.where(current_fitness == np.max(current_fitness))
        max_fitness_idx = max_fitness_idx[0][0]

        return max(current_fitness), pop[max_fitness_idx, :], hist

    def create_pop(self, num_individual, x_bound):
        # This function randomly generates population
        return np.random.uniform(x_bound[0], x_bound[1], size = (num_individual, x_bound.shape[1]))

    def select_mating_pool(self,pop, fitness, top_percentage):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        # The best individuals are defined as the individuals having the highest fitness.
        num_parents = int(pop.shape[0]*top_percentage)
        parents = np.empty((num_parents, pop.shape[1]))

        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = float('-inf')
        return parents   

    def crossover(self,parents, offspring_size):
        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = int(offspring_size[1]/2)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return np.array(offspring)

    def mutation(self, offspring_crossover, x_bound, mutation_prob):
        # Each gene can mutate by specified mutation probability 
        mask_prob = np.random.uniform(0,1,size=offspring_crossover.shape) < mutation_prob
        mutate_magnitude = np.random.uniform(-1.0, 1.0, size=offspring_crossover.shape)
        mask = np.multiply(mask_prob, mutate_magnitude)

        offspring_crossover = np.multiply(offspring_crossover, mask + 1)

        return np.clip(offspring_crossover, x_bound[0], x_bound[1])  