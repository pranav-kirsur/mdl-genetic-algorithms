import numpy as np
import json
import requests
import pickle


######### DO NOT CHANGE ANYTHING IN THIS FILE ##################
API_ENDPOINT = 'http://10.4.21.147'
PORT = 3000
MAX_DEG = 11

# Use very wisely
SECRET_KEY = "6Fwiv8RlF6adiEr2CoqQZFRxGhg8XnQszfULpkQCGEXTzcjuNB"

# functions that you can call


def get_errors(id, vector):
    """
    returns python array of length 2 
    (train error and validation error)
    """
    for i in vector:
        assert -10 <= abs(i) <= 10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(id, vector, 'geterrors'))


def submit(id, vector):
    """
    used to make official submission of your weight vector
    returns string "successfully submitted" if properly submitted.
    """
    for i in vector:
        assert -10 <= abs(i) <= 10
    assert len(vector) == MAX_DEG
    return send_request(id, vector, 'submit')

# utility functions


def urljoin(root, port, path=''):
    root = root + ':' + str(port)
    if path:
        root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root


def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, PORT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={'id': id, 'vector': vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response


# Genetic algo parameters
POPULATION_SIZE = 16
MATING_POOL_SIZE = 8


# Size of the population
population_size = (POPULATION_SIZE, 11)

init_array = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -
              6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]



def get_fitness(population):
    fitness_array = np.ones((POPULATION_SIZE, 1))
    for member_index in range(POPULATION_SIZE):
        err_arr = get_errors(SECRET_KEY, list(population[member_index, :]))
        fitness_array[member_index] = (0.5 * err_arr[1]) + (0.5 * err_arr[0])
    return fitness_array


def get_mating_pool(population, fitness):
    pop_fit = np.concatenate((population, fitness), axis=1)
    pop_fit = pop_fit[np.argsort(pop_fit[:, 11])]

    # Probablities of selection
    probablities = np.arange(1.0, POPULATION_SIZE+1, 1.0)

    probablities = np.reciprocal(probablities)
    probablities = probablities / np.sum(probablities)
    mating_pool = pop_fit[np.random.choice(
        POPULATION_SIZE, size=MATING_POOL_SIZE, replace=False, p=probablities)]

    # np.random.shuffle(mating_pool)
    return mating_pool


def crossover(parents, children_shape):
    children = np.zeros((children_shape[0], 12))

    # Index at which crossover happens
    crossover_index = 5

    for i in range(children_shape[0]):

        # Indices of the parents
        parent_1_index = i % parents.shape[0]
        parent_2_index = (i+1) % parents.shape[0]

        # Crossover for ith child
        children[i, 0: crossover_index] = parents[parent_1_index,
                                                  0: crossover_index]
        children[i, crossover_index:] = parents[parent_2_index, crossover_index:]
        # w1 = parents[parent_1_index, 11] + 1
        # w2 = parents[parent_2_index, 11] + 1
        # children[i, 0:] = np.divide((np.multiply(
        #     parents[parent_1_index, 0:], w2) + np.multiply(parents[parent_2_index, 0:], w1)), w1 + w2)

    return children[:, 0:11]


def mutation(children):
    # random = np.random.uniform(low=0.5, high=1.5, size=children.shape)
    random = np.ones(children.shape)
    for i in range(children.shape[0]):
        if(np.random.uniform() < 0.5):
            random[i] = np.random.uniform(low=0.5, high=1.5, size=(1,children.shape[1]))
    mutated_children = np.multiply(children, random)
    mutated_children = np.clip(mutated_children, -10, 10)
    return mutated_children


def reproduce(x, y):
    crossover_point = np.random.choice(11)
    child = np.array(x)
    child[0:crossover_point] = x[0:crossover_point]
    child[crossover_point:] = y[crossover_point:]
    return child

def mutate(child):
    random = np.random.uniform(low=-2.0, high=2.0, size=child.shape)
    child = np.multiply(child, random)
    child = np.clip(child, -10.0, 10.0)
    return child




# Creating the initial population randomly
population = np.random.uniform(low=-10, high=10, size=population_size)

population = np.tile(np.array(init_array), (POPULATION_SIZE, 1))

population = pickle.load( open( "population.p", "rb" ) )
print(population)


# population[0, :] = init_array


# Number of generations
NUMBER_OF_GENERATIONS = 10

# Run genetic algo
for generation in range(NUMBER_OF_GENERATIONS):
    print()
    print("Generation :", generation)

    # Calculate fitness of the population
    fitness_of_population = get_fitness(population)

    reci_fitness_of_population = np.reciprocal(fitness_of_population)

    # compute probablity of picking according to the parent
    probablity = (np.transpose(reci_fitness_of_population)/ np.sum(reci_fitness_of_population))[0]

    # Submit the current best and print it
    mini = np.argmin(fitness_of_population)
    submit(SECRET_KEY, list(population[mini, :]))
    print("Current best:", np.min(fitness_of_population))
    print("Best vector:" , list(population[mini, :]))

    new_population = np.zeros(population.shape)
    for i in range(POPULATION_SIZE):
        x = population[np.random.choice(POPULATION_SIZE, p = probablity)]
        y = population[np.random.choice(POPULATION_SIZE, p = probablity)]

        child = reproduce(x, y)
        if(np.random.uniform() < 0.9):
            child = mutate(child)
        new_population[i] = child
    
    population = new_population


        


print(population)
pickle.dump(population, open("population.p", "wb"))
