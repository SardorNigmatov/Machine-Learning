#Genetik qidiruv algoritmi

import random

def fitness(xromosoma,target):
    return abs(sum(xromosoma) - target) # har bir yechimning yaxshilik darajasini hisoblayapmiz


def crossover(parent1,parent2): # ota-onadan  yangi yechim  hosil qilyapmiz
    index = int(len(parent1)/2)
    child1 = parent1[:index] + parent2[index:]
    child2 = parent2[:index] + parent1[index:]
    return child1, child2

def mutate(xromosoma):
    index = random.randint(0,len(xromosoma) - 1) # biz o'zgartirayotgan genning indexi
    xromosoma[index] = random.randint(min(xromosoma),max(xromosoma)) #yangi qiymat
    return xromosoma


def genetic_algorithm(target,population_size,num_generations):
    
    population = [[random.randint(0,target) for i in range(4)] for i in range(population_size)] # Populatsiyani yaratib oldik
    
    for generation in range(num_generations):
        
        scores = [fitness(xromosoma,target) for xromosoma in population] # fitnes funksiyani hisoblayapmiz
        
        best_xromosoma = [population[i] for i in sorted(range(len(scores)),key=lambda i : scores[i])[:2]] # eng yaxshi yechimni tanlayapmiz
        
        population = [crossover(best_xromosoma[0],best_xromosoma[1])[i%2] for i in range(population_size)] # yangi yechimni kroslash
        
        population = [mutate(xromosoma) for xromosoma in population] # mutatsiya jarayoni
        
        scores = [fitness(xromosoma,target) for xromosoma in population]
        
        best_xromosoma = population[scores.index(min(scores))]
        
        return best_xromosoma # eng yaxshi yechimni qaytariyapmiz

target = 100

population_size = 50

num_generations = 200

best_xromosoma = genetic_algorithm(target, population_size, num_generations)

print('Eng yaxshi xromosoma:', best_xromosoma)

        