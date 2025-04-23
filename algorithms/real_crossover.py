
import numpy as np

def arithmetic_crossover(parents, crossover_rate):

    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1] if i+1 < len(parents) else parents[i]
        
        if np.random.rand() < crossover_rate:
            weight = np.random.random()
            child1 = weight * parent1 + (1 - weight) * parent2
            child2 = weight * parent2 + (1 - weight) * parent1
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    
    return np.array(offspring)

def linear_crossover(parents, crossover_rate):

    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1] if i+1 < len(parents) else parents[i]
        
        if np.random.rand() < crossover_rate:
            child1 = 0.5 * parent1 + 0.5 * parent2  
            child2 = 1.5 * parent1 - 0.5 * parent2  
            child3 = -0.5 * parent1 + 1.5 * parent2  
            
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    
    return np.array(offspring)

def blend_crossover_alpha(parents, crossover_rate, alpha=0.5):

    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1] if i+1 < len(parents) else parents[i]
        
        if np.random.rand() < crossover_rate:
            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent1)
            
            for j in range(len(parent1)):
                min_val = min(parent1[j], parent2[j])
                max_val = max(parent1[j], parent2[j])
                range_val = max_val - min_val
                
                lower_bound = min_val - alpha * range_val
                upper_bound = max_val + alpha * range_val
                
                child1[j] = lower_bound + np.random.random() * (upper_bound - lower_bound)
                child2[j] = lower_bound + np.random.random() * (upper_bound - lower_bound)
            
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    
    return np.array(offspring)

def blend_crossover_alpha_beta(parents, crossover_rate, alpha=0.5, beta=0.5):

    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1] if i+1 < len(parents) else parents[i]
        
        if np.random.rand() < crossover_rate:
            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent1)
            
            for j in range(len(parent1)):
                min_val = min(parent1[j], parent2[j])
                max_val = max(parent1[j], parent2[j])
                range_val = max_val - min_val
                
                lower_bound1 = min_val - alpha * range_val
                upper_bound1 = max_val + beta * range_val
                
                lower_bound2 = min_val - beta * range_val
                upper_bound2 = max_val + alpha * range_val
                
                child1[j] = lower_bound1 + np.random.random() * (upper_bound1 - lower_bound1)
                child2[j] = lower_bound2 + np.random.random() * (upper_bound2 - lower_bound2)
            
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    
    return np.array(offspring)

def average_crossover(parents, crossover_rate):

    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1] if i+1 < len(parents) else parents[i]
        
        if np.random.rand() < crossover_rate:
            child = (parent1 + parent2) / 2
            offspring.extend([child, child.copy()])
        else:
            offspring.extend([parent1, parent2])
    
    return np.array(offspring)