###########################################
### GENETIC ALGORITHM
### CONTINUOUS PROBLEM
### THE HIMMELBLAU FUNCTION
### MIN ((x**2)+y-11)**2+(x+(y**2)-7)**2
###########################################
### Owner: Dana Bani-Hani
### Copyright © 2019
###########################################


import numpy as np
import random as rd
import time


###########################################################################
###########################################################################
###########################################################################

##########################
### DEFINING FUNCTION ####
##########################


################################################
### CALCULATING THE OBJECTIVE FUNCTION VALUE ###
################################################

# calculate fitness value for the chromosome of 0s and 1s
def objective_value(chromosome):  
    
    lb_x = -6 # lower bound for chromosome x
    ub_x = 6 # upper bound for chromosome x
    len_x = (len(chromosome)//2) # length of chromosome x
    lb_y = -6 # lower bound for chromosome y
    ub_y = 6 # upper bound for chromosome y
    len_y = (len(chromosome)//2) # length of chromosome y
    
    precision_x = (ub_x-lb_x)/((2**len_x)-1) # precision for decoding x
    precision_y = (ub_y-lb_y)/((2**len_y)-1) # precision for decoding y
    
    z = 0 # because we start at 2^0, in the formula
    t = 1 # because we start at the very last element of the vector [index -1]
    x_bit_sum = 0 # initiation (sum(bit)*2^i is 0 at first)
    for i in range(len(chromosome)//2):
        x_bit = chromosome[-t]*(2**z)
        x_bit_sum = x_bit_sum + x_bit
        t = t+1
        z = z+1   
    
    z = 0 # because we start at 2^0, in the formula
    t = 1 + (len(chromosome)//2) # e.g. [6,8,3,9] (if the first 2 are y, so index will be 1+2 = -3)
    y_bit_sum = 0 # initiation (sum(bit)*2^i is 0 at first)
    for j in range(len(chromosome)//2):
        y_bit = chromosome[-t]*(2**z)
        y_bit_sum = y_bit_sum + y_bit
        t = t+1
        z = z+1
    
    # the formulas to decode the chromosome of 0s and 1s to an actual number, the value of x or y
    decoded_x = (x_bit_sum*precision_x)+lb_x
    decoded_y = (y_bit_sum*precision_y)+lb_y
    
    # the himmelblau function
    # min ((x**2)+y-11)**2+(x+(y**2)-7)**2
    # objective function value for the decoded x and decoded y
    obj_function_value = ((decoded_x**2)+decoded_y-11)**2+(decoded_x+(decoded_y**2)-7)**2
    
    return decoded_x,decoded_y,obj_function_value # the defined function will return 3 values



#################################################
### SELECTING TWO PARENTS FROM THE POPULATION ###
### USING TOURNAMENT SELECTION METHOD ###########
#################################################

# finding 2 parents from the pool of solutions
# using the tournament selection method 
def find_parents_ts(all_solutions):
    
    # make an empty array to place the selected parents
    parents = np.empty((0,np.size(all_solutions,1)))
    
    for i in range(2): # do the process twice to get 2 parents
        
        # select 3 random parents from the pool of solutions you have
        
        # get 3 random integers
        indices_list = np.random.choice(len(all_solutions),3,replace=False)
        
        # get the 3 possible parents for selection
        posb_parent_1 = all_solutions[indices_list[0]]
        posb_parent_2 = all_solutions[indices_list[1]]
        posb_parent_3 = all_solutions[indices_list[2]]
        
        # get objective function value (fitness) for each possible parent
        # index no.2 because the objective_value function gives the fitness value at index no.2
        obj_func_parent_1 = objective_value(posb_parent_1)[2] # possible parent 1
        obj_func_parent_2 = objective_value(posb_parent_2)[2] # possible parent 2
        obj_func_parent_3 = objective_value(posb_parent_3)[2] # possible parent 3
        
        # find which parent is the best
        min_obj_func = min(obj_func_parent_1,obj_func_parent_2,obj_func_parent_3)
        
        if min_obj_func == obj_func_parent_1:
            selected_parent = posb_parent_1
        elif min_obj_func == obj_func_parent_2:
            selected_parent = posb_parent_2
        else:
            selected_parent = posb_parent_3
        
        # put the selected parent in the empty array we created above
        parents = np.vstack((parents,selected_parent))
        
    parent_1 = parents[0,:] # parent_1, first element in the array
    parent_2 = parents[1,:] # parent_2, second element in the array
    
    return parent_1,parent_2 # the defined function will return 2 arrays



####################################################
### CROSSOVER THE TWO PARENTS TO CREATE CHILDREN ###
####################################################

# crossover between the 2 parents to create 2 children
# functions inputs are parent_1, parent_2, and the probability you would like for crossover
# default probability of crossover is 1
def crossover(parent_1,parent_2,prob_crsvr=1):
    
    child_1 = np.empty((0,len(parent_1)))
    child_2 = np.empty((0,len(parent_2)))
    
    
    rand_num_to_crsvr_or_not = np.random.rand() # do we crossover or no???
    
    if rand_num_to_crsvr_or_not < prob_crsvr:
        index_1 = np.random.randint(0,len(parent_1))
        index_2 = np.random.randint(0,len(parent_2))
        
        # get different indices
        # to make sure you will crossover at least one gene
        while index_1 == index_2:
            index_2 = np.random.randint(0,len(parent_1))
        
        
        # IF THE INDEX FROM PARENT_1 COMES BEFORE PARENT_2
        # e.g. parent_1 = 0,1,>>1<<,1,0,0,1,0 --> index = 2
        # e.g. parent_2 = 0,0,1,0,0,1,>>1<<,1 --> index = 6
        if index_1 < index_2:
            
            
            ### FOR PARENT_1 ###
            ### FOR PARENT_1 ###
            ### FOR PARENT_1 ###
            
            # first_seg_parent_1 -->
            # for parent_1: the genes from the beginning of parent_1 to the
                    # beginning of the middle segment of parent_1
            first_seg_parent_1 = parent_1[:index_1]
            
            # middle segment; where the crossover will happen
            # for parent_1: the genes from the index chosen for parent_1 to
                    # the index chosen for parent_2
            mid_seg_parent_1 = parent_1[index_1:index_2+1]
            
            # last_seg_parent_1 -->
            # for parent_1: the genes from the end of the middle segment of
                    # parent_1 to the last gene of parent_1
            last_seg_parent_1 = parent_1[index_2+1:]
            
            
            ### FOR PARENT_2 ###
            ### FOR PARENT_2 ###
            ### FOR PARENT_2 ###
            
            # first_seg_parent_2 --> same as parent_1
            first_seg_parent_2 = parent_2[:index_1]
            
            # mid_seg_parent_2 --> same as parent_1
            mid_seg_parent_2 = parent_2[index_1:index_2+1]
            
            # last_seg_parent_2 --> same as parent_1
            last_seg_parent_2 = parent_2[index_2+1:]
            
            
            ### CREATING CHILD_1 ###
            ### CREATING CHILD_1 ###
            ### CREATING CHILD_1 ###
            
            # the first segmant from parent_1
            # plus the middle segment from parent_2
            # plus the last segment from parent_1
            child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                      last_seg_parent_1))
            
            
            ### CREATING CHILD_2 ###
            ### CREATING CHILD_2 ###
            ### CREATING CHILD_2 ###
            
            # the first segmant from parent_2
            # plus the middle segment from parent_1
            # plus the last segment from parent_2
            child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                      last_seg_parent_2))
        
        
        
        ### THE SAME PROCESS BUT INDEX FROM PARENT_2 COMES BEFORE PARENT_1
        # e.g. parent_1 = 0,0,1,0,0,1,>>1<<,1 --> index = 6
        # e.g. parent_2 = 0,1,>>1<<,1,0,0,1,0 --> index = 2
        else:
            
            first_seg_parent_1 = parent_1[:index_2]
            mid_seg_parent_1 = parent_1[index_2:index_1+1]
            last_seg_parent_1 = parent_1[index_1+1:]
            
            first_seg_parent_2 = parent_2[:index_2]
            mid_seg_parent_2 = parent_2[index_2:index_1+1]
            last_seg_parent_2 = parent_2[index_1+1:]
            
            
            child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                      last_seg_parent_1))
            child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                      last_seg_parent_2))
     
    # when we will not crossover
    # when rand_num_to_crsvr_or_not is NOT less (is greater) than prob_crsvr
    # when prob_crsvr == 1, then rand_num_to_crsvr_or_not will always be less
            # than prob_crsvr, so we will always crossover then
    else:
        child_1 = parent_1
        child_2 = parent_2
    
    return child_1,child_2 # the defined function will return 2 arrays



############################################################
### MUTATING THE TWO CHILDREN TO CREATE MUTATED CHILDREN ###
############################################################

# mutation for the 2 children
# functions inputs are child_1, child_2, and the probability you would like for mutation
# default probability of mutation is 0.2
def mutation(child_1,child_2,prob_mutation=0.2):
    
    # mutated_child_1
    mutated_child_1 = np.empty((0,len(child_1)))
      
    t = 0 # start at the very first index of child_1
    for i in child_1: # for each gene (index)
        
        rand_num_to_mutate_or_not_1 = np.random.rand() # do we mutate or no???
        
        # if the rand_num_to_mutate_or_not_1 is less that the probability of mutation
                # then we mutate at that given gene (index we are currently at)
        if rand_num_to_mutate_or_not_1 < prob_mutation:
            
            if child_1[t] == 0: # if we mutate, a 0 becomes a 1
                child_1[t] = 1
            
            else:
                child_1[t] = 0  # if we mutate, a 1 becomes a 0
            
            mutated_child_1 = child_1
            
            t = t+1
        
        else:
            mutated_child_1 = child_1
            
            t = t+1
    
       
    # mutated_child_2
    # same process as mutated_child_1
    mutated_child_2 = np.empty((0,len(child_2)))
    
    t = 0
    for i in child_2:
        
        rand_num_to_mutate_or_not_2 = np.random.rand() # prob. to mutate
        
        if rand_num_to_mutate_or_not_2 < prob_mutation:
            
            if child_2[t] == 0:
                child_2[t] = 1
           
            else:
                child_2[t] = 0
            
            mutated_child_2 = child_2
            
            t = t+1
        
        else:
            mutated_child_2 = child_2
            
            t = t+1
    
    return mutated_child_1,mutated_child_2 # the defined function will return 2 arrays



###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

#########################
### GENETIC ALGORITHM ###
#########################


# hyperparameters (user inputted parameters)
prob_crsvr = 1 # probablity of crossover
prob_mutation = 0.4 # probablity of mutation
population = 180 # population number
generations = 160 # generation number


# x and y decision variables' encoding
# 13 genes for x and 13 genes for y (arbitrary number)
x_y_string = np.array([0,1,0,0,0,1,0,0,1,0,1,1,1,
                       0,1,1,1,0,0,1,0,1,1,0,1,1]) # initial solution


# create an empty array to put initial population
pool_of_solutions = np.empty((0,len(x_y_string)))


# create an empty array to store a solution from each generation
# for each generation, we want to save the best solution in that generation
best_of_a_generation = np.empty((0,len(x_y_string)+1))


# shuffle the elements in the initial solution (vector)
# shuffle n times, where n is the no. of the desired population
for i in range(population):
    rd.shuffle(x_y_string)
    pool_of_solutions = np.vstack((pool_of_solutions,x_y_string))


# so now, pool_of_solutions, has n (population) chromosomes


start_time = time.time() # start time (timing purposes)

gen = 1 # we start at generation no.1 (tracking purposes)


for i in range(generations): # do it n (generation) times
    
    # an empty array for saving the new generation
    # at the beginning of each generation, the array should be empty
    # so that you put all the solutions created in a certain generation
    new_population = np.empty((0,len(x_y_string)))
    
    # an empty array for saving the new generation plus its obj func val
    new_population_with_obj_val = np.empty((0,len(x_y_string)+1))
    
    # an empty array for saving the best solution (chromosome)
    # for each generation
    sorted_best = np.empty((0,len(x_y_string)+1))
    
    print()
    print()
    print("--> Generation: #", gen) # tracking purposes
    
    
    family = 1 # we start at family no.1 (tracking purposes)
    
    
    for j in range(int(population/2)): # population/2 because each gives 2 parents
        
        print()
        print("--> Family: #", family) # tracking purposes
        
            
        # selecting 2 parents using tournament selection
        # "genf.find_parents_ts"[0] gives parent_1
        # "genf.find_parents_ts"[1] gives parent_2
        parent_1 = find_parents_ts(pool_of_solutions)[0]
        parent_2 = find_parents_ts(pool_of_solutions)[1]
        
        
        # crossover the 2 parents to get 2 children
        # "genf.crossover"[0] gives child_1
        # "genf.crossover"[1] gives child_2
        child_1 = crossover(parent_1,parent_2,
                               prob_crsvr=prob_crsvr)[0]
        child_2 = crossover(parent_1,parent_2,
                               prob_crsvr=prob_crsvr)[1]
        
        
        # mutating the 2 children to get 2 mutated children
        # "genf.mutation"[0] gives mutated_child_1
        # "genf.mutation"[1] gives mutated_child_2
        mutated_child_1 = mutation(child_1,child_2,
                                      prob_mutation=prob_mutation)[0]  
        mutated_child_2 = mutation(child_1,child_2,
                                      prob_mutation=prob_mutation)[1] 
        
        
        # getting the obj val (fitness value) for the 2 mutated children
        # "genf.objective_value"[2] gives obj val for the mutated child
        obj_val_mutated_child_1 = objective_value(mutated_child_1)[2]
        obj_val_mutated_child_2 = objective_value(mutated_child_2)[2]
        
        
        
        # track each mutant child and its obj val
        print()
        print("Obj Val for Mutated Child #1 at Generation #{} : {}".
              format(gen,obj_val_mutated_child_1))
        print("Obj Val for Mutated Child #2 at Generation #{} : {}".
              format(gen,obj_val_mutated_child_2))

        
        
        # for each mutated child, put its obj val next to it
        mutant_1_with_obj_val = np.hstack((obj_val_mutated_child_1,
                                               mutated_child_1)) # lines 103 and 111
        
        mutant_2_with_obj_val = np.hstack((obj_val_mutated_child_2,
                                               mutated_child_2)) # lines 105 and 112
        
        
        # we need to create the new population for the next generation
        # so for each family, we get 2 solutions
        # we keep on adding them till we are done with all the families in one generation
        # by the end of each generation, we should have the same number as the initial population
        # so this keeps on growing and growing
        # when it's a new generation, this array empties and we start the stacking process
        # and so on
        new_population = np.vstack((new_population,
                                    mutated_child_1,
                                    mutated_child_2))
        
        
        # same explanation as above, but we include the obj val for each solution as well
        new_population_with_obj_val = np.vstack((new_population_with_obj_val,
                                                 mutant_1_with_obj_val,
                                                 mutant_2_with_obj_val))
        
        
        # after getting 2 mutated children (solutions), we get another 2, and so on
        # until we have the same number of the intended population
        # then we go to the next generation and start over
        # since we ended up with 2 solutions, we move on to the next possible solutions
        family = family+1
        
    
    # we replace the initial (before) population with the new one (current generation)
    # this new pool of solutions becomes the starting population of the next generation
    pool_of_solutions = new_population
    
    
    # for each generation
    # we want to find the best solution in that generation
    # so we sort them based on index [0], which is the obj val
    sorted_best = np.array(sorted(new_population_with_obj_val,
                                               key=lambda x:x[0]))
    
    
    # since we sorted them from best to worst
    # the best in that generation would be the first solution in the array
    # so index [0] of the "sorted_best" array
    best_of_a_generation = np.vstack((best_of_a_generation,
                                      sorted_best[0]))
    
    
    # increase the counter of generations (tracking purposes)
    gen = gen+1       



end_time = time.time() # end time (timing purposes)



# for our very last generation, we have the last population
# for this array of last population (convergence), there is a best solution
# so we sort them from best to worst
sorted_last_population = np.array(sorted(new_population_with_obj_val,
                                         key=lambda x:x[0]))

sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                         key=lambda x:x[0]))


# since we sorted them from best to worst
# the best would be the first solution in the array
# so index [0] of the "sorted_last_population" array
best_string_convergence = sorted_last_population[0]

best_string_overall = sorted_best_of_a_generation[0]


print()
print()
print("------------------------------")
print()
print("Execution Time in Seconds:",end_time - start_time) # exec. time
print()
print("Final Solution (Convergence):",best_string_convergence[1:]) # final solution entire chromosome
print("Encoded X (Convergence):",best_string_convergence[1:14]) # final solution x chromosome
print("Encoded Y (Convergence):",best_string_convergence[14:]) # final solution y chromosome
print()
print("Final Solution (Best):",best_string_overall[1:]) # final solution entire chromosome
print("Encoded X (Best):",best_string_overall[1:14]) # final solution x chromosome
print("Encoded Y (Best):",best_string_overall[14:]) # final solution y chromosome

# to decode the x and y chromosomes to their real values
# the "genf.objective_value" function returns 3 things -->
# [0] is the x value
# [1] is the y value
# [2] is the obj val for the chromosome
# we use "best_string[1:]" because we don't want to include the first element [0]
# because it's the obj val, which is not a part of the actual chromosome
final_solution_convergence = objective_value(best_string_convergence[1:])

final_solution_overall = objective_value(best_string_overall[1:])



print()
print("Decoded X (Convergence):",round(final_solution_convergence[0],5)) # real value of x
print("Decoded Y (Convergence):",round(final_solution_convergence[1],5)) # real value of y
print("Obj Value - Convergence:",round(final_solution_convergence[2],5)) # obj val of final chromosome
print()
print("Decoded X (Best):",round(final_solution_overall[0],5)) # real value of x
print("Decoded Y (Best):",round(final_solution_overall[1],5)) # real value of y
print("Obj Value - Best in Generations:",round(final_solution_overall[2],5)) # obj val of final chromosome
print()
print("------------------------------")


