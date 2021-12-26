# DE.py
This the implementation of the DE on MLP for the UT Algo Project 2021. The DE is applied on optimizing the network structure on both : 

1. Number of Hidden layer
2. Numer of neurons for each layer

Reference : https://github.com/OUStudent/EvolutionaryComputation

## Class structure
The file consist of 2 classes - DNN and DE_MLP/

### DNN 
It is the indvidiual unit that use to build the MLP with a list supplied, the list lenght will be the number of layer and each individual element will be the number of neuron for each layer, for example :
[784 100 50 25 1] is represent a MLP with 5 layers ( 3 hidden, 1 input and 1 output), the input layer is with size 784, the 1st hidden layer got 100 neurons etc. and the output size is 1.

### DE_MLP
It is the class that for the Differenal Evo.
#### Summary on how the flow on the DE : 
1. Calculate the input dimenosion size from the trainingset.
2. It will generate the first population with size = initSize, each element is a list of config that use to build the DNN as mention above. Each of the list is generated randomly base on the parameter :  mindepth, maxdepth, minneuron and maxneuron. Those generated list will then prepend the input dimension and append output dimension.
3. The first generation (configuration) will use to do the initial training by fit() function per configuration.
4. The best loss for each configuration will stored in scores array.
5. DE Section : 
```
6.      For each generation until Maxiter  :  (generation level)
8.          for j in the size of population (configurations) :
9.              let parent = configuration[j]
10.              **prepare the unit vector**
11.              x1,xs[0],xs[1] randomly choose 3 config from the population
12.             perform mutation with those 3 config and product a unit vector
13.             perform crossover operation with parent and unit vector which output a child config
14.             best_loss = perform fit() with the child config
15.             if the best_loss is better than the parent's scorce
16.                 update scores[j] = best_loss
17.                 update the population[j] to the child config
```

The first two steps is done in the constructor (__init__) , and the rest is done in the run() function

#### constructor parameters : 
1. outdim : the output dimension , default 1
2. maxdepth : the maximum number of hidden layers (excluding input and output layer ) , default 70
3. mindepth : the minimum number of hidden layers (excluding input and output layer ) , default 5
4. minneuron : the minimumn neurous number for hidden layers, default 4
5. maxneuron : the maximumn neurous number for hidden layers, default 10
6. bsize     : the batch size for individual MLP for trainiing , default 10
7. epoch     : the number of epoch for individual MLP for training default 100
8. initSize  : the initial population size , defualt 20
9. maxiter   : the number of generation that will run , default 10
10. stopcount : early stop that used in training, if the loss is not better than the current best for <stopcount> of epoch, the training will stop and return the current best , default 3.
11. trainingset : the dataset for training , assume to be in the shape (N,C,H,W) , default None
12. validationSet : the dataset for testing, assume it is the same shape with output dim
13. tariningTarget : the target data for training , assume it is the same shape with output dim
14. validationTarget : the target data for testing , assume it is the same shape with output dim
15. crossover        : it is used to choose which crossover to use , 1 for crossovermean and else will be crossoverRandomSwap

#### fit function 
parameters : 
1. config : the list that represent the MLP structure
2. id_    : the ID of the config , for display usage
3. p      : no use...

The fit function is just a normal training/testing block on NN, but it have early stop implemented which if the loss is not getting better for <stopcount> epochs, the training will be stop and return the current best loss.

#### mutation_1_2_z function 

formula : x1 + beta * (xs[0]-xs[1])
parameters :
1. x1 : the target vector(config)  
2. xs : It contain 2 vectors, xs[0] and xs[1] 
3. beta : the haraparameter of DE 
4. debug : print out details information 

The mutation function follows the standard one, but for this project, it will do the mutation on 2 aspects : the number of hidden layers and the number of neurons. So the coding is separated into 2 parts :

A. Mutation on the number of layers :
1. It is compare on those input vectors (x1,xs[0],xs[1]) to get the minimum length (hidden layers number)
2. do mutation to get the new hidden layers number by x1.length + beta * (xs[0].length - xs[1].length)
3. compare the new hidden layers number with minimum length from step 1 , update the minimum with the smallest one (use in neuron mutation)
4. If the new hidden layers number == 0 , just set it to x1's lenght 
5. If the new hidden layers number < 0 , apply abs

B Mutation on the number of neurons : 
As those x1 , xs[0] and xs[1] is come with different lenght, so it is not possible to apply the mutation formula at once, so the mutation is separated into 2 phases : apply mutation formula from 0 up to minimum length , then do the rest one by one.
1. apply mutation on x1, xs[0] and xs[1] from 0 to minlenght
    x1[:newminlen] + beta * (xs[0][:newminlen] - xs[1][:newminlen])
2. for the rest, (the new hidden layers number - minlen) , it is better to explain by example ... :P  
                                      x1    : [15 12 11] (3 hidden layers) </br>
   xs[0] : [6 7 8 9 4] (5 hidden layers)</br>
   xs[1] : [8 2 4 5 5 1 11 12 13] (9 hidden layers)</br>
   New Target lenght :  (6 hidden layers) , minimum lenght : 3 (x1 is the shortest) </br>
   So, as said above ,during the first phase the mutation formula will be apply from 0 to 2 (index) which is :</br>
   - 2a. 15 + 0.5 * (6-8)</br>
   - 2b. 12 + 0.5 * (7-2)</br>
   - 2c. 11 + 0.5 * (8-4)</br>
  
which the unit vector is now [14 , 14.5 , 13 , __ , ___ ,___ ] , still got 3 to go.for the remaining, which is from 3 to 5 (index) (3 positions remain), we get the number one by one by perform : </br>

   - 2d. x1 is already used up , so generate a random number in range of minneuron to maxneuron , let it be a </br>
   - 2e. xs[0] still got items (9) , let it be b </br>
   - 2f .xs[1] still got items (5) , let it be c</br>
   - 2g. t = a + beta * (b -c)  : 15 + 0.5 * (9-5) = 17</br>
   - 2h. so the t will be fill in the position 3 -> [14 , 14.5 , 13 , 17 , ___ ,___ ]</br>
    continuous until it fill up those remaining slots.</br>
3. After the above steps, apply boundary restriction (max and min number of neuron , round off decimal) to the unit vector
4. Its Done ~~~ 

####  crossoverMean function
parameters :
1. parent : the parent vector (config)
2. u      : the unit vector (config generated from mutation)

The function also follow the standard one, but need to due with different in length between parent and unit vector, the fix is simple, just repeat the smaller size vector to match up with the longer one, then perform crossover.

####  crossoverRandomSwap function
parameters :
1. parent : the parent vector (config)
2. u      : the unit vector (config generated from mutation)

The function also follow the standard one, but need to due with different in length between parent and unit vector, the fix is simple, just repeat the smaller size vector to match up with the longer one, then perform crossover.
The crossover is done by randomly choosing one element from either parent or from unit vector to form the child.

####  run function
parameters :
1. beta : the parameter that used in mutation 

Please refer to the summary section on the flow from step 3 . 

####  runMP function ( not working , dont use..)
parameters :
1. beta : the parameter that used in mutation 
