# DE.py
This the implementation of the DE on MLP for the UT Algo Project 2021. The DE is applied on optimizing the network structure on both : 

1. The number of Hidden layers
2. The number of neurons for each layer

Reference : https://github.com/OUStudent/EvolutionaryComputation

## Class structure
The py-file consists of 2 classes - DNN and DE_MLP

### DNN 
It is the indvidiual unit that is used to build the MLP with a list supplied. The list lenght will be the number of layers and each individual element will represent the number of neurons for each layer, f.e:
`[784 100 50 25 1]` represents a MLP with 5 layers (3 hidden, 1 input and 1 output), the input layer is with size of 784 neurons, the 1st hidden layer has got 100 neurons etc. and the output size is 1.

### DE_MLP
It is the class for the Differential Evolution.
#### Summary on how the flow on the DE: 
1. Calculate the input dimension size from the trainingset.
2. It will generate the first population with size = `initSize`. Each element is a list of config that is used to build the DNN as mentioned above. Each of the list is generated randomly based on the parameters:  `mindepth`, `maxdepth`, `minneuron` and `maxneuron`. Those generated lists will be then prepended with the input dimension and appended with the output dimension.
3. The first generation (configuration) will be used to do the initial training by `fit()` function per configuration.
4. The best loss of each configuration will be stored in `scores` array.
5. DE Section: 
```
6.      For each generation until Maxiter :  (generation level)
8.          for j in the size of population (configurations) :
9.              let parent = configuration[j]
10.              **prepare the unit vector**
11.              x1,xs[0],xs[1] randomly choosed 3 config from the population
12.             perform mutation with those 3 config and product a unit vector
13.             perform crossover operation with parent and unit vector which output a child config
14.             best_loss = perform fit() with the child config
15.             if the best_loss is better than the parent's scorce
16.                 update scores[j] = best_loss
17.                 update the population[j] to the child config
```

The first two steps is done in the constructor `(__init__)` , and the rest is done in the `run()` function

#### constructor parameters : 
1. `outdim` : the output dimension size, default 1
2. `maxdepth` : the maximum number of hidden layers (excluding input and output layer) , default 70
3. `mindepth` : the minimum number of hidden layers (excluding input and output layer) , default 5
4. `minneuron` : the minimum number of neurons for each hidden layer, default 4
5. `maxneuron` : the maximumn number of neurons for each hidden layer, default 10
6. `bsize`     : the batch size of individual MLP for training , default 10
7. `epoch`     : the number of epochs for individual MLP for training, default 100
8. `initSize`  : the initial population size , default 20
9. `maxiter`   : the number of generation that will run , default 10
10. `stopcount` : early stop that is used in training, if the loss is not better than the current best for `stopcount` of epoch, the training will stop and return the current best, default 3
11. `trainingset` : the dataset for training, assumed to be in the shape (N,C,H,W), default None
12. `validationSet` : the dataset for testing, assumed it is the same shape with `output dim`
13. `trainingTarget` : the target data for training, assumed it is the same shape with `output dim`
14. `validationTarget` : the target data for testing, assumed it is the same shape with `output dim`
15. `crossover`        : it is choosed which crossover to use, if 1 then `crossoverMean` and else `crossoverRandomSwap`

#### fit function 
parameters : 
1. `config` : the list that represents the MLP structure
2. `id_`    : the ID of the config, for display usage
3. `p`      : no use...

The fit function is just a normal training/testing block on NN, but it has early stopping implemented: if the loss is not getting better for `stopcount` epochs, the training will be stopped and the current best loss is returned.

#### mutation_1_2_z function 

formula : `x1 + beta * (xs[0]-xs[1])`
parameters :
1. `x1` : the target vector(config)  
2. `xs` : It contain 2 vectors, xs[0] and xs[1] for difference vector
3. `beta` : the hyperparameter of DE 
4. `debug` : print out details information 

The mutation function follows the standard one, but for this project, it will do the mutation on 2 aspects: the number of hidden layers and the number of neurons. So the coding is separated into 2 parts:

A. Mutation on the number of layers:
1. It is compared on those input vectors (`x1,xs[0],xs[1]`) to get the minimum length (hidden layers number)
2. The mutation is done to get the new hidden layers number by `x1.length + beta * (xs[0].length - xs[1].length)`
3. Compare the new hidden layers number with minimum length from step 1, update the minimum with the smallest one (used in neuron mutation)
4. If the new hidden layers number == 0, it is set to `x1`'s lenght 
5. If the new hidden layers number < 0, it is applied the absoute value

B. Mutation on the number of neurons : 
As vectors `x1`, `xs[0]` and `xs[1]` may come with different lenght, it is not possible to apply the mutation formula at once. So, the mutation is separated into 2 phases: apply mutation formula from 0 up to minimum length, then do the rest one by one.
1. apply mutation on `x1`, `xs[0]` and `xs[1]` from 0 to `minlenght`
    `x1[:newminlen] + beta * (xs[0][:newminlen] - xs[1][:newminlen])`
2. for the rest, (the new hidden layers number - minlen) , it is better to explain by example ... :P  
                                      `x1`    : `[15 12 11]` (3 hidden layers) </br>
   `xs[0]` : `[6 7 8 9 4]` (5 hidden layers)</br>
   `xs[1]` : `[8 2 4 5 5 1 11 12 13]` (9 hidden layers)</br>
   New Target lenght :  6 hidden layers; minimum lenght : 3 (`x1` is the shortest) </br>
   So, as said above, during the first phase the mutation formula will be applied from 0 to 2 (index) which is :</br>
   - 2a. 15 + 0.5 * (6-8)</br>
   - 2b. 12 + 0.5 * (7-2)</br>
   - 2c. 11 + 0.5 * (8-4)</br>
  
The unit vector is now `[14, 14.5, 13, __ , __ ,__ ]` , still got 3 to go. For the remaining, indexes from 3 to 5 (3 positions remain), we get the numbers one by one as follows: </br>

   - 2d. `x1` is already used up, so generate a random number in range of `minneuron` to `maxneuron`, let it be `a` </br>
   - 2e. `xs[0]` still got items (9), let it be `b` </br>
   - 2f. `xs[1]` still got items (5), let it be `c` </br>
   - 2g. `t = a + beta * (b -c)`: 15 + 0.5 * (9-5) = 17</br>
   - 2h. so the `t` will fill the position 3 -> `[14, 14.5, 13, 17, __ ,__ ]`</br>
    it is continued until the remaining slots are filled up.</br>
1. After the above steps, apply boundary restriction (max and min number of neuron, round off decimal) to the unit vector
2. It's Done ~~~ 

####  crossoverMean function
parameters :
1. `parent`: the parent vector (config)
2. `u`      : the unit vector (config generated from mutation)

The function also follows the standard one, but needs to deal with the difference in length between the parent and the unit vector. The fix is simple, just repeat the vector of smaller size to match up with the longer one, then perform crossover, taking the mean over relevant elements. Child has the same lenght as the longer vector.

####  crossoverRandomSwap function
parameters :
1. `parent` : the parent vector (config)
2. `u`      : the unit vector (config generated from mutation)

This function also follows the standard one, but need to deals with the difference in length between the parent and the unit vector. The fix is simple, just repeat the vector of smaller size to match up with the longer one, then perform crossover.
The crossover is done by randomly choosing one element either from the parent vector or from the unit vector to form the child. Child has the same lenght as the longer vector.

####  run function
parameters :
1. `beta`: the parameter that used in mutation 

Please, refer to the summary section on the flow from step 3. 

####  runMP function ( not working , dont use..)
parameters :
1. `beta` : the parameter that used in mutation 

# Comparison to Baselines

One can find the baselines for hyperparamater tuning under the notebook named as algorithmics-optuna-baselines. We used Random Search and Bayesian Search for obtaining baselines, while expecting the Random Search to be the ultimate baseline because of it being most primitive among all the search methods we used.

In short, DE is clearly superior to random search with best DE variant scoring ~0.05 validation loss compared to random search's validation loss of 2.30. DE obtained lower validation loss than bayesian search too, with the latter baseline having a loss value of 0.58 on the validation set. 

Yet, in terms of runtime; neural architecture search through DE took more time to finish, which was around 100 minutes. Compared to this value, the best peforming  baseline, bayesian search, only required 25 minutes to achieve a loss of 0.58. Increasing the number of iterations for the bayesian search would have produced a more competitive result to the JDE variant of differential evolution.

# DE with VAE for Kernel Size on the first CNN layer(the closest to input layer) as feature selection ? 
It is a follow up investigation on the UT CNS project(https://medium.com/decoding-the-human-brain/decoding-the-human-brain-b4804740df47). Afte the presentation and comment by the lecturer, decoding brain signal from EEG/MEG is really a tough task for neurosciense as the the accuray of the model is heavily depending on  feature engineering in compare to which modelling method is used. For example on the MEG data set after FFT transform, the range of frequency selection and segmentation is already a big problem, and in additional to that, spatial information of those sensors are also under consideration which make the feature into a spatial-freqency domain. However, as the experiement is adopted with CNN, I was wondering why the CNN doesn't capture/perform well on this type of input, and after followup testing with different kernel size of the first layer (rectangle shape, large kernel (36x60) etc.), it does got some effect on the reconstruction loss. So, in result, I try to adopt the DE method to see if it would help on it problem. The experiement is still on going. 
 
Coding under folder : DE_VAE_FirstlayerOnly
1. mainDE_VAE.py - serial processing on the ordinary DE method. 
2. DE_VAE_MPI.py - distribute version with MPI implementation on the ordinary DE method.

Further thinking on distribute version with MPI ?
During the implementation of distribute version with MPI, it is found that MPI is overkilling. Actually the whole DE schema can be develop with a much ligher architecture such as web API with nodejs. May be will do it during free time.

 


