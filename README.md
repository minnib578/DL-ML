# Linear Regression and Logistic Regression

#### 1)  What is linear regression? What is logistic regression? What is the difference?
* Linear regression is used to predict the continuous variable based on a given set of independent variables, and it is used for solving Regression problem.Least square estimation method is used for the estimation of accuracy.
example: House pricing

   
   ![image](https://user-images.githubusercontent.com/63558665/117905179-bcfada00-b2a0-11eb-826e-1daf125867b6.png)
   
   ![image](https://user-images.githubusercontent.com/63558665/117905255-e4ea3d80-b2a0-11eb-8f2b-97c830082280.png)

* Logistic Regression is used to predict the categorical variable using a given set of independent variables, and it is used for solving classification problem.Maximum likelihood estimation method is used for the estimation of accuracy.The output of Logistic Regression problem can be only between the 0 and 1.In logistic regression, we pass the weighted sum of inputs through an activation function that can map values in between 0 and 1. Such activation function is known as sigmoid function.

example:Spam Detection:Spam detection is a binary classification problem where we are given an email and we need to classify whether or not it is spam. If the email is spam, we label it 1; if it is not spam, we label it 0. 


#### 2)  what is gradient descent?
Gradient descent is an optimization algorithm that's used when training a machine learning model. It's based on a convex function and tweaks its parameters iteratively to   minimize a given function to its local minimum.

how does gradient descent work?
The equation below describes what gradient descent does: b is the next position of our climber, while a represents his current position. The minus sign refers to the minimization part of gradient descent. The gamma in the middle is a waiting factor(stepsize) and the gradient term ( Δf(a) ) is simply the direction of the steepest descent.

   ![image](https://user-images.githubusercontent.com/63558665/117905892-e9632600-b2a1-11eb-9a4d-6a7eb210113c.png)

So this formula basically tells us the next position we need to go, which is the direction of the steepest descent

challenges:
* Converge to local minimum
* If there are multiple local minima in the error surface, there is no guarantee that the procedure will find the global minimum
* Performance depends on the shape of the error surface. Too many valleys/wells will make it easy to be trapped in local minima

solutions:
* Try nets with different number of hidden layers and hidden nodes (they may lead to different error surfaces, some might be better than others)
* Try different initial weights (different starting points on the surface)
* Forced escape from local minima by random perturbation (e.g., simulated annealing)
* adding momentum term
* batch mode of weight update
* variations of learning rate
* adaptive learning rate


#### 3) how to choose the learning rate?
For gradient descent to reach the local minimum we must set the learning rate to an appropriate value, which is neither too low nor too high. This is important because if the steps it takes are too big, it may not reach the local minimum because it bounces back and forth between the convex function of gradient descent (see left image below). If we set the learning rate to a very small value, gradient descent will eventually reach the local minimum but that may take a while(see right image).

   ![image](https://user-images.githubusercontent.com/63558665/117906321-ace3fa00-b2a2-11eb-9767-71d24d1923c2.png)
   
if the loss of model is up and down interatively, it may be caused by high learning rate.
in order to choose a good learning, plotting the cost function as the optimization runs also is a method
chosing a learning rate:
* Try multiple choise,crude rule of thumb: update changes weight about 0.1-1%
* Linesearch: keep walking in the same direction as long as 𝑓 is still decreasing


#### 4)what is batch gradient descent, stochastic gradient descent, mini-batch gradient descent?
1. batch gradient descent calculates the error for each sample within the training dataset, but only after all training samples have been evaluated does the model get updated
     
     ![image](https://user-images.githubusercontent.com/63558665/117908665-d2730280-b2a6-11eb-8265-935f26e9e2d8.png)
     
     * advantage:
          computational efficient
          it produces a stable error gradient and a stable convergence
          Less oscillations and noisy steps taken towards the global minima of the loss function due to updating the parameters by computing the average of all the training samples rather than the value of a single sample
     * disadvantage:
          stable error gradient can lead to a local minimum.
          The entire training set can be too large to process in the memory due to which additional memory might be needed. 
          Depending on computer resources it can take too long for processing all the training samples as a batch
          
2. stochastic gradient descent: one training sample (example) is passed through the neural network at a time and the parameters (weights) of each layer are updated with the computed gradient

    ![image](https://user-images.githubusercontent.com/63558665/117910801-98a3fb00-b2aa-11eb-864b-de57abb9ff20.png)

    * advantage:
       It is easier to fit into memory due to a single training sample being processed by the network
       It is computationally fast as only one sample is processed at a time
       For larger datasets it can converge faster as it causes updates to the parameters more frequently
       Due to frequent updates the steps taken towards the minima of the loss function have oscillations which can help getting out of local minimums of the loss function
    * disadvantage:
       The minima is very noisy caused by frequent update the step taken towards
       It may take longer to achieve convergence to the minima of loss function
       Frequent update causes computation expensive due to using all resources for processing one training sample at a time
       It loses the advanatage of vetorized operations
       
 3. minni-batch gradient descent: Mini-batch gradient descent is the go-to method since it’s a combination of the concepts of SGD and batch gradient descent. It simply splits the training dataset into small batches and performs an update for each of those batches
 
    ![image](https://user-images.githubusercontent.com/63558665/117910758-80cc7700-b2aa-11eb-85e9-a93a486a046f.png)
    
    * advantage:
       Easily fits in the memory
       It is computationally efficient
       Benefit from vectorization
       If stuck in local minimums, some noisy steps can lead the way out of them
       Average of the training samples produces stable error gradients and convergence

#### 5) what is mean sqaured error, cross-entropy ? and what is the difference ?
* MSE measures the average of the squares of the errors
    
    ![image](https://user-images.githubusercontent.com/63558665/117914164-c3914d80-b2b0-11eb-936c-90862ce28ab3.png)
    
* Cross-entropy is a measure of the difference between two probability distributions for a given random variable or set of events.
    
    ![image](https://user-images.githubusercontent.com/63558665/117914617-b45ecf80-b2b1-11eb-8248-343d3f13f2d2.png)
    
    MSE is used for linear regression and cross-entropy is used for classification problems
    
#### 6) what is activation function? 
An activation function in a neural network defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network

* sigmoid used in logistic regression classification.The function takes any real value as input and outputs values in the range 0 to 1. The larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to 0.0. 
   sigmoid cause vanishing gradient, and the gradient will be zero when x is inf and -inf.

    ![image](https://user-images.githubusercontent.com/63558665/117915606-9b571e00-b2b3-11eb-8088-9833fa80ca75.png)
    
* Tanh: tanh is also like logistic sigmoid but better. The range of the tanh function is from (-1 to 1). tanh is also sigmoidal (s - shaped).The advantage is that The larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to -1.0.

    ![image](https://user-images.githubusercontent.com/63558665/117915997-54b5f380-b2b4-11eb-8634-8374ed7a6fea.png)
    
* ReLU: the ReLU is half rectified (from bottom). f(z) is zero when z is less than zero and f(z) is equal to z when z is above or equal to zero.
     
     * simple, easy to calcualte, gradient is 1, which can partially solve the gradient vanishing problem
     * some part of neural network will never update.All the negative values become zero immediately which decreases the ability of the model to fit or train from the data properly. That means any negative input given to the ReLU activation function turns the value into zero immediately in the graph, which in turns affects the resulting graph by not mapping the negative values appropriately.

     ![image](https://user-images.githubusercontent.com/63558665/117916165-99418f00-b2b4-11eb-84e1-7f389050b1ce.png)

* ELU:
     * avoid the dying ReLU
     * high computation cost

     ![image](https://user-images.githubusercontent.com/63558665/117920827-9e570c00-b2bd-11eb-8ab2-0918a2596b3e.png)

* Leaky ReLU:It is an attempt to solve the dying ReLU problem.The leak helps to increase the range of the ReLU function. Usually, the value of a is 0.01 or so.Therefore the range of the Leaky ReLU is (-infinity to infinity)

     ![image](https://user-images.githubusercontent.com/63558665/117916241-c42be300-b2b4-11eb-9c71-1424670f90fd.png)
     
* softmax function outputs a vector of values that sum to 1.0 that can be interpreted as probabilities of class membership

     ![image](https://user-images.githubusercontent.com/63558665/117916946-2507eb00-b2b6-11eb-8344-0865360bdd61.png)

     The activation function used in hidden layers is typically chosen based on the type of neural network architecture.
     ![image](https://user-images.githubusercontent.com/63558665/117916781-d0fd0680-b2b5-11eb-860d-c9c7367f4e1c.png)
     
     Output activation function:
     
     ![image](https://user-images.githubusercontent.com/63558665/117917100-66989600-b2b6-11eb-956c-806cd1c85ff0.png)
     
     
#### 7) Does global optimal can be reached by SGD, why？

when the learning rates decrease with an appropriate rate, and subject to relatively mild assumptions, stochastic gradient descent converges almost surely to a global minimum when the objective function is convex or pseudoconvex, and otherwise converges almost surely to a local minimum

#### 8) what is convex optimization and non-convex optimization?
* Non-convex: local minima and sensitive to initilization
* convex: global minima


#### 9) what vanishing/exploding gradient?

If each term is (much) greater than 1 -->explosion of gradients
If each term is (much) less than 1 -->vanishing gradients

solution for gradient vanishing:
* carefully initilization--want variance to remain approximate constant since variance increase in backward pass causes exploding gradient and variance decrease in backward causes vanishing gradient
     * "MSRA initialization": weights=Gaussiance with 0 mean and variance =2/(k*k*d)
     
* batch normalization:normalize so that each layer output has zero mean and unit variance
      * compute mean and variance for each channel
      * aggregate over batch
      * subtract mean, divide by std
      * no"batchs" during test,at test, The mean/std are not computed  based on the batch. Instead, a single  fixed empirical mean of activations  during training is used!
      * after Fully  Connected or Convolutional layers,  and before nonlinearity!
      * batch normalization for fully-connected network-->layer normalization
      
     ![image](https://user-images.githubusercontent.com/63558665/117922902-2f7bb200-b2c1-11eb-8c75-32717ca566f1.png)

     * why batch normalize?
            Improves gradient flow through  the network
            Allows higher learning rates
            Reduces the strong dependence on initialization
            Acts as a form of regularization in a funny way, and slightly  reduces the need for dropout
      
     ![image](https://user-images.githubusercontent.com/63558665/117922474-86cd5280-b2c0-11eb-8c58-5cbe6d707002.png)

 
* Residual connections:allow gradients to flow unimpeded, instead of single layers, have residual connections over block
      * doubling of feature channel
      * increase channel by 1x1 convolution
      * decrease spatial resoution by subsampling
      
     ![image](https://user-images.githubusercontent.com/63558665/117920511-f80b0680-b2bc-11eb-9ea0-bd53a72c971c.png)
     
* bottleneck blocks:use 1x1 to project to lower dimensionality do convolution, then come back-->avoid that when channels increases, 3x3 convolutions introduce many parameters.
      
     ![image](https://user-images.githubusercontent.com/63558665/117921786-56d17f80-b2bf-11eb-8c9b-930a6261fe5a.png)



# Regularization, Overfitting and Model/Feature Selection/Evaluation.

#### 1) what is regularization? what is L1 and L2 regularization? what is the difference between L1 ,L2 and Linear regression?
 regularization is A technique that discourages learning a more complex or flexible model, so as to avoid the risk of overfitting.
 Methods:
 Ridge (L2 norm)
 Lasso (L1 norm)
 Dropout:In each forward pass, randomly set some neurons to zero Probability of dropping is a hyperparameter; 0.5 iscommon.
 Batchnormalization:normalize so that each layer output has zero mean and unit variance
 Data augmentation
 early stop: train our model for an arbitrarily large number of epochs and plot the validation loss graph (e.g., using hold-out). Once the validation loss begins to degrade (e.g., stops decreasing but rather begins increasing), we stop the training and save the current model
 
 * Linear regression finds the parameters to minimize the mean squared error or residuals between the predictions and the targets.Overfitting occurs when the model makes much better predictions on known data than on unknown data. The model begins to memorize the training data and is unable to generalize to unseen test data. and then we need to simplefy the model by introducing the regularization L1 and L2
 * L1 regularization (Lasso regression) to force some coefficients to be exactly zero. This means some features are completely ignored by the model.If lambda is zero then we will get back OLS whereas very large value will make coefficients zero hence it will under-fit
 * L2 regularization (Ridge regression) adds “squared magnitude” of coefficient as penalty term to the loss function.if lambda is zero then you can imagine we get back OLS. However, If lambda is very large then it will add too much weight and it will lead to under-fitting.
    
    
    ![image](https://user-images.githubusercontent.com/63558665/117913152-d2770080-b2ae-11eb-9f7a-d73ad0156754.png)


    ![image](https://user-images.githubusercontent.com/63558665/117913034-9e9bdb00-b2ae-11eb-81f1-97763aa9163f.png)
    
The key difference between these techniques is that .
The obvious disadvantage of ridge regression, is model interpretability. It will shrink the coefficients for least important predictors, very close to zero. But it will never make them exactly zero. In other words, the final model will include all predictors. However, in the case of the lasso, the L1 penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero . Therefore,Lasso shrinks the less important feature’s coefficient to zero,removing some feature altogether when the tuning parameter λ is sufficiently large. So, this works well for feature selection in case we have a huge number of features and is said to yield sparse models.

#### 2) What is overfitting?
overfitting representes a model learns the patterns of a training dataset too well, perfectly explaining the training data set but failing to generalize its predictive power to other sets of data.
reasons:
* imbalanced data
* complexible model
* training iteration (epochs)

solutions:
* cross-validation: validation approach, k-folder, LOOCV (data)
* data augmentation(data)
* feature selection(data)
* regularization (learning algorithm)
* ensembling: bagging and boosting

Overfitting can be identified by checking validation metrics as accuracy and loss.The validation metrics usually increase until a point where they stagnate or start declining when the model is affected by overfitting.

#### 3) what is data augmentation? and techniques used to augment data?
Data augmentation are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data.
Basic image process:
* Flip horizontally and vertically
* Rotation
* Scale
* Crop
* Translation
* kernel filters
* Random erasing
* Mixing images
* Color space transfer: constrast and brightness

Deep learning:
* GANS
* Adversarial training
* Neural style transfer

#### 4) what is imbalanced dataset? how to dela with that?
An imbalanced dataset is one that has different proportions of target categories. For example, a dataset with medical images where we have to detect some illness will typically have many more negative samples than positive samples—say, 98% of images are without the illness and 2% of images are with the illness.
There are different options to deal with imbalanced datasets:

* resample the training set:
    * under-sampling: reducing the size of the abundant class,used when quantity of data is sufficient.keeping all samples in the rare class and randomly selecting an equal number of samples in the abundant class, a balanced new dataset can be retrieved for further modelling.
    * Over-sampling: increasing the size of rare samples. Rather than getting rid of abundant samples, new rare samples are generated by using e.g. repetition, bootstrapping or SMOTE (Synthetic Minority Over-Sampling Technique),used when the quantity of data is insufficient.

* Data augmentation. We can add data in the less frequent categories by modifying existing data in a controlled way. In the example dataset, we could flip the images with illnesses, or add noise to copies of the images in such a way that the illness remains visible.

* Using appropriate metrics. In the example dataset, if we had a model that always made negative predictions, it would achieve a precision of 98%. There are other metrics such as precision, recall, and F-score that describe the accuracy of the model better when using an imbalanced dataset.

* clusterthe abundant class: clustering the abundant class in r groups, with r being the number of cases in r. For each group, only the medoid (centre of cluster) is kept. The model is then trained with the rare class and the medoids only

* Design appropriate model that suited for imbalanced data and designing a cost function that is penalizing wrong classification of the rare class more than wrong classifications of the abundant class
* 
#### 5) what is F1 score,recall,precision and AUC?

   ![image](https://user-images.githubusercontent.com/63558665/118019414-f7a75580-b326-11eb-958a-a4b7ccd613bd.png)

* F1 score: The F1 score is the harmonic mean of the precision and recall

   ![image](https://user-images.githubusercontent.com/63558665/118020136-bebbb080-b327-11eb-8eab-e1d752e23de2.png)

* Accuracy:Accuracy is the quintessential classification metric (classification)
   
   ![image](https://user-images.githubusercontent.com/63558665/118019531-16a5e780-b327-11eb-958f-fe0864e73f41.png)
   
* Recall:how often the classifier predict yes when the actual value is yes : true positive/total actual positive

   ![image](https://user-images.githubusercontent.com/63558665/118019959-98961080-b327-11eb-966b-abf409635f1f.png)

* Presicion:when the classifier predict yes, how often is it correct
    
    ![image](https://user-images.githubusercontent.com/63558665/118019901-82885000-b327-11eb-80e1-4bb6237c9be6.png)
 
 * AUC (Area under the ROC curve) is scale-invariant. It measures how well predictions are ranked, rather than their absolute values.
The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives)

   ![image](https://user-images.githubusercontent.com/63558665/118021645-7604f700-b329-11eb-9581-f22317a88357.png)
     
   ![image](https://user-images.githubusercontent.com/63558665/118021703-84531300-b329-11eb-867d-cf3773ec150b.png)
     
   ![image](https://user-images.githubusercontent.com/63558665/118022093-f62b5c80-b329-11eb-8eb3-4ab9945076b9.png)

#### 6) What is the Bias-Variance Trade-off?
Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. 

Variance is the variability of model prediction for a given data point or a value which tells us spread of our data.

Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data.Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before.

The bias-variance tradeoff refers to a decomposition of the prediction error in machine learning as the sum of a bias and a variance term

#### 7) What is cross validation?
Cross Validation is a very useful technique for assessing the effectiveness of your model, particularly in cases where you need to mitigate overfitting. It is also of use in determining the hyper parameters of your model, in the sense that which parameters will result in lowest test error.
* 	Validation set approach: randomly divide the available set of observation into training set and validation set. And fit the model on the training set and compute the validation set error as an estimate of the test error rate.
    * Ad: conceptually simple and easy to implement
    * Disad: The validation estimates the test error rate is highly variable depending on which observation in the training set and which observations in validation set. Only a subset of the observation used to fit the model (less training data). With smaller observation, the validation set error rate may be tended to overestimate the test error rate for the model fit on the entire dataset.
   
* Loocv: leave -one-out-cross validation
Leave out one observation for the validation set and use the remaining n - 1 observations as the training set. Fit the model on training set and make a prediction for the excluded observation and calculate the error rate. Repeat for n time, and each time choose a different observation for the validation set. And the test error rate is the average of these n test error estimation.
   * Ad:
         Less bias than the validation set because we repeatedly fit the model with n-1 observation.
         Not overestimate the test error rate as the validation set approach.
         Performing multiple times will yield the same results since there is no randomly split the training set and validation set
   * Disad:Computation expensive since the model has to fit n times with larger training data.

* K-fold CV:
Randomly divide the set of observations into k groups, or folds, of approximately equal size.
Treat the first fold as validation set and fit the model with remaing k-1 folder, calculate the error rate from the prediction for the validation set. Repeat the produce k time , each time using a different fold as validation set, and the k-fold estimate the test error rate is the average of these k test error rate etimates.
k-fold cross-validation bias can be minimized with k=n (loocv), but this estimate has high variance,k=5/10 will have good bias-variance trade-off
   * ad: less computation than LOOCV and lower variance than LOOCV
   * disad: bias upward

k-fold: high bias with less variance
loocv: less bias with high variance

# Deep learning
#### 1) what is perceptron?
The perceptron is an algorithm for supervised learning of binary classifiers. A binary classifier is a function which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.

  ![image](https://user-images.githubusercontent.com/63558665/118027887-6e951c00-b330-11eb-871c-2deb5ce88451.png)

If the training data is linear separate and small learning rate, we can get the weight from the set of sample pattern and perceptron learning will converge
Training rule:
* Start with a randomly chosen weight vector w0
* While some input vectors remain misclassified update weight based on error
* stop until all samples correctly classified

#### 2) what is backpropagation?
backpropagation learning: Propagating errors at output nodes down to hidden nodes, these computed errors on hidden nodes drives the update of weights

#### 3) what is cnn? why cnn for image? what is pooling?
* CNN is a convolutional neural network (CNN, or ConvNet) is a class of deep neural network, most commonly applied to analyze visual imagery
   
   ![image](https://user-images.githubusercontent.com/63558665/118033750-2d543a80-b337-11eb-89ec-b237e486f48c.png)

* Why cnn for image?
    * Some patterns are much smaller than the whole image
    * The same patterns appear in different regions
    * Subsampling the pixels will not change the object
    * Filters to capture different patterns in the input space 
    * genrating different feature maps with using different kernels
    * each filter detects a small patter
* pooling: is a layer that used to reduce the dimensions of the feature maps. Thus, it reduces the number of parameters to learn and the amount of computation performed in the networ
     * Max pooling:is a pooling operation that selects the maximum element from the region of the feature map covered by the filter
     * Average pooling:computes the average of the elements present in the region of feature map covered by the filter
     * global pooling:reduces each channel in the feature map to a single value
     
#### 4) what is momentum, weight decay,learning rate decay?
#### 5) why 1x1 convolution?
The number of feature maps often increases with the depth of the network, so we can use the 1x1 convolution, which also named channel-wise pooling,to downsample feature maps
* The 1×1 filter can be used to create a linear projection of a stack of feature maps.
* The projection created by a 1×1 can act like channel-wise pooling and be used for dimensionality reduction.
* The projection created by a 1×1 can also be used directly or be used to increase the number of feature maps in a model.

#### 6) What is Batch Normalization and Layer Normalization?
In batch normalization, input values of the same neuron for all the data in the mini-batch are normalized. Whereas in layer normalization, input values for all neurons in the same layer are normalized for each data sample.

Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. The idea is then to normalize the inputs of each layer in such a way that they have a mean output activation of zero and standard deviation of one. This is done for each individual mini-batch at each layer i.e compute the mean and variance of that mini-batch alone, then normalize. This is analogous to how the inputs to networks are standardized. How does this help? We know that normalizing the inputs to a network helps it learn. But a network is just a series of layers, where the output of one layer becomes the input to the next. That means we can think of any layer in a neural network as the first layer of a smaller subsequent network. Thought of as a series of neural networks feeding into each other, we normalize the output of one layer before applying the activation function, and then feed it into the following layer (sub-network).

#### 7) Why would you use many small convolutional kernels such as 3x3 rather than a few large ones? [src]
This is very well explained in the VGGNet paper. There are 2 reasons: First, you can use several smaller kernels rather than few large ones to get the same receptive field and capture more spatial context, but with the smaller kernels you are using less parameters and computations. Secondly, because with smaller kernels you will be using more filters, you'll be able to use more activation functions and thus have a more discriminative mapping function being learned by your CNN.

#### 8) What is Dropout? and why?
Dropout is randomly set some neurons to zero Probability of dropping is a hyperparameter in each forward pass,  0.5 iscommon.
why? --> prevent overfitting
If your deep neural net is significantly overfitting, dropout will usually reduce the number of errors by a lot.

#### 9) Techniques used to initialize the weight?
Weight initialization is a procedure to set the weights of a neural network to small random values that define the starting point for the optimization (learning or training) of the neural network model.

Each time, a neural network is initialized with a different set of weights, resulting in a different starting point for the optimization process, and potentially resulting in a different final set of weights with different performance characteristics.
* Zeros: Initializer that generates tensors initialized to 0.
* Ones: Initializer that generates tensors initialized to 1.
* Constant: Initializer that generates tensors initialized to a constant value.
* RandomNormal: Initializer that generates tensors with a normal distribution.
* RandomUniform: Initializer that generates tensors with a uniform distribution.
* TruncatedNormal: Initializer that generates a truncated normal distribution.
* VarianceScaling: Initializer capable of adapting its scale to the shape of weights.
* Orthogonal: Initializer that generates a random orthogonal matrix.
* Identity: Initializer that generates the identity matrix.
* lecun_uniform: LeCun uniform initializer.
* glorot_normal: Glorot normal initializer, also called Xavier normal initializer.
* glorot_uniform: Glorot uniform initializer, also called Xavier uniform initializer.
* he_normal: He normal initializer.
* lecun_normal: LeCun normal initializer.
* he_uniform: He uniform variance scaling initializer.

* why not zero weight initilization? 
If all the weights are initialized to zeros, the derivatives will remain same for every w in W[l]. As a result, neurons will learn same features in each iterations. This problem is known as network failing to break symmetry. And not only zero, any constant initialization will produce a poor result.

* why randnom weight initialization?
 In this method, the weights are initialized very close to zero, but randomly. This helps in breaking symmetry when backprogragating and every neuron is no longer performing the same computation.
 
 #### 10) optimizer
* Gradient Descent:
   * Batch gradient descent
       * Advantages:
               Easy computation.
               Easy to implement.
               Easy to understand.
       * Disadvantages:
               May trap at local minima.
               Weights are changed after calculating gradient on the whole dataset. So, if the dataset is too large than this may take years to converge to the minima.
               Requires large memory to calculate gradient on the whole dataset.
   * Stochastic gradient descent
       * Advantages:
               Frequent updates of model parameters hence, converges in less time.
               Requires less memory as no need to store values of loss functions.
               May get new minima’s.
       * Disadvantages:
               High variance in model parameters.
               May shoot even after achieving global minima.
               To get the same convergence as gradient descent needs to slowly reduce the value of learning rate.
   * Mini-batch gradient descent
       * Advantages:
               Frequently updates the model parameters and also has less variance.
               Requires medium amount of memory.

All types of Gradient Descent have some challenges:
Choosing an optimum value of the learning rate. If the learning rate is too small than gradient descent may take ages to converge.
Have a constant learning rate for all the parameters. There may be some parameters which we may not want to change at the same rate.
May get trapped at local minima.

* Adaptive:
   * Momentum: Momentum was invented for reducing high variance in SGD and softens the convergence. It accelerates the convergence towards the relevant direction and reduces the fluctuation to the irrelevant direction.
       * Advantages:
               Reduces the oscillations and high variance of the parameters.
               Converges faster than gradient descent.
        * Disadvantages:
               One more hyper-parameter is added which needs to be selected manually and accurately.
   * Adagrad: One of the disadvantages of all the optimizers explained is that the learning rate is constant for all parameters and for each cycle. This optimizer changes the learning rate. It changes the learning rate ‘η’ for each parameter and at every time step ‘t’. learning rate which is modified for given parameter θ(i) at a given time based on previous gradients calculated for given parameter θ(i).
        * Advantages:
               Learning rate changes for each training parameter.
               Don’t need to manually tune the learning rate.
               Able to train on sparse data.
         * Disadvantages:
               Computationally expensive as a need to calculate the second order derivative.
               The learning rate is always decreasing results in slow training.
   * Adadelta:It is an extension of AdaGrad which tends to remove the decaying learning Rate problem of it. Instead of accumulating all previously squared gradients, Adadelta limits the window of accumulated past gradients to some fixed size w. In this exponentially moving average is used rather than the sum of all the gradients.
        * Advantages:
               Now the learning rate does not decay and the training does not stop.
        * Disadvantages:
               Computationally expensive.
   * Adam: Adam (Adaptive Moment Estimation) works with momentums of first and second order. The intuition behind the Adam is that we don’t want to roll so fast just because we can jump over the minimum, we want to decrease the velocity a little bit for a careful search. In addition to storing an exponentially decaying average of past squared gradients like AdaDelta, Adam also keeps an exponentially decaying average of past gradients M(t).
       * Advantages:
               The method is too fast and converges rapidly.
               Rectifies vanishing learning rate, high variance.
        * Disadvantages:
               Computationally costly.
how to choose a optimizer?
Adam is the best optimizers. If one wants to train the neural network in less time and more efficiently than Adam is the optimizer.
For sparse data use the optimizers with dynamic learning rate.
If, want to use gradient descent algorithm than min-batch gradient descent is the best option.

#### 11) tricks for designing,training a model?
Design:
* Reduce filter sizes (except possibly at the lowest layer), factorize filters aggressively
* Use 1x1 convolutions to reduce and expand the number of feature maps judiciously
* Use skip connections and/or create multiple paths through the network

training:
* Training tricks and details: initialization, regularization, normalization
* Training data augmentation
* Averaging classifier outputs over multiple crops/flips
* Ensembles of networks

#### 12) what are RNN, LSTM, GRU? And their pros and cons?
* RNN:
Recurrent Neural Network is a generalization of feedforward neural network that has an internal memory. RNN is recurrent in nature as it performs the same function for every input of data while the output of the current input depends on the past one computation. After producing the output, it is copied and sent back into the recurrent network. For making a decision, it considers the current input and the output that it has learned from the previous input.

   ![image](https://user-images.githubusercontent.com/63558665/118041468-d4899f80-b340-11eb-9d3e-6caa5e07bb28.png)
   
  * Advantages of Recurrent Neural Network
      RNN can model sequence of data so that each sample can be assumed to be dependent on previous ones
      Recurrent neural network are even used with convolutional layers to extend the effective pixel neighbourhood.
  * Disadvantages of Recurrent Neural Network
      Gradient vanishing and exploding problems.
      Training an RNN is a very difficult task.
      It cannot process very long sequences if using tanh or relu as an activation function.
      there is no finer control over which part of the context needs to be carried forward and how much of the past needs to be ‘forgotten’. 
 * LSTM:(sigmoid function)
Long Short-Term Memory (LSTM) networks are a modified version of recurrent neural networks, which makes it easier to remember past data in memory. The vanishing gradient problem of RNN is resolved here. LSTM is well-suited to classify, process and predict time series given time lags of unknown duration. It trains the model by using back-propagation.
    * advantage: solve the vanishing gradient
    * disadvantage:
        it fail to remove vanishing gradient completely
        Memory and input are added
        it requires a lot of resources and time to get trained and become ready for real-world applications
        LSTMs get affected by different random weight initializations and hence behave quite similar to that of a feed-forward neural net
        LSTMs are prone to overfitting and it is difficult to apply the dropout algorithm to curb this issue
* GRU:A very simplified version of the LSTM, which merges forget and input gate into a single ‘update’ gate and merges cell and hidden state
     * advantage:Has 
          fewer parameters than an LSTM
          it has been shown to outperform LSTM on some tasks
          GRU uses less memory and is faster than LSTM
     * disadvantage: 
           slow convergence and low learning efficiency
 #### 13) what is transfer learning and why?
Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.train a model by source data, then fine-tune the model by target data!

Transfer learning is an optimization, a shortcut to saving time or getting better performance.
* Challenge: only limited target data, so be careful about overfitting

   ![image](https://user-images.githubusercontent.com/63558665/118047014-24b83000-b348-11eb-9dea-13313f4580c3.png)
 
 https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
#### 14) What is the significance of Residual Networks? 
The main thing that residual connections did was allow for direct feature access from previous layers. This makes information propagation throughout the network much easier. One very interesting paper about this shows how using local skip connections gives the network a type of ensemble multi-path structure, giving features multiple paths to propagate throughout the network.

#### 15) Why do we have max-pooling in classification CNNs? [src]
for a role in Computer Vision. Max-pooling in a CNN allows you to reduce computation since your feature maps are smaller after the pooling. You don't lose too much semantic information since you're taking the maximum activation. There's also a theory that max-pooling contributes a bit to giving CNNs more translation in-variance. Check out this great video from Andrew Ng on the benefits of max-pooling.

#### 16 ) Why do we use convolutions for images rather than just FC layers? [src]
Firstly, convolutions preserve, encode, and actually use the spatial information from the image. If we used only FC layers we would have no relative spatial information. Secondly, Convolutional Neural Networks (CNNs) have a partially built-in translation in-variance, since each convolution kernel acts as it's own filter/feature detector.

#### 17) Why is ReLU better and more often used than Sigmoid in Neural Networks? [src]
Imagine a network with random initialized weights ( or normalised ) and almost 50% of the network yields 0 activation because of the characteristic of ReLu ( output 0 for negative values of x ). This means a fewer neurons are firing ( sparse activation ) and the network is lighter.

#### 18)  What is "One Hot Encoding"? Why and when do you have to use it?
One Hot encoding converts categorical data into an integer representation.
* Categorical data is defined as variables with a finite set of label values.
* Most machine learning algorithms require numerical input and output variables.
* An integer and one hot encoding is used to convert categorical data to integer data.

#### 19) How Will You Know Which Machine Learning Algorithm to Choose for Your Classification Problem?
While there is no fixed rule to choose an algorithm for a classification problem, you can follow these guidelines:

* If accuracy is a concern, test different algorithms and cross-validate them
* If the training dataset is small, use models that have low variance and high bias
* If the training dataset is large, use models that have high variance and little bias

#### 20) Advantages and disadvantages of Principal Component Analysis in Machine Learning?
Principal Component Analysis (PCA) is a statistical techniques used to reduce the dimensionality of the data (reduce the number of features in the dataset) by selecting the most important features that capture maximum information about the dataset. Advantages of Principal Component Analysis
* Removes Correlated Features.
* Improves Algorithm Performance.
* Reduces Overfitting.
* Improves Visualization.
Disadvantages of Principal Component Analysis
* Independent variables become less interpretable: After implementing PCA on the dataset, your original features will turn into Principal Components. Principal Components are the linear combination of your original features. Principal Components are not as readable and interpretable as original features.
* Data standardization is must before PCA: You must standardize your data before implementing PCA, otherwise PCA will not be able to find the optimal Principal Components. Use StandardScaler from Scikit Learn to standardize the dataset features onto unit scale (mean = 0 and standard deviation = 1) which is a requirement for the optimal performance of many Machine Learning algorithms.
* Information Loss: Although Principal Components try to cover maximum variance among the features in a dataset, if we don't select the number of Principal Components with care, it may miss some information as compared to the original list of features.


#### 21) What are the main differences between K-means and K-nearest neighbours?
K-means is a clustering algorithm that tries to partition a set of points into K sets (clusters) such that the points in each cluster tend to be near each other. It is unsupervised because the points have no external classification.

KNN is a classification (or regression) algorithm that in order to determine the classification of a point, combines the classification of the K nearest points. It is supervised because you are trying to classify a point based on the known classification of other points.

* Question: What is the difference between K-means and KNN?¶
K-NN is a Supervised machine learning while K-means is an unsupervised machine learning.
K-NN is a classification or regression machine learning algorithm while K-means is a clustering machine learning algorithm.
K-NN is a lazy learner while K-Means is an eager learner. An eager learner has a model fitting that means a training step but a lazy learner does not have a training phase.
K-NN performs much better if all of the data have the same scale but this is not true for K-means.
* Question: How to optimize the result of K means?
visualize your data and the clustering results (PCA, so you see the outliers).
Start with a smaller sample. Sample them at random and test multiple starting points.
Use other clustering algorithms than k-means.


#### 22) What is support vector machine (SVM)?
SVM stands for support vector machine, it is a supervised machine learning algorithm which can be used for both Regression and Classification. If you have n features in your training data set, SVM tries to plot it in n-dimensional space with the value of each feature being the value of a particular coordinate. SVM uses hyper planes to separate out different classes based on the provided kernel function. The SVM finds the maximum margin separating hyperplane.

* Question: What is the best separating hyperplane?¶
The one that maximizes the distance to the closest data points from both classes. We say it is the hyperplane with maximum margin.

* Question: What are margin, support vectors in SVM?
Maximum margin: the maximum distance between data points of both classes. Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier.

* Question: What are the different kernels functions in SVM ?
The function of kernel is to take data as input and transform it into the required form. For example linear, nonlinear, polynomial, radial basis function (RBF), and sigmoid. The kernel functions return the inner product between two points in a suitable feature space.

* Question: Hard and soft margin Support Vector Machine (SVM)?
Soft margin is extended version of hard margin SVM.
Hard margin SVM can work only when data is completely linearly separable without any errors (noise or outliers). In case of errors either the margin is smaller or hard margin SVM fails. On the other hand soft margin SVM was proposed by Vapnik to solve this problem by introducing slack variables.
As for as their usage is concerned since Soft margin is extended version of hard margin SVM so we use Soft margin SVM.
* Question: What is the difference between SVM and logistic regression?
In the case of two classes are linearly separable. LR finds any solution that separates the two classes. Hard SVM finds "the" solution among all possible ones that has the maximum margin. In case of soft SVM and the classes not being linearly separable. LR finds a hyperplane that corresponds to the minimization of some error. Soft SVM tries to minimize the error (another error) and at the same time trades off that error with the margin via a regularization parameter. SVM is a hard classifier but LR is a probabilistic one.



# Other
## Segmentation: https://nanonets.com/blog/semantic-image-segmentation-2020/
 ## Reference
 * https://github.com/AllenCX/DS-ML-Interview-Questions
 * https://github.com/andrewekhalel/MLQuestions
 * https://yanjin.space/blog/2020/2020305.html
