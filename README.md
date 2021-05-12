# Linear Regression and Logistic Regression

#### 1)  What is linear regression? What is logistic regression? What is the difference?
* Linear regression is used to predict the continuous dependent variable using a given set of independent variables, and it is used for solving Regression problem.In Linear regression, we predict the value of continuous variables. Least square estimation method is used for estimation of accuracy.
example: House pricing

   
   ![image](https://user-images.githubusercontent.com/63558665/117905179-bcfada00-b2a0-11eb-826e-1daf125867b6.png)
   
   ![image](https://user-images.githubusercontent.com/63558665/117905255-e4ea3d80-b2a0-11eb-8f2b-97c830082280.png)

* Logistic Regression is used to predict the categorical dependent variable using a given set of independent variables, and it is used for solving classification problem.In logistic Regression, we predict the values of categorical variables.Maximum likelihood estimation method is used for estimation of accuracy.The output of Logistic Regression problem can be only between the 0 and 1.In logistic regression, we pass the weighted sum of inputs through an activation function that can map values in between 0 and 1. Such activation function is known as sigmoid function and the curve obtained is called as sigmoid curve.
example:credit card default

#### 2)  what is gradient descent?
Gradient descent is an optimization algorithm that's used when training a machine learning model. It's based on a convex function and tweaks its parameters iteratively to   minimize a given function to its local minimum.Gradient descent is a convex function.

how does gradient descent work?
The equation below describes what gradient descent does: b is the next position of our climber, while a represents his current position. The minus sign refers to the minimization part of gradient descent. The gamma in the middle is a waiting factor(stepsize) and the gradient term ( Δf(a) ) is simply the direction of the steepest descent.

    ![image](https://user-images.githubusercontent.com/63558665/117905892-e9632600-b2a1-11eb-9a4d-6a7eb210113c.png)

So this formula basically tells us the next position we need to go, which is the direction of the steepest descent


#### 3) how to choose the learning rate? and how to choose learning rate
For gradient descent to reach the local minimum we must set the learning rate to an appropriate value, which is neither too low nor too high. This is important because if the steps it takes are too big, it may not reach the local minimum because it bounces back and forth between the convex function of gradient descent (see left image below). If we set the learning rate to a very small value, gradient descent will eventually reach the local minimum but that may take a while(see right image).

   ![image](https://user-images.githubusercontent.com/63558665/117906321-ace3fa00-b2a2-11eb-9767-71d24d1923c2.png)
   
if the learning rate is up and down interatively, it may be caused by high learning rate.
in order to choose a good learning, plotting the cost function as the optimization runs also is a method
chosing a learning rate:
* Try multiple choise,crude rule of thumb: update changes weight about 01-1%
* Linesearch: keep walking in the same direction as long as 𝑓 is still decreasing


#### 4)what is batch gradient descent, stachastic gradient descent, mini-batch gradient descent?
1. batch gradient descent calculates the error for each example within the training dataset, but only after all training examples have been evaluated does the model get updated
     
     ![image](https://user-images.githubusercontent.com/63558665/117908665-d2730280-b2a6-11eb-8265-935f26e9e2d8.png)
     
     * advantage:computational efficient,it produces a stable error gradient and a stable convergence,Less oscillations and noisy steps taken towards the global minima of the loss function due to updating the parameters by computing the average of all the training samples rather than the value of a single sample
     * disadvantage:stable error gradient can lead to a local minimum.The entire training set can be too large to process in the memory due to which additional memory might be needed. Depending on computer resources it can take too long for processing all the training samples as a batch
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
 
 
 #### 5) what is L1 and L2 regularization? what is the difference between L1 ,L2 and Linear regression?
 * Linear regression finds the parameters to minimize the mean squared error or residuals between the predictions and the targets.Overfitting occurs when the model makes much better predictions on known data than on unknown data. The model begins to memorize the training data and is unable to generalize to unseen test data. and then we need to simplefy the model by introducing the regularization L1 and L2
 * L1 regularization (Lasso regression) to force some coefficients to be exactly zero. This means some features are completely ignored by the model.If lambda is zero then we will get back OLS whereas very large value will make coefficients zero hence it will under-fit
 * L2 regularization (Ridge regression) adds “squared magnitude” of coefficient as penalty term to the loss function.if lambda is zero then you can imagine we get back OLS. However, If lambda is very large then it will add too much weight and it will lead to under-fitting.
    
    
    ![image](https://user-images.githubusercontent.com/63558665/117913152-d2770080-b2ae-11eb-9f7a-d73ad0156754.png)


    ![image](https://user-images.githubusercontent.com/63558665/117913034-9e9bdb00-b2ae-11eb-81f1-97763aa9163f.png)
    
The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.

#### 6)what is mean sqaured error, cross-entropy ? and what is the difference ?
* MSE measures the average of the squares of the errors
    
    ![image](https://user-images.githubusercontent.com/63558665/117914164-c3914d80-b2b0-11eb-936c-90862ce28ab3.png)
* Cross-entropy is a measure of the difference between two probability distributions for a given random variable or set of events.
    
    ![image](https://user-images.githubusercontent.com/63558665/117914617-b45ecf80-b2b1-11eb-8248-343d3f13f2d2.png)
    
    MSE is used for linear regression and cross-entropy is used for classification problems
    
#### 7) what is activation function? and the classification of activation function.
An activation function in a neural network defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network
* sigmoid used in logistic regression classification.The function takes any real value as input and outputs values in the range 0 to 1. The larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to 0.0. sigmoid cause vanishing gradient, and the gradient will be zero when x is inf and -inf.

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
     
     
#### 8) Does global optimal can be reached by SGD, why？
when the learning rates decrease with an appropriate rate, and subject to relatively mild assumptions, stochastic gradient descent converges almost surely to a global minimum when the objective function is convex or pseudoconvex, and otherwise converges almost surely to a local minimum

### 9) what is convex optimization and non-convex optimization?
* Non-convex: local minima and sensitive to initilization
* convex: global minima


#### 10) what vanishing/exploding gradient?

If each term is (much) greater than 1 -->explosion of gradients
If each term is (much) less than 1 -->vanishing gradients
solution for gradient vanishing:
* carefully initilization--want variance to remian approximate constant since variance increase in backward pass causes exploding gradient and variance decrease in backward causes vanishing gradient
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
            Reduces the strong dependence  on initialization
            Acts as a form of regularization  in a funny way, and slightly  reduces the need for dropout
      
      ![image](https://user-images.githubusercontent.com/63558665/117922474-86cd5280-b2c0-11eb-8c58-5cbe6d707002.png)

 
* Residual connections:allow gradients to flow unimpeded, instead of single layers, have residual connections over block
      * doubling of feature channel
      * increase channel by 1x1 convolution
      * decrease spatial resoution by subsampling
      
      ![image](https://user-images.githubusercontent.com/63558665/117920511-f80b0680-b2bc-11eb-9ea0-bd53a72c971c.png)
* bottleneck blocks:use 1x1 to project to lower dimensionality do convolution, then come back-->avoid that when channels increases, 3x3 convolutions introduce many parameters.
      
      ![image](https://user-images.githubusercontent.com/63558665/117921786-56d17f80-b2bf-11eb-8c9b-930a6261fe5a.png)

#### 11) why 1x1 convolution?
* The 1×1 filter can be used to create a linear projection of a stack of feature maps.
* The projection created by a 1×1 can act like channel-wise pooling and be used for dimensionality reduction.
* The projection created by a 1×1 can also be used directly or be used to increase the number of feature maps in a model.

# Regularization, Overfitting and Model/Feature Selection/Evaluation.

#### 1)  What is regularization, why do we use it, and give some examples of common methods? 
     
