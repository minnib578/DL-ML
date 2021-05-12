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
 
#### 5)what is mean sqaured error, cross-entropy ? and what is the difference ?
* MSE measures the average of the squares of the errors
    
    ![image](https://user-images.githubusercontent.com/63558665/117914164-c3914d80-b2b0-11eb-936c-90862ce28ab3.png)
* Cross-entropy is a measure of the difference between two probability distributions for a given random variable or set of events.
    
    ![image](https://user-images.githubusercontent.com/63558665/117914617-b45ecf80-b2b1-11eb-8248-343d3f13f2d2.png)
    
    MSE is used for linear regression and cross-entropy is used for classification problems
    
#### 6) what is activation function? and the classification of activation function.
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
     
     
#### 7) Does global optimal can be reached by SGD, why？
when the learning rates decrease with an appropriate rate, and subject to relatively mild assumptions, stochastic gradient descent converges almost surely to a global minimum when the objective function is convex or pseudoconvex, and otherwise converges almost surely to a local minimum

#### 8) what is convex optimization and non-convex optimization?
* Non-convex: local minima and sensitive to initilization
* convex: global minima


#### 9) what vanishing/exploding gradient?

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

#### 10) why 1x1 convolution?
* The 1×1 filter can be used to create a linear projection of a stack of feature maps.
* The projection created by a 1×1 can act like channel-wise pooling and be used for dimensionality reduction.
* The projection created by a 1×1 can also be used directly or be used to increase the number of feature maps in a model.

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

