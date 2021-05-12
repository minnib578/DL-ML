# Linear Regression and Logistic Regression
## What is linear regression? What is logistic regression? What is the difference?
* Linear regression is used to predict the continuous dependent variable using a given set of independent variables, and it is used for solving Regression problem.In Linear regression, we predict the value of continuous variables. Least square estimation method is used for estimation of accuracy.
example: House pricing

   
   ![image](https://user-images.githubusercontent.com/63558665/117905179-bcfada00-b2a0-11eb-826e-1daf125867b6.png)
   ![image](https://user-images.githubusercontent.com/63558665/117905255-e4ea3d80-b2a0-11eb-8f2b-97c830082280.png)


* Logistic Regression is used to predict the categorical dependent variable using a given set of independent variables, and it is used for solving classification problem.In logistic Regression, we predict the values of categorical variables.Maximum likelihood estimation method is used for estimation of accuracy.The output of Logistic Regression problem can be only between the 0 and 1.In logistic regression, we pass the weighted sum of inputs through an activation function that can map values in between 0 and 1. Such activation function is known as sigmoid function and the curve obtained is called as sigmoid curve.
example:credit card default

## what is gradient descent?
Gradient descent is an optimization algorithm that's used when training a machine learning model. It's based on a convex function and tweaks its parameters iteratively to   minimize a given function to its local minimum.Gradient descent is a convex function.

how does gradient descent work?
The equation below describes what gradient descent does: b is the next position of our climber, while a represents his current position. The minus sign refers to the minimization part of gradient descent. The gamma in the middle is a waiting factor(stepsize) and the gradient term ( Δf(a) ) is simply the direction of the steepest descent.
    ![image](https://user-images.githubusercontent.com/63558665/117905892-e9632600-b2a1-11eb-9a4d-6a7eb210113c.png)

So this formula basically tells us the next position we need to go, which is the direction of the steepest descent
## how to choose the learning rate? and how to choose learning rate
For gradient descent to reach the local minimum we must set the learning rate to an appropriate value, which is neither too low nor too high. This is important because if the steps it takes are too big, it may not reach the local minimum because it bounces back and forth between the convex function of gradient descent (see left image below). If we set the learning rate to a very small value, gradient descent will eventually reach the local minimum but that may take a while(see right image).
   ![image](https://user-images.githubusercontent.com/63558665/117906321-ace3fa00-b2a2-11eb-9767-71d24d1923c2.png)
if the learning rate is up and down interatively, it may be caused by high learning rate.
in order to choose a good learning, plotting the cost function as the optimization runs also is a method
chosing a learning rate:
* Try multiple choise,crude rule of thumb: update changes weight about 01-1%
* Linesearch: keep walking in the same direction as long as 𝑓 is still decreasing

## what is batch gradient descent, stachastic gradient descent, mini-batch gradient descent?
* batch gradient descent calculates the error for each example within the training dataset, but only after all training examples have been evaluated does the model get updated
     
     ![image](https://user-images.githubusercontent.com/63558665/117908665-d2730280-b2a6-11eb-8265-935f26e9e2d8.png)
     * advantage:computational efficient,it produces a stable error gradient and a stable convergence,Less oscillations and noisy steps taken towards the global minima of the loss function due to updating the parameters by computing the average of all the training samples rather than the value of a single sample
     * disadvantage:stable error gradient can lead to a local minimum.The entire training set can be too large to process in the memory due to which additional memory might be needed. Depending on computer resources it can take too long for processing all the training samples as a batch
* stochastic gradient descent: one training sample (example) is passed through the neural network at a time and the parameters (weights) of each layer are updated with the computed gradient

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
 * minni-batch gradient descent: Mini-batch gradient descent is the go-to method since it’s a combination of the concepts of SGD and batch gradient descent. It simply splits the training dataset into small batches and performs an update for each of those batches
 
    ![image](https://user-images.githubusercontent.com/63558665/117910758-80cc7700-b2aa-11eb-85e9-a93a486a046f.png)
    
    * advantage:
    Easily fits in the memory
    It is computationally efficient
    Benefit from vectorization
    If stuck in local minimums, some noisy steps can lead the way out of them
    Average of the training samples produces stable error gradients and convergence

 
 
 

