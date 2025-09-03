
# Machine Learning Interview Questions
A collection of technical interview questions for machine learning and computer vision engineering positions.

#### 1) What's the trade-off between bias and variance? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data. [[src]](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)

#### 2) What is gradient descent? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
[[Answer]](https://towardsdatascience.com/gradient-descent-in-a-nutshell-eaf8c18212f0)

#### 3) Explain over- and under-fitting and how to combat them? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
[[Answer]](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)

#### 4) How do you combat the curse of dimensionality? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

 - Manual Feature Selection
 - Principal Component Analysis (PCA)
 - Multidimensional Scaling
 - Locally linear embedding  
[[src]](https://towardsdatascience.com/why-and-how-to-get-rid-of-the-curse-of-dimensionality-right-with-breast-cancer-dataset-7d528fb5f6c0)

#### 5) What is regularization, why do we use it, and give some examples of common methods? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
A technique that discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. 
Examples
 - Ridge (L2 norm)
 - Lasso (L1 norm)  
The obvious *disadvantage* of **ridge** regression, is model interpretability. It will shrink the coefficients for least important predictors, very close to zero. But it will never make them exactly zero. In other words, the *final model will include all predictors*. However, in the case of the **lasso**, the L1 penalty has the effect of forcing some of the coefficient estimates to be *exactly equal* to zero when the tuning parameter λ is sufficiently large. Therefore, the lasso method also performs variable selection and is said to yield sparse models.
[[src]](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

#### 6) Explain Principal Component Analysis (PCA)? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
[[Answer]](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)

#### 7) Why is ReLU better and more often used than Sigmoid in Neural Networks? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
Imagine a network with random initialized weights ( or normalised ) and almost 50% of the network yields 0 activation because of the characteristic of ReLu ( output 0 for negative values of x ). This means a fewer neurons are firing ( sparse activation ) and the network is lighter. [[src]](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)


#### 8) Given stride S and kernel sizes  for each layer of a (1-dimensional) CNN, create a function to compute the [receptive field](https://www.quora.com/What-is-a-receptive-field-in-a-convolutional-neural-network) of a particular node in the network. This is just finding how many input nodes actually connect through to a neuron in a CNN. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 9) Implement [connected components](http://aishack.in/tutorials/labelling-connected-components-example/) on an image/matrix. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]


#### 10) Implement a sparse matrix class in C++. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]


#### 11) Create a function to compute an [integral image](https://en.wikipedia.org/wiki/Summed-area_table), and create another function to get area sums from the integral image.[[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]


#### 12) How would you [remove outliers](https://en.wikipedia.org/wiki/Random_sample_consensus) when trying to estimate a flat plane from noisy samples? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 13) How does [CBIR](https://www.robots.ox.ac.uk/~vgg/publications/2013/arandjelovic13/arandjelovic13.pdf) work? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 14) How does image registration work? Sparse vs. dense [optical flow](http://www.ncorr.com/download/publications/bakerunify.pdf) and so on. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 15) Describe how convolution works. What about if your inputs are grayscale vs RGB imagery? What determines the shape of the next layer? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 16) Talk me through how you would create a 3D model of an object from imagery and depth sensor measurements taken at all angles around the object. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 17) Implement SQRT(const double & x) without using any special functions, just fundamental arithmetic. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 18) Reverse a bitstring. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 19) Implement non maximal suppression as efficiently as you can. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 20) Reverse a linked list in place. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 21) What is data normalization and why do we need it? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
Data normalization is very important preprocessing step, used to rescale values to fit in a specific range to assure better convergence during backpropagation. In general, it boils down to subtracting the mean of each data point and dividing by its standard deviation. If we don't do this then some of the features (those with high magnitude) will be weighted more in the cost function (if a higher-magnitude feature changes by 1%, then that change is pretty big, but for smaller features it's quite insignificant). The data normalization makes all features weighted equally.

#### 22) Why do we use convolutions for images rather than just FC layers? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
Firstly, convolutions preserve, encode, and actually use the spatial information from the image. If we used only FC layers we would have no relative spatial information. Secondly, Convolutional Neural Networks (CNNs) have a partially built-in translation in-variance, since each convolution kernel acts as it's own filter/feature detector.

#### 23) What makes CNNs translation invariant? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
As explained above, each convolution kernel acts as it's own filter/feature detector. So let's say you're doing object detection, it doesn't matter where in the image the object is since we're going to apply the convolution in a sliding window fashion across the entire image anyways.

#### 24) Why do we have max-pooling in classification CNNs? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
for a role in Computer Vision. Max-pooling in a CNN allows you to reduce computation since your feature maps are smaller after the pooling. You don't lose too much semantic information since you're taking the maximum activation. There's also a theory that max-pooling contributes a bit to giving CNNs more translation in-variance. Check out this great video from Andrew Ng on the [benefits of max-pooling](https://www.coursera.org/learn/convolutional-neural-networks/lecture/hELHk/pooling-layers).

#### 25) Why do segmentation CNNs typically have an encoder-decoder style / structure? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
The encoder CNN can basically be thought of as a feature extraction network, while the decoder uses that information to predict the image segments by "decoding" the features and upscaling to the original image size.

#### 26) What is the significance of Residual Networks? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
The main thing that residual connections did was allow for direct feature access from previous layers. This makes information propagation throughout the network much easier. One very interesting paper about this shows how using local skip connections gives the network a type of ensemble multi-path structure, giving features multiple paths to propagate throughout the network.

#### 27) What is batch normalization and why does it work? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. The idea is then to normalize the inputs of each layer in such a way that they have a mean output activation of zero and standard deviation of one. This is done for each individual mini-batch at each layer i.e compute the mean and variance of that mini-batch alone, then normalize. This is analogous to how the inputs to networks are standardized. How does this help? We know that normalizing the inputs to a network helps it learn. But a network is just a series of layers, where the output of one layer becomes the input to the next. That means we can think of any layer in a neural network as the first layer of a smaller subsequent network. Thought of as a series of neural networks feeding into each other, we normalize the output of one layer before applying the activation function, and then feed it into the following layer (sub-network).

#### 28) Why would you use many small convolutional kernels such as 3x3 rather than a few large ones? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
This is very well explained in the [VGGNet paper](https://arxiv.org/pdf/1409.1556.pdf). There are 2 reasons: First, you can use several smaller kernels rather than few large ones to get the same receptive field and capture more spatial context, but with the smaller kernels you are using less parameters and computations. Secondly, because with smaller kernels you will be using more filters, you'll be able to use more activation functions and thus have a more discriminative mapping function being learned by your CNN.

#### 29) Why do we need a validation set and test set? What is the difference between them? [[src](https://www.toptal.com/machine-learning/interview-questions)]
When training a model, we divide the available data into three separate sets:

 - The training dataset is used for fitting the model’s parameters. However, the accuracy that we achieve on the training set is not reliable for predicting if the model will be accurate on new samples.
 - The validation dataset is used to measure how well the model does on examples that weren’t part of the training dataset. The metrics computed on the validation data can be used to tune the hyperparameters of the model. However, every time we evaluate the validation data and we make decisions based on those scores, we are leaking information from the validation data into our model. The more evaluations, the more information is leaked. So we can end up overfitting to the validation data, and once again the validation score won’t be reliable for predicting the behaviour of the model in the real world.
 - The test dataset is used to measure how well the model does on previously unseen examples. It should only be used once we have tuned the parameters using the validation set.

So if we omit the test set and only use a validation set, the validation score won’t be a good estimate of the generalization of the model.

#### 30) What is stratified cross-validation and when should we use it? [[src](https://www.toptal.com/machine-learning/interview-questions)]
Cross-validation is a technique for dividing data between training and validation sets. On typical cross-validation this split is done randomly. But in stratified cross-validation, the split preserves the ratio of the categories on both the training and validation datasets.

For example, if we have a dataset with 10% of category A and 90% of category B, and we use stratified cross-validation, we will have the same proportions in training and validation. In contrast, if we use simple cross-validation, in the worst case we may find that there are no samples of category A in the validation set.

Stratified cross-validation may be applied in the following scenarios:

 - On a dataset with multiple categories. The smaller the dataset and the more imbalanced the categories, the more important it will be to use stratified cross-validation.
 - On a dataset with data of different distributions. For example, in a dataset for autonomous driving, we may have images taken during the day and at night. If we do not ensure that both types are present in training and validation, we will have generalization problems.

#### 31) Why do ensembles typically have higher scores than individual models? [[src](https://www.toptal.com/machine-learning/interview-questions)]
An ensemble is the combination of multiple models to create a single prediction. The key idea for making better predictions is that the models should make different errors. That way the errors of one model will be compensated by the right guesses of the other models and thus the score of the ensemble will be higher.

We need diverse models for creating an ensemble. Diversity can be achieved by:
 - Using different ML algorithms. For example, you can combine logistic regression, k-nearest neighbors, and decision trees.
 - Using different subsets of the data for training. This is called bagging.
 - Giving a different weight to each of the samples of the training set. If this is done iteratively, weighting the samples according to the errors of the ensemble, it’s called boosting.
Many winning solutions to data science competitions are ensembles. However, in real-life machine learning projects, engineers need to find a balance between execution time and accuracy.

#### 32) What is an imbalanced dataset? Can you list some ways to deal with it? [[src](https://www.toptal.com/machine-learning/interview-questions)]
An imbalanced dataset is one that has different proportions of target categories. For example, a dataset with medical images where we have to detect some illness will typically have many more negative samples than positive samples—say, 98% of images are without the illness and 2% of images are with the illness.

There are different options to deal with imbalanced datasets:
 - Oversampling or undersampling. Instead of sampling with a uniform distribution from the training dataset, we can use other distributions so the model sees a more balanced dataset.
 - Data augmentation. We can add data in the less frequent categories by modifying existing data in a controlled way. In the example dataset, we could flip the images with illnesses, or add noise to copies of the images in such a way that the illness remains visible.
 - Using appropriate metrics. In the example dataset, if we had a model that always made negative predictions, it would achieve a precision of 98%. There are other metrics such as precision, recall, and F-score that describe the accuracy of the model better when using an imbalanced dataset.

#### 33) Can you explain the differences between supervised, unsupervised, and reinforcement learning? [[src](https://www.toptal.com/machine-learning/interview-questions)]
In supervised learning, we train a model to learn the relationship between input data and output data. We need to have labeled data to be able to do supervised learning.

With unsupervised learning, we only have unlabeled data. The model learns a representation of the data. Unsupervised learning is frequently used to initialize the parameters of the model when we have a lot of unlabeled data and a small fraction of labeled data. We first train an unsupervised model and, after that, we use the weights of the model to train a supervised model.

In reinforcement learning, the model has some input data and a reward depending on the output of the model. The model learns a policy that maximizes the reward. Reinforcement learning has been applied successfully to strategic games such as Go and even classic Atari video games.

#### 34) What is data augmentation? Can you give some examples? [[src](https://www.toptal.com/machine-learning/interview-questions)]
Data augmentation is a technique for synthesizing new data by modifying existing data in such a way that the target is not changed, or it is changed in a known way.

Computer vision is one of fields where data augmentation is very useful. There are many modifications that we can do to images:
 - Resize
 - Horizontal or vertical flip
 - Rotate
 - Add noise
 - Deform
 - Modify colors
Each problem needs a customized data augmentation pipeline. For example, on OCR, doing flips will change the text and won’t be beneficial; however, resizes and small rotations may help.

#### 35) What is Turing test? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
The Turing test is a method to test the machine’s ability to match the human level intelligence. A machine is used to challenge the human intelligence that when it passes the test, it is considered as intelligent. Yet a machine could be viewed as intelligent without sufficiently knowing about people to mimic a human.

#### 36) What is Precision?  
Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances  
Precision = true positive / (true positive + false positive)  
[[src]](https://en.wikipedia.org/wiki/Precision_and_recall)

#### 37) What is Recall?  
Recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.
Recall = true positive / (true positive + false negative)  
[[src]](https://en.wikipedia.org/wiki/Precision_and_recall)

#### 38) Define F1-score. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
It is the weighted average of precision and recall. It considers both false positive and false negative into account. It is used to measure the model’s performance.  
F1-Score = 2 * (precision * recall) / (precision + recall)

#### 39) What is cost function? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
Cost function is a scalar functions which Quantifies the error factor of the Neural Network. Lower the cost function better the Neural network. Eg: MNIST Data set to classify the image, input image is digit 2 and the Neural network wrongly predicts it to be 3

#### 40) List different activation neurons or functions. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
 - Linear Neuron
 - Binary Threshold Neuron
 - Stochastic Binary Neuron
 - Sigmoid Neuron
 - Tanh function
 - Rectified Linear Unit (ReLU)

#### 41) Define Learning rate.
Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient. [[src](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)]

#### 42) What is Momentum (w.r.t NN optimization)?
Momentum lets the optimization algorithm remembers its last step, and adds some proportion of it to the current step. This way, even if the algorithm is stuck in a flat region, or a small local minimum, it can get out and continue towards the true minimum. [[src]](https://www.quora.com/What-is-the-difference-between-momentum-and-learning-rate)

#### 43) What is the difference between Batch Gradient Descent and Stochastic Gradient Descent?
Batch gradient descent computes the gradient using the whole dataset. This is great for convex, or relatively smooth error manifolds. In this case, we move somewhat directly towards an optimum solution, either local or global. Additionally, batch gradient descent, given an annealed learning rate, will eventually find the minimum located in it's basin of attraction.

Stochastic gradient descent (SGD) computes the gradient using a single sample. SGD works well (Not well, I suppose, but better than batch gradient descent) for error manifolds that have lots of local maxima/minima. In this case, the somewhat noisier gradient calculated using the reduced number of samples tends to jerk the model out of local minima into a region that hopefully is more optimal. [[src]](https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent)

#### 44) Epoch vs Batch vs Iteration.
Epoch: one forward pass and one backward pass of **all** the training examples  
Batch: examples processed together in one pass (forward and backward)  
Iteration: number of training examples / Batch size  

#### 45) What is vanishing gradient? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
As we add more and more hidden layers, back propagation becomes less and less useful in passing information to the lower layers. In effect, as information is passed back, the gradients begin to vanish and become small relative to the weights of the networks.

#### 46) What are dropouts? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
Dropout is a simple way to prevent a neural network from overfitting. It is the dropping out of some of the units in a neural network. It is similar to the natural reproduction process, where the nature produces offsprings by combining distinct genes (dropping out others) rather than strengthening the co-adapting of them.

#### 47) Define LSTM. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
Long Short Term Memory – are explicitly designed to address the long term dependency problem, by maintaining a state what to remember and what to forget.

#### 48) List the key components of LSTM. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
 - Gates (forget, Memory, update & Read)
 - tanh(x) (values between -1 to 1)
 - Sigmoid(x) (values between 0 to 1)

#### 49) List the variants of RNN. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
 - LSTM: Long Short Term Memory
 - GRU: Gated Recurrent Unit
 - End to End Network
 - Memory Network

#### 50) What is Autoencoder, name few applications. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
Auto encoder is basically used to learn a compressed form of given data. Few applications include
 - Data denoising
 - Dimensionality reduction
 - Image reconstruction
 - Image colorization

#### 51) What are the components of GAN? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
 - Generator
 - Discriminator

#### 52) What's the difference between boosting and bagging?
Boosting and bagging are similar, in that they are both ensembling techniques, where a number of weak learners (classifiers/regressors that are barely better than guessing) combine (through averaging or max vote) to create a strong learner that can make accurate predictions. Bagging means that you take bootstrap samples (with replacement) of your data set and each sample trains a (potentially) weak learner. Boosting, on the other hand, uses all data to train each learner, but instances that were misclassified by the previous learners are given more weight so that subsequent learners give more focus to them during training. [[src]](https://www.quora.com/Whats-the-difference-between-boosting-and-bagging)

#### 53) Explain how a ROC curve works. [[src]](https://www.springboard.com/blog/machine-learning-interview-questions/)
The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).

#### 54) What’s the difference between Type I and Type II error? [[src]](https://www.springboard.com/blog/machine-learning-interview-questions/)
Type I error is a false positive, while Type II error is a false negative. Briefly stated, Type I error means claiming something has happened when it hasn’t, while Type II error means that you claim nothing is happening when in fact something is.
A clever way to think about this is to think of Type I error as telling a man he is pregnant, while Type II error means you tell a pregnant woman she isn’t carrying a baby.

#### 55) What’s the difference between a generative and discriminative model? [[src]](https://www.springboard.com/blog/machine-learning-interview-questions/)
A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. Discriminative models will generally outperform generative models on classification tasks.




# 100-Days-Of-ML-Code

100 Days of Machine Learning Coding as proposed by [Siraj Raval](https://github.com/llSourcell)

Get the datasets from [here](https://github.com/Avik-Jain/100-Days-Of-ML-Code/tree/master/datasets)

## Data PreProcessing | Day 1
Check out the code from [here](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%201_Data%20PreProcessing.md).

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%201.jpg">
</p>

## Simple Linear Regression | Day 2
Check out the code from [here](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day2_Simple_Linear_Regression.md).

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%202.jpg">
</p>

## Multiple Linear Regression | Day 3
Check out the code from [here](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day3_Multiple_Linear_Regression.md).

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%203.jpg">
</p>

## Logistic Regression | Day 4

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%204.jpg">
</p>

## Logistic Regression | Day 5
Moving forward into #100DaysOfMLCode today I dived into the deeper depth of what Logistic Regression actually is and what is the math involved behind it. Learned how cost function is calculated and then how to apply gradient descent algorithm to cost function to minimize the error in prediction.  
Due to less time I will now be posting an infographic on alternate days.
Also if someone wants to help me out in documentaion of code and already has some experince in the field and knows Markdown for github please contact me on LinkedIn :) .

## Implementing Logistic Regression | Day 6
Check out the Code [here](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%206%20Logistic%20Regression.md)

## K Nearest Neighbours | Day 7
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%207.jpg">
</p>

## Math Behind Logistic Regression | Day 8 

#100DaysOfMLCode To clear my insights on logistic regression I was searching on the internet for some resource or article and I came across this article (https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc) by Saishruthi Swaminathan. 

It gives a detailed description of Logistic Regression. Do check it out.

## Support Vector Machines | Day 9
Got an intution on what SVM is and how it is used to solve Classification problem.

## SVM and KNN | Day 10
Learned more about how SVM works and implementing the K-NN algorithm.

## Implementation of K-NN | Day 11  

Implemented the K-NN algorithm for classification. #100DaysOfMLCode 
Support Vector Machine Infographic is halfway complete. Will update it tomorrow.

## Support Vector Machines | Day 12
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%2012.jpg">
</p>

## Naive Bayes Classifier | Day 13

Continuing with #100DaysOfMLCode today I went through the Naive Bayes classifier.
I am also implementing the SVM in python using scikit-learn. Will update the code soon.

## Implementation of SVM | Day 14
Today I implemented SVM on linearly related data. Used Scikit-Learn library. In Scikit-Learn we have SVC classifier which we use to achieve this task. Will be using kernel-trick on next implementation.
Check the code [here](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%2013%20SVM.md).

## Naive Bayes Classifier and Black Box Machine Learning | Day 15
Learned about different types of naive bayes classifiers. Also started the lectures by [Bloomberg](https://bloomberg.github.io/foml/#home). First one in the playlist was Black Box Machine Learning. It gives the whole overview about prediction functions, feature extraction, learning algorithms, performance evaluation, cross-validation, sample bias, nonstationarity, overfitting, and hyperparameter tuning.

## Implemented SVM using Kernel Trick | Day 16
Using Scikit-Learn library implemented SVM algorithm along with kernel function which maps our data points into higher dimension to find optimal hyperplane. 

## Started Deep learning Specialization on Coursera | Day 17
Completed the whole Week 1 and Week 2 on a single day. Learned Logistic regression as Neural Network. 

## Deep learning Specialization on Coursera | Day 18
Completed the Course 1 of the deep learning specialization. Implemented a neural net in python.

## The Learning Problem , Professor Yaser Abu-Mostafa | Day 19
Started Lecture 1 of 18 of Caltech's Machine Learning Course - CS 156 by Professor Yaser Abu-Mostafa. It was basically an introduction to the upcoming lectures. He also explained Perceptron Algorithm.

## Started Deep learning Specialization Course 2 | Day 20
Completed the Week 1 of Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization.

## Web Scraping | Day 21
Watched some tutorials on how to do web scraping using Beautiful Soup in order to collect data for building a model.

## Is Learning Feasible? | Day 22
Lecture 2 of 18 of Caltech's Machine Learning Course - CS 156 by Professor Yaser Abu-Mostafa. Learned about Hoeffding Inequality.

## Decision Trees | Day 23
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%2023.jpg">
</p>

## Introduction To Statistical Learning Theory | Day 24
Lec 3 of Bloomberg ML course introduced some of the core concepts like input space, action space, outcome space, prediction functions, loss functions, and hypothesis spaces.

## Implementing Decision Trees | Day 25
Check the code [here.](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%2025%20Decision%20Tree.md)

## Jumped To Brush up Linear Algebra | Day 26
Found an amazing [channel](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) on youtube 3Blue1Brown. It has a playlist called Essence of Linear Algebra. Started off by completing 4 videos which gave a complete overview of Vectors, Linear Combinations, Spans, Basis Vectors, Linear Transformations and Matrix Multiplication. 

Link to the playlist [here.](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

## Jumped To Brush up Linear Algebra | Day 27
Continuing with the playlist completed next 4 videos discussing topics 3D Transformations, Determinants, Inverse Matrix, Column Space, Null Space and Non-Square Matrices.

Link to the playlist [here.](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

## Jumped To Brush up Linear Algebra | Day 28
In the playlist of 3Blue1Brown completed another 3 videos from the essence of linear algebra. 
Topics covered were Dot Product and Cross Product.

Link to the playlist [here.](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)


## Jumped To Brush up Linear Algebra | Day 29
Completed the whole playlist today, videos 12-14. Really an amazing playlist to refresh the concepts of Linear Algebra.
Topics covered were the change of basis, Eigenvectors and Eigenvalues, and Abstract Vector Spaces.

Link to the playlist [here.](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

## Essence of calculus | Day 30
Completing the playlist - Essence of Linear Algebra by 3blue1brown a suggestion popped up by youtube regarding a series of videos again by the same channel 3Blue1Brown. Being already impressed by the previous series on Linear algebra I dived straight into it.
Completed about 5 videos on topics such as Derivatives, Chain Rule, Product Rule, and derivative of exponential.

Link to the playlist [here.](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

## Essence of calculus | Day 31
Watched 2 Videos on topic Implicit Diffrentiation and Limits from the playlist Essence of Calculus.

Link to the playlist [here.](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

## Essence of calculus | Day 32
Watched the remaining 4 videos covering topics Like Integration and Higher order derivatives.

Link to the playlist [here.](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

## Random Forests | Day 33
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%2033.jpg">
</p>

## Implementing Random Forests | Day 34
Check the code [here.](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%2034%20Random_Forest.md)

## But what *is* a Neural Network? | Deep learning, chapter 1  | Day 35
An Amazing Video on neural networks by 3Blue1Brown youtube channel. This video gives a good understanding of Neural Networks and uses Handwritten digit dataset to explain the concept. 
Link To the [video.](https://www.youtube.com/watch?v=aircAruvnKk&t=7s)

## Gradient descent, how neural networks learn | Deep learning, chapter 2 | Day 36
Part two of neural networks by 3Blue1Brown youtube channel. This video explains the concepts of Gradient Descent in an interesting way. 169 must watch and highly recommended.
Link To the [video.](https://www.youtube.com/watch?v=IHZwWFHWa-w)

## What is backpropagation really doing? | Deep learning, chapter 3 | Day 37
Part three of neural networks by 3Blue1Brown youtube channel. This video mostly discusses the partial derivatives and backpropagation.
Link To the [video.](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

## Backpropagation calculus | Deep learning, chapter 4 | Day 38
Part four of neural networks by 3Blue1Brown youtube channel. The goal here is to represent, in somewhat more formal terms, the intuition for how backpropagation works and the video moslty discusses the partial derivatives and backpropagation.
Link To the [video.](https://www.youtube.com/watch?v=tIeHLnjs5U8)

## Deep Learning with Python, TensorFlow, and Keras tutorial | Day 39
Link To the [video.](https://www.youtube.com/watch?v=wQ8BIBpya2k&t=19s&index=2&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN)

## Loading in your own data - Deep Learning basics with Python, TensorFlow and Keras p.2 | Day 40
Link To the [video.](https://www.youtube.com/watch?v=j-3vuBynnOE&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=2)

## Convolutional Neural Networks - Deep Learning basics with Python, TensorFlow and Keras p.3 | Day 41
Link To the [video.](https://www.youtube.com/watch?v=WvoLTXIjBYU&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=3)

## Analyzing Models with TensorBoard - Deep Learning with Python, TensorFlow and Keras p.4 | Day 42
Link To the [video.](https://www.youtube.com/watch?v=BqgTU7_cBnk&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=4)

## K Means Clustering | Day 43
Moved to Unsupervised Learning and studied about Clustering.
Working on my website check it out [avikjain.me](http://www.avikjain.me/)
Also found a wonderful animation that can help to easily understand K - Means Clustering [Link](http://shabal.in/visuals/kmeans/6.html)

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%2043.jpg">
</p>

## K Means Clustering Implementation | Day 44
Implemented K Means Clustering. Check the code [here.]()

## Digging Deeper | NUMPY  | Day 45
Got a new book "Python Data Science HandBook" by JK VanderPlas Check the Jupyter notebooks [here.](https://github.com/jakevdp/PythonDataScienceHandbook)
<br>Started with chapter 2 : Introduction to Numpy. Covered topics like Data Types, Numpy arrays and Computations on Numpy arrays.
<br>Check the code - 
<br>[Introduction to NumPy](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.00-Introduction-to-NumPy.ipynb)
<br>[Understanding Data Types in Python](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.01-Understanding-Data-Types.ipynb)
<br>[The Basics of NumPy Arrays](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb)
<br>[Computation on NumPy Arrays: Universal Functions](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.03-Computation-on-arrays-ufuncs.ipynb)

## Digging Deeper | NUMPY | Day 46
Chapter 2 : Aggregations, Comparisions and Broadcasting
<br>Link to Notebook:
<br>[Aggregations: Min, Max, and Everything In Between](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.04-Computation-on-arrays-aggregates.ipynb)
<br>[Computation on Arrays: Broadcasting](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.05-Computation-on-arrays-broadcasting.ipynb)
<br>[Comparisons, Masks, and Boolean Logic](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.06-Boolean-Arrays-and-Masks.ipynb)

## Digging Deeper | NUMPY | Day 47
Chapter 2 : Fancy Indexing, sorting arrays, Struchered Data
<br>Link to Notebook:
<br>[Fancy Indexing](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.07-Fancy-Indexing.ipynb)
<br>[Sorting Arrays](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.08-Sorting.ipynb)
<br>[Structured Data: NumPy's Structured Arrays](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.09-<br>Structured-Data-NumPy.ipynb)

## Digging Deeper | PANDAS | Day 48
Chapter 3 : Data Manipulation with Pandas
<br> Covered Various topics like Pandas Objects, Data Indexing and Selection, Operating on Data, Handling Missing Data, Hierarchical Indexing, ConCat and Append.
<br>Link To the Notebooks:
<br>[Data Manipulation with Pandas](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.00-Introduction-to-Pandas.ipynb)
<br>[Introducing Pandas Objects](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.01-Introducing-Pandas-Objects.ipynb)
<br>[Data Indexing and Selection](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.02-Data-Indexing-and-Selection.ipynb)
<br>[Operating on Data in Pandas](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.03-Operations-in-Pandas.ipynb)
<br>[Handling Missing Data](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.04-Missing-Values.ipynb)
<br>[Hierarchical Indexing](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.05-Hierarchical-Indexing.ipynb)
<br>[Combining Datasets: Concat and Append](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.06-Concat-And-Append.ipynb)

## Digging Deeper | PANDAS | Day 49
Chapter 3: Completed following topics- Merge and Join, Aggregation and grouping and Pivot Tables.
<br>[Combining Datasets: Merge and Join](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.07-Merge-and-Join.ipynb)
<br>[Aggregation and Grouping](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.08-Aggregation-and-Grouping.ipynb)
<br>[Pivot Tables](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.09-Pivot-Tables.ipynb)

## Digging Deeper | PANDAS | Day 50
Chapter 3: Vectorized Strings Operations, Working with Time Series
<br>Links to Notebooks:
<br>[Vectorized String Operations](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.10-Working-With-Strings.ipynb)
<br>[Working with Time Series](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.11-Working-with-Time-Series.ipynb)
<br>[High-Performance Pandas: eval() and query()](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.12-Performance-Eval-and-Query.ipynb)

## Digging Deeper | MATPLOTLIB | Day 51
Chapter 4: Visualization with Matplotlib 
Learned about Simple Line Plots, Simple Scatter Plotsand Density and Contour Plots.
<br>Links to Notebooks: 
<br>[Visualization with Matplotlib](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.00-Introduction-To-Matplotlib.ipynb)
<br>[Simple Line Plots](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.01-Simple-Line-Plots.ipynb)
<br>[Simple Scatter Plots](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.02-Simple-Scatter-Plots.ipynb)
<br>[Visualizing Errors](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.03-Errorbars.ipynb)
<br>[Density and Contour Plots](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.04-Density-and-Contour-Plots.ipynb)

## Digging Deeper | MATPLOTLIB | Day 52
Chapter 4: Visualization with Matplotlib 
Learned about Histograms, How to customize plot legends, colorbars, and buliding Multiple Subplots.
<br>Links to Notebooks: 
<br>[Histograms, Binnings, and Density](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.05-Histograms-and-Binnings.ipynb)
<br>[Customizing Plot Legends](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.06-Customizing-Legends.ipynb)
<br>[Customizing Colorbars](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.07-Customizing-Colorbars.ipynb)
<br>[Multiple Subplots](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.08-Multiple-Subplots.ipynb)
<br>[Text and Annotation](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.09-Text-and-Annotation.ipynb)

## Digging Deeper | MATPLOTLIB | Day 53
Chapter 4: Covered Three Dimensional Plotting in Mathplotlib.
<br>Links to Notebooks:
<br>[Three-Dimensional Plotting in Matplotlib](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.12-Three-Dimensional-Plotting.ipynb)

## Hierarchical Clustering | Day 54
Studied about Hierarchical Clustering.
Check out this amazing [Visualization.](https://cdn-images-1.medium.com/max/800/1*ET8kCcPpr893vNZFs8j4xg.gif)
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%2054.jpg">
</p>


## Contributions
Contributions are most welcomed.
 1. Fork the repository.
 2. Commit your *questions* or *answers*.
 3. Open **pull request**.
