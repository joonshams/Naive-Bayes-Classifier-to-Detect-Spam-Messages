# Naive-Bayes-Classifier-to-Detect-Spam-Messages
This MATLAB example demonstrates how a Naive Bayes classifer can be used to detect spam text messages using a training dataset

Naïve Bayes Training Model
To classify a test sample text message as spam or not spam, I take the Naïve Bayes’ approach. In this approach, the feature vector is x_i∈{0,1}^(n_i )
for n_i number of words in the i-th text message. Using this definition and assuming that the conditional probability of x is independent
of each other given target y, we can estimate the probability of a message being spam or not (p(x|y) from this assumption. That is, 

p(x_1…x_n│y)=p(x_1│y)p(x_2│y)…p(x_n |y)

Instead of

p(x_1…x_n│y)=p(x_1│y)p(x_2│y,x_1 )…p(x_n |y,x_1,x_2,..x_(n-1))

The implications of this assumption are that once you classify a text message as spam or not, the probability of each spam word 
in the text message is independent from the rest of the words that make up that message. Obviously, this can be a problematic assumption ergo the name of the model. 
To train our model using the Naïve Bayes approach, I estimate p(y), p(x|y=1), and p(x|y=0) using the training model.
For p(y=1), I just estimate the ratio of spam messages to the total messages in our training model. Subsequently, p(y=0) is estimated as 1-p(y=1). 

Lastly, the probabilities of the text message being spam or not are compared, the greater probability overrides the other and the corresponding target label is generated for the test sample. 
>> spamprob=(px4ys)*(py)

spamprob =

    0.1350

>> notspamprob=(px4yns)*(1-py)

notspamprob =

    0.8657

One of the obvious limitations is that for a test sample with equal probabilities of being spam or not spam (both ~0.5) can become problematic. If I were to update the model, I would represent the target as a probability metric instead of a binary label. This would allow for possibly a more informed approach to identify if a text is spam or not. Additionally, I use a generative approach to solve this problem but one could use a discriminative approach like logistic regression. The logistic regression might work better if the Gaussian assumption is not true for our training data.   
The accuracy of the training model when tested on the entire training dataset is estimated to be 0.73 and 0.78 if I didn’t remove the vowels ‘a’, ‘e’, ‘i’. Since the accuracy is not 1 or close to 1, I assume that the model is not over-fitting. There are different permutations of characters one can add or remove and each one would yield a different accuracy. The goal of this script is to present proof of concept as opposed to optimizing the NB model for best prediction accuracy. 
