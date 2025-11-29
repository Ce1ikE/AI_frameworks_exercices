
from fastai.vision.all import *
from fastai.vision.learner import *
from fastai.text.all import *
from fastai.tabular.all import *
from fastai.callback import *


from pathlib import Path
RESULTS = Path("results")

## Exercises

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 1 - What two steps does the `fine_tune` method do?

# well it is quite more complex then i thought because when call the function vision_learner
# a Learner object is returned (pretty obvious) but this object has a no method called fine_tune
# when checking the source code of this object.
# when we read a StackOverflow post (https://stackoverflow.com/questions/76608775/fast-ai-course-lesson-1-object-has-no-attribute-fine-tune)
# which in turn rederects us to the fastai source code (https://github.com/fastai/fastai/blob/b273fbb32d075ef1d6fd372687b5f56564cead9a/fastai/callback/schedule.py#L161)
# you can also find it inside your local installation of fastai library fastai/callback/schedule.py
# we can see that the fine_tune method is defined as follows:
"""
    @patch
    @delegates(Learner.fit_one_cycle)
    def fine_tune(self:Learner, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100,
                pct_start=0.3, div=5.0, **kwargs):
        "Fine tune with `Learner.freeze` for `freeze_epochs`, then with `Learner.unfreeze` for `epochs`, using discriminative LR."
        self.freeze() # 1
        self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs) # 2
        base_lr /= 2
        self.unfreeze()
        self.fit_one_cycle(epochs, slice(base_lr/lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)
"""
# what this method does is:
# 1 - it freezes the body of the model 
#     under the hood we can go down the rabbit hole freeze() -> freeze_to(-1) -> BaseOptimizer.freeze_to(-1) -> 
#     the gradients of the parameters are all set to True first and then the parameters of the model up to layer are set to False
#     
# 2 - it fits the model for freeze_epochs epochs with the body frozen thus it only trains the head of the model
#     it uses the fit_one_cycle ,which AGAIN is not a method of the Learner class but is patched to it
#     which fits the model using the 1cycle policy for the specified number of epochs
#     the params being optimized are learning and momentum 
#
# 3 - it unfreezes the body of the model so that all the parameters are trainable
#
# 4 - it fits the model for epochs epochs with the body unfrozen thus it trains both the body and the head of the model
#     it uses again the fit_one_cycle method
#     it uses discriminative learning rates by passing a slice object as learning rate
#     the slice object specifies a range of learning rates from base_lr/lr_mult to base_lr
#     this means that the layers closer to the input will have a lower learning rate and the layers closer to the output will have a higher learning rate
#     this is useful because the layers closer to the input are more general and thus require less fine-tuning
#     while the layers closer to the output are more specific and thus require more fine-tuning
# 
# to go back to the decorators this function 
# @patch is used in fastai to add methods to existing classes without modifying their source code
# @delegates is used to pass additional keyword arguments to the fit_one_cycle method


# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 2 - What are discriminative learning rates?

# well this kind of answered in the previous question
# discriminative learning rates is a technique where different layers of the model are trained with different learning rates
# typically, the layers closer to the input are trained with lower learning rates because they capture more general features
# while the layers closer to the output are trained with higher learning rates because they capture more specific features
# this is useful when fine-tuning a pre-trained model because the lower layers are already well-trained and do not require much adjustment
# while the higher layers need more fine-tuning to "adapt" to the new task
# discriminative (from the word to discriminate) means to differentiate or to make a distinction
# so in this context it means to differentiate the learning rates for different layers of the model


# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 3 - What method does fastai provide to view the data in a `DataLoaders`? 
# What is DataLoaders?
# What four things do we need to tell fastai to create DataLoaders?

# from fastai.data import *
# DataBlock
# https://docs.fast.ai/data.load.html

# to view the data in a DataLoaders object, we can use the show_batch() method,
# which displays a batch of data along with their corresponding labels

# a group of Dataset objects or a DataBlock objects that defines how to load and preprocess the data
# this is similar to PyTorch Dataset where you define how to get the data and labels with __getitem__ and __len__

# To create a DataLoaders object in fastai, we need to provide four main components:
# 1 - a source of data: This can be a list of file paths, a Pandas DataFrame, or any other iterable that contains the data we want to use
# 2 , 3 and 4 ??? (most parameters are optional) 

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 4 - What method does fastai provide to help you debug a `DataBlock`?

# DataBlock.summary() -> see fastai/data/block.py (function is pathched to DataBlock class)

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 5 - Give two examples of ways that image transformations can degrade the quality of the data.

# 1 - rotation: rotating an image can lead to loss of important features or context, especially if the rotation is extreme.
#     e.g.: rotating a picture of a cat upside down may make it harder to recognize
# 2 - cropping: cropping an image can remove important parts of the image, leading to loss of context or information.
#     e.g.:  cropping a picture of a dog may remove its tail or ears, making it harder to identify the breed

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 6 - What are the two pieces that are combined into cross-entropy loss in PyTorch?

# we kind of saw this in the previous lectures (see Lecture_3_Logistic_Regression/lib/LogisticRegression.py)
# where we used softmax to get the probabilities for multi-class classification
    # softmax(z_i) = exp(z_i) / sum(exp(z_j) for j = 1 to K)
    # z_i (logit) is the output of the previous iteration for class i
    # K is the number of classes
    # z_j is the logit for each class
    # this ensures that the output is a probability distribution over K classes (i.e. all values are between 0 and 1 and sum to 1)
    # to improve numerical stability, we subtract the max value from z before applying exp
# cross-entropy loss then compares these predicted probabilities to the true labels using the negative log likelihood
    # NLLLoss = -log(p) where p is the predicted probability of the true
# the reason we use the word "entropy" can be traced back to Math for AI:
    # "The entropy of a random variable is the average amount of information 
    # we can gain from the outcome that variable"
# It is also in this course we saw why we use a negative log likelihood:
    # "To understand why there is a negative sign, consider the function f(x) = log(x) for x in (0, 1]."
    # "Hence the higher the probability of X the higher the P(X). But intuitively it makes that values that appear less"
    # "should contain more information, as we learn more from a unusual case than from a common one. That is why we take the negative of the log"

# so in short steps cross-entropy loss in PyTorch combines:
# 1 - softmax function to convert logits to probabilities
# 2 - negative log likelihood to compare predicted probabilities to true labels

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 7 - What are the two properties of activations that softmax ensures? 
# Why is this important?

# kind of answered in the previous question
# softmax ensures two properties of activations:
# 1 - all output values are between 0 and 1
# 2 - the sum of all output values is equal to 1

# this is important because it allows us to interpret the output of the model as a probability distribution over the classes
# this is also usefull for multi-class classification problems where we want to assign a probability to each class
# it also helps in making a clear distinction between classes by emphasizing larger values and suppressing smaller ones
# this is especially useful when we have more than two classes to choose from

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 8 - Why can't we use `torch.where` to create a loss function for datasets where our label can have more than two categories?

# torch.where is like np.where you provide a condition and two tensors
# it returns a tensor where the condition is true from the first tensor and where the condition is false from the second tensor
# this is useful for binary classification where we have only two classes (e.g., 0 and 1)
# but in multi-class classification where we have more than two categories (e.g., 0, 1, 2, ..., K-1)
# we need to compute the loss for each class separately and then combine them
# using torch.where would require us to create multiple conditions and tensors for each class
# which would be inefficient and cumbersome + it's not dynamic
# e.g., for 3 classes we would need to create 3 conditions and 3 tensors
# torch.where(condition1, tensor1, torch.where(condition2, tensor2, tensor3))
# it would be much better to use a loss function that can handle multiple classes directly
# such as cross-entropy loss

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 9 - How is a Python `slice` object interpreted when passed as a learning rate to fastai?

# when a slice object is passed as a learning rate to fastai
# it is interpreted as a range of learning rates for different layers of the model
# the start of the slice corresponds to the learning rate for the layers closer to the input
# the stop of the slice corresponds to the learning rate for the layers closer to the output

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 10 - What will fastai do if you don't provide a validation set?

# if no validation set is provided fastai will create one for you
# it will take a random sample of the training data and use it as the validation set
# see fastai/learner/learner.py 
"""
    def one_batch(self, i, b):
        self.iter = i
        b = self._set_device(b)
        self._split(b)
        self._with_events(self._do_one_batch, 'batch', CancelBatchException)
"""
# and 
"""
    def _split(self, b):
        i = getattr(self.dls, 'n_inp', 1 if len(b)==1 else len(b)-1)
        self.xb,self.yb = b[:i],b[i:]
"""

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 11 - Can we always use a random sample for a validation set? Why or why not?

# not always
# in some cases a random sample may not be representative of the entire dataset
# e.g., in imbalanced datasets where one class is much more prevalent than others
# a random sample may not contain enough examples of the minority class
# leading to biased evaluation metrics
# in such cases it is better to use stratified sampling
# where the validation set is created by sampling from each class in proportion to the entire dataset
# this ensures that the validation set is representative of the entire dataset
# e.g.: in medical diagnosis datasets where the number of positive cases is much lower than negative cases
# a random sample may not contain enough positive cases
# leading to biased results

# scikit-learn -> train_test_split(stratify=y)

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 12 - What is the "head" of a model?

# the "head" of a model refers to the final layers of the model that are responsible for making predictions
# we also have the "body" which in contrast refers to the initial layers

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 13 - What is segmentation?

# segmentation is a computer vision task that involves dividing an image into multiple segments or regions
# each segment corresponds to a specific object or part of the image
# the goal of segmentation is to assign a label to each pixel in the image
# so that pixels with the same label belong to the same object or region
# there are two main types of segmentation:

# 1 - semantic segmentation: where each pixel is assigned a label corresponding to a class of objects
#     e.g., in an image of a street scene, all pixels belonging to cars
#     would be labeled as "car", all pixels belonging to pedestrians would be labeled as "pedestrian", etc.

# 2 - instance segmentation: where each pixel is assigned a label corresponding to a specific instance of an object
#     e.g., in the same street scene image, each car would be labeled with a unique identifier
#     so that we can distinguish between different cars in the image

# see Aurelien Geron's book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" chapter 14 pages 521-534

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 14 - Why is a GPU useful for deep learning? 
# How is a CPU different, and why is it less effective for deep learning?

# a GPU (Graphics Processing Unit) is useful for deep learning because it is 
# designed to handle parallel processing tasks efficiently (e.g., rendering graphics)
# deep learning involves performing a large number of matrix operations
# which can be parallelized effectively on a GPU
# this allows for faster training and inference times compared to using a CPU (Central Processing Unit)
# a CPU is designed to handle a wide range of tasks sequentially
# it has fewer cores optimized for single-threaded performance
# while a GPU has many more cores optimized for parallel processing
# this makes a CPU less effective for deep learning tasks that require processing large amounts of data simultaneously

# if we go a step beyond GPU we have TPU (Tensor Processing Unit) 
# these are specialized hardware accelerators designed specifically for deep learning workloads

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 15 - What is the difference between `item_tfms` and `batch_tfms`?

# tfms = transformations
# item_tfms are transformations that are applied to each individual item in the dataset
# before they are combined into batches
# e.g., resizing an image to a specific size or normalizing pixel values
# batch_tfms are transformations that are applied to the entire batch of items
# after they have been combined into batches
# e.g., data augmentation techniques like random cropping or flipping

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 16 - What is a confusion matrix?

# a cm is a table that is used to evaluate the performance of a classification model
# it shows the number of correct and incorrect predictions made by the model for each class
# the rows of the matrix represent the actual classes
# while the columns represent the predicted classes

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 17 - What does `export` save?

# https://fastai1.fast.ai/tutorial.inference.html#:~:text=export%20to%20save%20all%20the,in%20a%20file%20named%20export.
# export saves the entire Learner object including the model architecture
# the model weights
# the data processing pipeline

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 18 - How do we encode the dependent variable in a multi-label classification problem?

# using one-hot encoding
# where each label is represented as a binary vector
# e.g.:

# category | 
# -------- | -----
#    A     | 1 0 0
#    B     | 0 1 0
#    C     | 0 0 1

# one hot encoding increases the dimensionality of the data by n (where n is the number of unique categories)

# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 19 - Why can't we use regular accuracy in a multi-label problem?

# in multi-label classification problems
# each instance can belong to multiple classes simultaneously
# regular accuracy measures the proportion of correct predictions
# but in multi-label problems, a prediction is only considered correct if all labels are predicted correctly
# this means that if a model predicts 3 out of 4 labels correctly for an instance
# it would be considered incorrect according to regular accuracy
# this can lead to misleading results
# instead, we use metrics like F1-score, precision, recall
# which take into account the partial correctness of predictions

# Recall = TP / TP + FN (how many actual positives were identified correctly)
# Precision = TP / TP + FP (how many predicted positives were actually correct)


# ////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
# 20 - When is it okay to tune a hyperparameter on the validation set?

# it is okay to tune a hyperparameter on the validation set
# when we have a separate test set that is not used during the training or validation process
# this allows us to evaluate the final performance of the model on unseen data
# if we tune hyperparameters on the validation set without a separate test set
# we risk overfitting to the validation set (this called data leakage (see AI fundamentals))

# in CV (cross-validation) we can use the validation set for hyperparameter tuning
# because we average the performance across multiple folds

def main():
    pass

if __name__ == "__main__":
    main()

