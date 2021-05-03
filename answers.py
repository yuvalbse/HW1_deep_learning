r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

**In kNN, increasing k at first will most likely improve the generalization for unseen data, but too large k will damage 
the accuracy of the model.
Looking on too small environment around specific sample might be highly affected by noise, e.g. by extremal and 
un-representative samples from other wrong classes. On the other hand, by taking into account too many neighbors we 
might consider samples that are not close in their properties (and thus, probably, class) to our predicted sample and 
predict wrong. Moreover, if we'll look on a bad distributed dataset, where on class is much more frequent, in the case 
of too big environment, we'll predict the same frequent label for all the samples.
In conclusion, in cases of too low k we might tackle over-fitting of the training set, and in cases of too high k we 
might under-fit to the training set. The best k will be the one that balance these too, and we can assert that by using 
validation set as we've done in this assignment.**

"""

part2_q2 = r"""
**Your answer:**

**1. Using k-fold CV is better than training on the entire train-set with various models and selecting the best model 
with respect to train-set accuracy because in such cases we will be at high risk to over-fit our model to the train-set 
because the evaluation is made on the same training set.**

**2. Using k-fold CV is better than training on the entire train-set with various models and selecting the best model 
with respect to test-set accuracy because it improves the generalization of the model, by testing few different 'mini 
test sets' (the k different validation sets). By that you achieve a more robust model that will probably fit better to
more unseen samples.**

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

**The ideal pattern for residual plot is when all the test samples are correctly labeled, so that $y-\hat{y}=0$. 
That implies that all the points will seat on the line y=0.**

**According to that, we can conclude that after the CV we got a better trained model than with the 5 best features, 
because most of the test set samples are close to $y-\hat{y}=0$, and it support the statistical (MSE) conclusion.**

"""

part4_q2 = r"""
**Your answer:**

**1. The effect of adding non-linear features to the data *does not* imply the model isn't linear anymore. That is 
because the relation between the weights and the features to the prediction is still linear ($W^{T}X$)**

**2. No. Non-linear functions that can't be represented as super position of linear and non-linear functions, but only 
with non-linear relation between the features, are not good candidates for this approach.**

**3. The decision boundary will be an hyperplane in the new domain, $\phi(x_{1},...,x_{n})$, because, as we mentioned 
before, it is still a linear model. 
However, when we return to the original domain, $(x_{1},...,x_{n})$, it is not linear hyperplane anymore.**
**
"""

part4_q3 = r"""
**Your answer:**

**1. In CV we are trying to tune our model with the best hyper-parameters, but we can't feed it with all the 
possibilities, because they are endless. So, to understand the regularization effect on our data, we are giving 
different order of magnitude of $\lambda$s. In that manner, we can later repeat the learning process with different 
values of the best order of magnitude we found and it can be done with np.linspace.**

**2. 3 k-fold with 3 different degrees and 20 different $\lambda$s -> 180 times during the CV.**
"""

# ==============
