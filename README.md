[![HW7-Regression](https://github.com/christine-wan/HW7-Regression/actions/workflows/test.yml/badge.svg?event=push)](https://github.com/christine-wan/HW7-Regression/actions/workflows/test.yml)

# HW 7: Logistic regression

In this assignment, you'll implement a classifier using logistic regression, optimized with gradient descent.

## Overview

In class, we went over an implementation of linear regression using gradient descent. For this homework, you will be implementing a logistic regression model using the same framework. Logistic regression is useful for binary classification because the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) outputs a value between 0 and 1.

In this repository, you are given a set of [simulated](https://doi.org/10.1093/jamia/ocx079) medical record data from patients with small cell and non-small cell lung cancers. Your goal to apply a logistic regression classifier to this dataset, predicting whether a patient has small cell or non-small cell lung cancer based on features of their medical record prior to diagnosis.

### Logistic regression

As stated above, logistic regression involves using a [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to model the data. Just like in linear regression, we will define a loss function to keep track of how well the model performs. But instead of mean-squared error, you will implement the binary cross entropy loss function. This function minimizes the error when the predicted y is close to an expected value of 1 or 0. Here are some resources to get you started: [[1]](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a), [[2]](https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11), [[3]](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html).

### Dataset 

You will find the full dataset in `data/nsclc.csv`. Class labels are encoded in the `NSCLC` column of the dataset, with 1 = NSCLC and 0 = small cell. A set of features has been pre-selected for you to use in your model during testing (see `main.py`), but you are encouraged to submit unit tests that look at different features. The full list of features can be found in `logreg/utils.py`.   

## Tasks + Grading
* Algorithm Implementation (6):
  * Complete the `make_prediction` method (2)
  * Complete the `loss_function` method (2)
  * Complete the `calculate_gradient` method (2)
* Unit tests (3):
  * Unit test for `test_prediction`
  * Unit test for `test_loss_function`
  * Unit test for `test_gradient`
  * Unit test for `test_training`
* Code Readability (1)
* Extra credit (0.5)
  * Github actions and workflows (up to 0.5)

## Getting started

Fork this repository to your own GitHub account. Work on the codebase locally and commit changes to your forked repository. 

You will need following packages:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pytest](https://docs.pytest.org/en/7.2.x/)

## Additional notes

Try tuning the hyperparameters if you find that your model doesn't converge. Too high of a learning rate or too large of a batch size can sometimes cause the model to be unstable (e.g. loss function goes to infinity). If you're interested, scikit-learn also has some built-in [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) that you can use for testing.

We're applying a pretty simple model to a relatively complex problem here, so you should expect your classifier to perform decently but not amazingly. It's also possible for a given optimization run to get stuck in a local minima depending on the initialization. With that said, if your implementation is correct and you found reasonable hyperparameters, you should almost always at least do better than chance.
