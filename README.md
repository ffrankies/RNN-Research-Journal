# RNN-Research-Journal

Online journal for my RNN independent study and research class. Documents my
progress and the resources I use.

---

## Table of Contents

*   [Objectives](#objectives)
*   [Log](#log)
*   [Resources](#resources)

---

<a name="objectives"/>

## Objectives

This project's objectives are, in relative order:

### Learning Outcomes

1.  Learn about Neural Networks and their applications.
2.  Learn how to use the Theano framework for python to build Neural Networks.
3.  Learn how Recurrent Neural Networks (RNNs) work.
4.  Learn how to use Recurrent Neural Networks to generate legible text.
5.  Learn how to optimize RNN performance.

### Coding Milestones

1.  ~~Build a single neuron that 'learns' via stochastic gradient descent.~~
2.  ~~Build a simple classifying Neural Network with no hidden layers.~~
3.  ~~Build a simple classifying Neural Network with a hidden layer.~~
4.  ~~Build a single-layer "vanilla" RNN for text generation.~~
5.  Build a single-layer RNN with GRU units for remembering long-term
    dependencies.
6.  Optimize the previously built RNN for faster training.
7.  Implement word embeddings in the optimized RNN.
8.  Implement multiple hidden layers in the optimized RNN with word embeddings
    for better performance.
9.  Build "Terry" - an RNN that outputs short-story-sized text that hopefully
    makes sense.
10. If time permits, optimize Terry for better performance.

### Other
*   Present a poster or presentation on Student Scholars Day.s

---

<a name="log"/>

## Log

### Jan 09 2017

*   Started the research journal.
*   Training RNN with GRU units in the hidden layer. Expected training time:
    ~40 hours.
*   Still don't quite understand how the numpy.random.multinomial() function
    works in randomizing the output. The parameters used seem wrong, but work,
    so I'm definitely missing something here.

### Jan 10 2017

*   Couldn't log in to seawolf, connection timed out.
*   Worked on journal.

---

<a name="resources"/>

## Resources

*   [Tutorials](#tutorials)
*   [Books](#books)
*   [Other](#other)

<a name="tutorials"/>

### Tutorials

These are the tutorials I followed along. Not all of these were completed
from start to finish, but they were all used up to a point.

#### WildML RNN Tutorial [link](http://goo.gl/gHSQYm)

*   Author: Denny Britz
*   A 4-part tutorial for coding up an RNN using Theano. The resulting RNN
    model is not optimized, and the dataset link used by the tutorial did not
    work (for me, at least).
*   Contains many useful links to other RNN resources.
*   I found the variable naming here very un-intuitive, but otherwise it was a
    great resource.

#### Basic Theano Tutorial [link](http://goo.gl/TN8qVD)

*   I believe this is the official Theano tutorial.
*   Contains sections on how to install Theano and how to use it.
*   Relatively easy to follow, not so easy to use it for building a neural
    network without further tutorials.
*   It may take a while to figure out when to use plain python, and when to
    use the Theano library.

#### Deep Learning with Theano Tutorial [link](http://goo.gl/mp5zMA)

*   Tutorial from [DeepLearning.net](http://deeplearning.net)
*   Probably the most complete Deep Learning tutorial out there.
*   Contains tutorials for building many different types of Neural Networks,
    from plain to Convolutional to Recurrent.
*   Also contains datasets and code to follow along with the tutorial.
*   I found the tutorials here harder to follow than the WildMl RNN tutorial.

<a name="books"/>

### Books

#### The Deep Learning Textbook [link](http://www.deeplearningbook.org)

*   Authors: Ian Goodfellow, Yoshua Bengio, Aaron Courville
*   This is an in-depth look at Deep Learning and various Deep Neural Network
    Models.
*   It is pretty heavy reading, and generally hard to follow, at least from the
    perspective of an undergrad with almost no Machine Learning experience.
*   I switched from the book to the tutorials pretty early because they were
    easier to follow, and because I prefer a hands-on approach to learning.

<a name="other"/>

### Other Resources

#### Andrej Karpathy's Blog Post on RNNs [link](https://goo.gl/kIzxwg)

*   Author: Andrej Karpathy
*   Title: The Unreasonable Effectiveness of Recurrent Neural Networks
*   Has a great non-technical overview of how RNNs work, especially with
    regards to text generation.
*   Contains lots of examples of text generated by RNNs.
*   Has links to code for an RNN with LSTM units, but it is written using Lua.
*   Other great examples of stuff generated with RNNs can be found in the
    comments.
