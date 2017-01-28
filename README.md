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
*   Present a poster or presentation on Student Scholars Days.

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

### Jan 13 2017

*   Updated journal with all the resources used up to this point (that I could
    remember)

### Jan 15 2017

*   Found a new resource - a github post by Karpathy on more efficient
    gradient descent.
*   Takeaways:
    *   Use Nesterov Momentum / Adam to optimize gradient descent (might want
        to try using RMSPROP first, since that is demonstrated in the blog).
    *   To test whether gradient descent works, attempt to achieve 100%
        accuracy on a small sample of data.
*   Found another resource - a blog post by Sebastian Ruder on word embeddings.
*   Takeaways:
    *   Word embedding is the practice of reducing the vocabulary of a
        language model to a vector of a smaller dimension that represents the
        any given word in the vocabulary.
    *   Essentially, it is another layer inside the neural network that
        represents any given word with a vector.
    *   This can either be trained together with the network, as the initial
        layer (thereby adding a new layer to the network, thus significantly
        slowing down training), or pre-trained.
    *   Training in tandem with the network makes the word embeddings more
        specific to the task at hand, while pre-trained word embeddings
        increase the accuracy of the model without a high training time cost.

### Jan 18 2017

*   Started creation of a testing suite for the models.
*   Attempting to change how logging is done in my Neural Networks, by
    switching from print statements to the python logging module.
*   Attempted to install g++ through cygwin to speed up theano implementations,
    got a long traceback error that I couldn't make sense of. Removed g++.exe
    path from environment PATH variable, error did not return.

### Jan 19 2017

*   Worked on test suite. It performs operations on multiple networks
    simultaneously by utilizing multiple threads.
*   Test suite should now be able to test whether gradient descent works,
    train networks and generate sentences from the networks.
*   Test suite is only partially tested, but appears to work as expected.
*   Logging using the python logging module works as expected as well.

### Jan 20 2017

*   Coded up another RNN, that implements both GRU hidden units and the Adam
    algorithm for gradient descent. Cannot test at this time.
*   I chose Adam over RMSPROP, which was used in the tutorial, because the
    source listed in the tutorial recommended it over the other methods.
*   The Adam algorithm should make gradient descent converge to optimum values
    much quicker than before.
*   Reservation 1 - Adam introduces a lot of new parameters to the network,
    which makes network models take up much more space
*   Reservation 2 - There are much more parameters to update, and more matrix
    operations, so while the network might converge to optimum parameter values
    in fewer iterations/epochs, chances are each epoch will take longer to
    train.

### Jan 22 2017

*   Added code for RNN with Adam to save new parameters as well
    during training.
*   Started to enable command line arguments for the test suite using the
    argparse module.
*   Minimum goal here is to be able to specify which tests to carry out and
    which RNNs to carry them out on using command line arguments.
*   Optimum goal is to be able to specify a json config file that includes
    RNN hyperparameters as well.

### Jan 23 2017

*   Gained access to seawolf.
*   Installed tmux locally as an alternative to using nohup.
*   Got working tmux installation instructions [here](https://goo.gl/Tn1TfV)
*   Installed anaconda version 4.2.9 for python 3.5 and other libraries.
*   Running Theano with either python 3.5 or 2.7 gives a 5000+ line warning,
    but appears to work anyway. Error did not show up on previous os.
*   Restructured the way updates were made in sgd_step in the Adam RNN
    implementation. Starting to think that using subtensors wasn't such a good
    idea - code looks unnecessarily verbose.
*   Got errors when using the testSuite. Could not get error to go down to 0 on
    models. AdamRNN failed testing of gradient descent, but for some reason is
    the only network that is training on full dataset.

### Jan 24 2017

*   Focusing on VanillaRNN and the test suite.
*   AdamRNN still training on first epoch at 20:29 - so it takes over a day to
    train a single epoch. Might have to switch to Nesterov Momentum instead.
*   Started training a VanillaRNN for 10 epochs. Training happening relatively
    slowly.
*   Ran the GPU test available in the theano documentation, and it turns out
    the GPU optimization is not working.
*   With THEANO_FLAGS=device=gpu, I get the errors:
    -   ERROR (theano.sandbox.cuda): Failed to compile cuda_ndarray.cu: ...
    -   ERROR (theano.sandbox.cuda): CUDA is installed, but device gpu is not
        available.
*   With THEANO_FLAGS=device=cuda, I get the error:
    -   ERROR (theano.sandbox.cuda): pygpu was configured but could not be
        imported.

### Jan 28 2017

*   May have figured out why error wasn't going down when using training suite:
    I changed the number of examples used to 20, but left patience at 100000.
    

---

<a name="resources"/>

## Resources

*   [Tutorials](#tutorials)
*   [Books and Blog Posts](#books)
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

### Books and Blog Posts

#### Andrej Karpathy's Blog Post on RNNs [link](https://goo.gl/kIzxwg)

*   Author: Andrej Karpathy
*   Title: The Unreasonable Effectiveness of Recurrent Neural Networks
*   Has a great non-technical overview of how RNNs work, especially with
    regards to text generation.
*   Contains lots of examples of text generated by RNNs.
*   Has links to code for an RNN with LSTM units, but it is written using Lua.
*   Other great examples of stuff generated with RNNs can be found in the
    comments.

#### The Deep Learning Textbook [link](http://www.deeplearningbook.org)

*   Authors: Ian Goodfellow, Yoshua Bengio, Aaron Courville
*   This is an in-depth look at Deep Learning and various Deep Neural Network
    Models.
*   It is pretty heavy reading, and generally hard to follow, at least from the
    perspective of an undergrad with almost no Machine Learning experience.
*   I switched from the book to the tutorials pretty early because they were
    easier to follow, and because I prefer a hands-on approach to learning.

#### Understanding LSTM Networks [link](https://goo.gl/pknkFU)

*   Author: Christopher Olah
*   A blogposts explaining LSTMs, and other methods to reduce problem of
    learning long-term-dependencies on RNNs.
*   It has some good diagrams to go along with the explanations, as well as
    links to relevant research papers and other resources.

#### Gradient Descent Notes [link](https://goo.gl/ItqfGz)

*   Author: Andrej Karpathy
*   These appear to be notes for a Standford Deep Learning Class.
*   Contains information for optimizing gradient descent, as well as tips
    for picking hyperparameters and testing that the gradient descent works.
*   Advises to train 'model ensembles' - a number of independent models whose
    predictions are averaged at test time. This has some computational cost,
    but can also improve network accuracy, especially when a variety of models
    is used.

#### On Word Embeddings [link](http://sebastianruder.com/word-embeddings-1/)

*   Author: Sebastian Ruder
*   This is a two-part blogpost essentially summarizing various research into
    word embeddings.
*   Explains what word embeddings are and what they are used for, providing
    mathematical equations when necessary.
*   Explains how word embedding layers for various natural language processing
    tasks are trained, including a table comparing the different methods on
    efficiency and performance on small and large vocabulary datasets.

<a name="other"/>

### Other Resources

#### Coursera's Machine Learning Course [link](https://goo.gl/XhLwLv)

*   Author: Andrew Ng
*   Coursera's Machine Learning course, which can be taken for free, teaches
    various machine learning concepts and algorithms using MATLAB's free
    alternative: Octave.
*   I took the free version of the course, and followed it until the Neural
    Networks chapters.
*   I found Octave more tedious than python, and Theano abstracts a lot of the
    operations that had to be implemented manually in Octave.

#### Udacity's Intro to Machine Learning Course [link](https://goo.gl/QkrKqg)

*   Authors: Katie Malone and Sebastian Thrun
*   This course taught how to use python with numpy to write various machine
    learning functions.
*   It was a good course, but I only followed it through half of the Support
    Vector Machines chapter, since it had nothing on Neural Networks.
