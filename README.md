## Deep Learning Framework Implemented In Java
The purpose of this project is to create a deep learning library in Java. It is based on the Python implementation I've already done, in turn based on the work by Harrison Kinsley and Daniel Kukiela in their book Neural Networks From Scratch. That project was done in Python, and as a learning experience, I have implemented similar functionality in Java. The following section is the project structure, which shows the classes which have been implemented. This project may be useful to individuals interested in exploring how deep learning models work under the hood, are familiar with Java but may not be familiar with deep learning mathematics and methodologies. Each class name has a description, unless the name is self-explanatory. 

### Project Structure:
- src (root folder)
    - dtypes package
        - Tensor class: Object representing a standardized data type used by all classes in this project. Includes static functions for relevant mathematical operations. Analogous to arrays in Numpy or tensors in PyTorch. 
    - layers package
        - Abstract Layer Class
        - Dense Layer Class: Standard feedforward layer in a multilayer perceptron. 
        - Dropout Layer Class: Standard dropout layer used to help prevent model overfitting. 
        - InputLayer class: Necessary for the current implementation of the Model class. It is used to simplify the method of passing information through the model, but shouldn't really need to be used otherwise. 
    - activations package
        - RelU Class
        - Softmax Class
        - Sigmoid Class
        - Linear Class
    - losses package
        - Abstract Loss Class
        - Categorical Crossentropy Class
        - Categorical Crossentropy with Softmax Class
        - Binary Crossentropy Class
        - Mean Squared Error Class
        - Mean Absolute Error Class
    - metrics package
        - Accuracy Class
    - optimizers package
        - Abstract Optimizer Class
        - SGD Class
        - Adagrad Class
        - RMSprop Class
        - Adam Class
    - model package
        - Model class: A model object that simplifies the creation, training and use of a model. Made up of a series of layers, as well as an optimizer, loss and metrics. 
