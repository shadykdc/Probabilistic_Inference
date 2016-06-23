# Probabilistic_Inference

This is just my coursework from my Intelligent Data and Probabilistic Inference course.

There are some data files necessary to run the code which can be found [here](https://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/Coursework/).

[Here](https://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/Bayesian.html) is a link to the course material.

## Coursework Part 1: The Naive Bayesian Network
Calculate joint and conditional probability tables (link matrices) and make inferences with a naive Bayesian network.

### Functions:
* **Prior:** calculates the prior distribution over the states of the variable passed as the parameter 'root' in the data array.
* **CPT:*** calculates the conditional probability table (link matrix) between two variables.
* **JPT:** calculates the joint probability (frequency of a state pair occurring in the data) table of any two variables.
* **JPT2CPT:** calculates a conditional probability table from a joint probability table.
* **Query:** calculates the probability distribution over the root node of a naive Bayesian network. To represent a naive network in Python, a list containing an entry for each node (in numeric order) is used giving the associated probability table: [prior, cpt1, cpt2, cpt3, cpt4, cpt5]. A query is a list of the instantiated states of the child nodes, for example [1,0,3,2,0]. The returned value is a list (or vector) giving the posterior probability distribution over the states of the root
node, for example [0.1,0.3,0.4,0.2].


## Coursework Part 2: The Maximally Weighted Spanning Tree
Use the spanning tree algorithm to find a singly connected Bayesian network from a data set.

### Functions:
* **MutualInformation:** calculates the mutual information (or Kullback Leibler divergence) of two variables from their joint probability table.
* **DependencyMatrix:** 
Complete the function “DependencyMatrix” which uses mutual information as a measure and creates a symmetric
matrix showing the pairwise dependencies between the variables in a data set.
Task 2.3: 6 marks
Complete the function “DependencyList” which turns the dependency matrix into a list or arcs ordered by their
dependency. Your list items should be triplets: [dependency, node1, node2]. Using your dependency list draw
by hand the network for the HepatitisC data set. Make an image file of your network (either by scanning your
hand drawing or using a drawing package (eg powerpoint or open office)).
Task 2.4: 6 marks (very difficult)
Starting with a dependency list find the maximally weighted spanning tree automatically, and append it as a list
to your results file. Don’t wory about finding the causal directions.
