# Introduction to Deep Learning with ConvNetJS at Holberton School

Deep Learning Day 1 by Louis Monier and Gregory Renard.
https://www.holbertonschool.com/

## Requirements

- Download the GitHub repository : **Precise the URL of the GitHub**
- Open the web page in a browser : **ConvNetJS-HS-Demo.html**

We recommend you use Chrome as your browser.

## Task 1 : Observation

![Start](https://github.com/gregrenard/hs/blob/master/ConvNetJS/images/capture1.png)

As you open the page, the Network starts to train itself.
You can observe 3 modules on the page :

1. the setup of a neural network (#1 = top)
2. the graphic representation of the distribution of points (#2 = bottom left)
3. the transformed representation of all grid points at the output of two neurons in a given layer. (#3 = bottom right)
 
The 1st module (Setup of the Neural Network) presents the structure of the current Network.  Every Network is a linear list of layers. The order matters, as a layer is connected to the layers described on the previous and next lines.
The 1st layer must be the 'input' (where you declare the sizes of your input data), the last layer must be a loss layer ('softmax' or 'svm' for classification, or 'regression' for regression)



Below the description of the code at the initial load of the page :


```javascript

// species a 1-layers neural network with one hidden layer of 1 neurons
layer_defs = [];

//input layer declares size of input. here: 2-D data
// ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
// then the first two dimensions (sx, sy) will always be kept at size 1
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});

// declare 1 neurons, followed by a tanh
layer_defs.push({type:'fc', num_neurons:1, activation: 'tanh'});

// declare the linear classifier on top of the previous hidden layer
layer_defs.push({type:'softmax', num_classes:2});

// Network instantiation and layers definition
net = new convnetjs.Net();
net.makeLayers(layer_defs);

// species the trainer as a SGD+Momentum trainer.  Performs a weight update every 10 examples (batch size)
trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, momentum:0.1, batch_size:10, l2_decay:0.001});

```

Notice that this simple network separates the points in 2 regions through a line only.
Remember to click on the button "change network" after editing the Network.

## Task 2 : Add a point

As you're seeing the training of the Network, you can click at any place of the graph to add a new red or green point.
- CLICK: Add red data point
- SHIFT+CLICK: Add green data point
- CTRL+CLICK: Remove closest data point

You can observe the classification's evolution.

## Task 3 : Change the distribution shape 

Now try more complex distribution shapes by clicking on the buttons : simple, circle, spiral, ring, check, spots or target.

![More Neurons](https://github.com/gregrenard/hs/blob/master/ConvNetJS/images/capture2.png)

You can observe the difficulty of your current Network to identify the right classification model as it is limited to a separating with a single line.  

**Solve it :** Increase the number of neurons "num_neurons:1" to 2, 3, 5... in the layer type 'fc' and try again the different shapes.

*Remember to click the button **"Change network"** to apply your modification !*

```javascript
// species a 1-layers neural network with one hidden layer of 5 neurons
layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});

// declare 5 neurons, followed by a tanh
layer_defs.push({type:'fc', num_neurons:5, activation: 'tanh'});

layer_defs.push({type:'softmax', num_classes:2});

```

You can observe that some shapes (like the circle, or the check mark) are easier for the Network to model than others.

![alt tag](https://github.com/gregrenard/hs/blob/master/ConvNetJS/images/capture3.png)


Exercices :
- Write down how many neurons it takes for the network to master each shape in less than a minute.
- How many neurons does it take to get the "target" to converge quickly?

How can we optimize our Network ?

## Task 4 : Add layers to your Network

Now you can try to add a layer to your Network with the replication of the whole line with type 'fc' and change the number of neurons to 10 in each line.

```javascript
// species a 2-layers neural network with one hidden layer of 10 neurons
layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});

// declare 10 neurons, followed by a tanh
layer_defs.push({type:'fc', num_neurons:10, activation: 'tanh'});
layer_defs.push({type:'fc', num_neurons:10, activation: 'tanh'});

layer_defs.push({type:'softmax', num_classes:2});

```

After replicated the whole line with "num_neurons:". Now you have a 2-layer network.

![Add Layers](https://github.com/gregrenard/hs/blob/master/ConvNetJS/images/capture4.png)

Exercise:
- **How quickly can it crack the "target"?**
- Compare to the largest single-layer network you have tried on the target.
