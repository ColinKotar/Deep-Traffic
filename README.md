# Deep-Traffic
My best Deep Traffic network for a self-navigating car

## Introduction
This network is for MIT's Deep Traffic simulator:
http://selfdrivingcars.mit.edu/deeptrafficjs/

## Neural Network

lanesSide = 3

patchesAhead = 4

patchesBehind = 0

trainIterations = 50000


Input Layer: 127

FC Layer: 30, ReLU

FC Layer: 10, ReLU

Output Layer: 5

### Hyperparameters

learning_rate: 0.01

momentum: 0.00

batch_size: 50

l2_decay: 0.01


opt.experience_size = 4000

opt.start_learn_threshold = 100

opt.gamma = 0.7

opt.learning_steps_total = 10000

opt.learning_steps_burnin = 1000

opt.epsilon_min = 0.0

opt.epsilon_test_time = 0.0
