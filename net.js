
//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = 0;
patchesAhead = 1;
patchesBehind = 0;
trainIterations = 10000;

var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 3;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

var layer_defs = [];
layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
layer_defs.push({
    type: 'fc',
    num_neurons: 1,
    activation: 'relu'
});
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 64,
    l2_decay: 0.01
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 3000;
opt.start_learn_threshold = 500;
opt.gamma = 0.7;
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.0;
opt.epsilon_test_time = 0.0;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
    brain.backward(lastReward);
    var action = brain.forward(state);

    draw_net();
    draw_stats();

    return action;
}

//]]>
    
/*###########*/
if (brain) {
brain.value_net.fromJSON({"layers":[{"out_depth":19,"out_sx":1,"out_sy":1,"layer_type":"input"},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":19,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":19,"w":{"0":-0.4363561485837658,"1":0.10085454570741255,"2":0.06395231017981753,"3":-0.09976337553720888,"4":-0.10155003050752201,"5":-0.151381436138858,"6":-0.2015498714150042,"7":-0.15950195563258052,"8":-0.034452188898474914,"9":0.17340910993022163,"10":0.14466688539897504,"11":-0.12818922761900722,"12":0.2822181065409149,"13":0.1976183398602759,"14":0.22394032866421312,"15":0.0035461596992068037,"16":0.34882667309222287,"17":0.3193579380034829,"18":0.0186046397326188}}],"biases":{"sx":1,"sy":1,"depth":1,"w":{"0":0.1}}},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"relu"},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":1,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":1,"w":{"0":0.9495613653622288}},{"sx":1,"sy":1,"depth":1,"w":{"0":0.21456282758090914}},{"sx":1,"sy":1,"depth":1,"w":{"0":-0.2651255907574694}},{"sx":1,"sy":1,"depth":1,"w":{"0":-1.4610904806907499}},{"sx":1,"sy":1,"depth":1,"w":{"0":1.8098943833510168}}],"biases":{"sx":1,"sy":1,"depth":5,"w":{"0":0,"1":0,"2":0,"3":0,"4":0}}},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"regression","num_inputs":5}]});
}