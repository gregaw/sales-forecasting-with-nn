

# 'Improve the model' challenge

The challenge as advertised after the dataKrk meetup was about improving the model. 
More specifically: 
> whoever improves the github model most by 20th Dec 18:30 (as judged by the PR with your code changes and the MAE metrics on the development test set) wins a free ticket to data.sphere.it 

Very exciting.

By 20th December 2017 18:30, we had 4 PRs and various attempts at improving the model:
- getting a bigger network, 
- adding label and one-hot encoders, 
- adding additional features,
- bigger and broader windows,
- using different network architectures (CNN)

**WELL DONE!**

tada......

## The final ranking

Ranking (name/id_on_graph/: MAE):
1. msurdziel/03/:			0.022428641325932266
2. marcinkosztolowicz/02/:	0.043058410371890996
3. maciekbb/04/:			0.070445201781086139
4. lina2002/01/:			0.085867498296635231

NOTE: some solutions used more epochs, so we've cut it to 200.
It didn't impact the ranking anyhow (check the tensorboard graph if you're not convinced).

## Tensor board MAE graph

![Improve the model Challenge results graph](mae-graph.png?raw=true "Improve the model Challenge")

## Some details

---------- baseline: 00
*** Evaluation epoch[49] metrics [0.022819430682411553, 0.11011138396911295]

---------- lina2002: 01

gist: 

*** Evaluation epoch[199] metrics [0.018078467607703555, 0.085867498296635231]

---------- marcinkosztolowicz: 02

gist: add more windows (open, customers), increase epochs, bigger network

*** Evaluation epoch[2999] metrics [0.0040588357176865406, 0.043058410371890996]

---------- msurdziel: 03

gist: more features, bigger network, label and onehot encoding

*** Evaluation epoch[299] metrics [0.0011109148450735814, 0.022428641325932266]

---------- maciekbb: 04 (* changed to use more epochs - to be able to compare)

gist: CNN network with dropout, not using features from the day (like OPEN), lookback on all the features

*** Evaluation epoch[199] metrics [0.011424110350712511, 0.070445201781086139]


