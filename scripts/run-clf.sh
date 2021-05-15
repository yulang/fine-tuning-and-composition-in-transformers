#!/bin/sh

rand=52431
layers=13
WORKING_DIR=/path/to/phrasal/composition/src

# classification by layer
# python ./classifier.py --by_layer --n_layers=$layers --rand_seed=$rand
# classification by token
python $WORKING_DIR/classifier.py --n_layers=$layers --rand_seed=$rand