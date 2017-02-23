# word2vec in Go

Sample use of the [TensorFlow](https://www.tensorflow.org)
[Go package](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)
to train a word embedding using the [word2vec](https://www.tensorflow.org/tutorials/word2vec)
model.

## Quickstart

1. Install the [TensorFlow Go package](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/go/README.md)

2. Download and unzip the training dataset. Something like:

   ```sh
   curl -L "http://mattmahoney.net/dc/text8.zip" -o text8.zip
   unzip text8.zip
   rm text8.zip
   ```

3. `go get` and run:

   ```sh
   go get github.com/asimshankar/go-tensorflow/word2vec
   word2vec -dataset ./text8
   ```


## Notes

- The model (`word2vec_train_graph.pb`) is created by a Python program instead
  of from within Go to avail of the higher level model construciton primitives
  like [`tf.nn.embedding_lookup`]. To recreate the model, [install TensorFlow python packages](https://www.tensorflow.org/install/)
  (typically, `pip install tensorflow`) and execute `python ./create_graph.py`.

- In the current form, the trained model isn't saved to disk. Yes, that should be done.

- `main.go` in this directory is roughly equivalent to
  [`word2vec_basic.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py).
  `word2vec_basic.py`, while instructive, has poor performance because it is
  bottlenecked on feeding training data from Python. This Go example is faster,
  but there is still much headroom to ensure that input feeding isn't the
  bottleneck for training. See the [TensorFlow
  article](https://www.tensorflow.org/tutorials/word2vec#optimizing_the_implementation)
  on optimizing the implementaiton. 

## Speed

See notes above, but for comparison I ran `word2vec_basic.py` and this Go
program on the same machine and one one random day in February 2017 on the
same machine, Python was taking ~6 seconds per 2000 steps while the code
here was taking 3 seconds.
