TensorFlow's Python API provides a
[`Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) class for
managing model parameters. Similar abstractions are not part of the Go API,
but since the Python `Variable` class essentially wraps over a set of
primitive operations, TensorFlow variables can be used in Go.

See `variable_test.go` for a demonstration of implementing variables in Go.

## Quickstart

1. Install the [TensorFlow Go package](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/go/README.md)

2. `go get` and run:

   ```sh
   go get -t github.com/asimshankar/go-tensorflow/variable
   go test github.com/asimshankar/go-tensorflow/variable
   ```

## Caveats

As of March 2017, the internal implementation of variables in the TensorFlow
Python API was on a path of change. In particular, using "resource types"
instead of "reference types". The details are beyond the scope of this README,
but suffice to say that the implementation here is with resource types. See
[`use_resource`](https://github.com/tensorflow/tensorflow/blob/83cd3fd279037c242017cd0ab8c825f30c375564/tensorflow/python/ops/variable_scope.py#L254)
in the Python API.
