# `Example` protos in Go

Some TensorFlow programs use the
[`Example`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto)
protocol buffer and the `ParseExample` operation. The [Reading Data](https://www.tensorflow.org/programmers_guide/reading_data#file_formats)
programmer's guide talks about this.

This trivial program demonstrates usage of this protocol buffer in TensorFlow in Go.

## Quickstart

1. Install the [TensorFlow Go package](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/go/README.md)

2. `go get` and run:

   ```sh
   go get -t github.com/asimshankar/go-tensorflow/example-proto
   go test github.com/asimshankar/go-tensorflow/example-proto
   ```
