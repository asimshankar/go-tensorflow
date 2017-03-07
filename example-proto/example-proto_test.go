//go:generate sh generate.sh

package main

import (
	pb "github.com/asimshankar/go-tensorflow/example-proto/proto/tensorflow/core/example"
	"github.com/golang/protobuf/proto"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"reflect"
	"testing"
)

func TestExampleProto(t *testing.T) {
	// Construct a tf.Example message
	xBytes, err := proto.Marshal(&pb.Example{
		&pb.Features{map[string]*pb.Feature{
			"age":            floatFeature(44),
			"education_num":  floatFeature(10),
			"capital_gain":   floatFeature(0),
			"capital_loss":   floatFeature(7688),
			"hours_per_week": floatFeature(40),
			"workclass":      bytesFeature([]byte("Private")),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Create a session for a graph that applies ParseExampleOp to the
	// input and feed xBytes to that.
	sess, input, age, hoursPerWeek, err := initSession()
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	xTensor, err := tf.NewTensor([]string{string(xBytes)})
	output, err := sess.Run(map[tf.Output]*tf.Tensor{input: xTensor}, []tf.Output{age, hoursPerWeek}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if got, want := output[0].Value(), []float32{44}; !reflect.DeepEqual(got, want) {
		t.Errorf("Got %v, want %v", got, want)
	}
	if got, want := output[1].Value(), []float32{40}; !reflect.DeepEqual(got, want) {
		t.Errorf("Got %v, want %v", got, want)
	}
}

func floatFeature(v ...float32) *pb.Feature {
	return &pb.Feature{&pb.Feature_FloatList{&pb.FloatList{v}}}
}

func bytesFeature(v ...[]byte) *pb.Feature {
	return &pb.Feature{&pb.Feature_BytesList{&pb.BytesList{v}}}
}

func initSession() (sess *tf.Session, in, age, hoursPerWeek tf.Output, err error) {
	var (
		s               = op.NewScope()
		input           = op.Placeholder(s, tf.String)
		zeroFloat       = op.Const(s.SubScope("zero_float"), float32(0))
		unknownShape    = tf.Shape{}
		_, _, _, output = op.ParseExample(s,
			input,
			op.Const(s.SubScope("empty_names"), []string{}),
			nil, /* sparse_keys */
			[]tf.Output{
				op.Const(s.SubScope("dense_key_1"), "age"),
				op.Const(s.SubScope("dense_key_2"), "hours_per_week"),
			},
			[]tf.Output{zeroFloat, zeroFloat}, /* dense_defaults */
			nil, /* sparse_types */
			[]tf.Shape{unknownShape, unknownShape})
	)
	graph, err := s.Finalize()
	if err != nil {
		return nil, in, age, hoursPerWeek, err
	}
	sess, err = tf.NewSession(graph, nil)
	return sess, input, output[0], output[1], err
}
