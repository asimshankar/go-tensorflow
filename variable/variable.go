// Package variable is a test implementation of TensorFlow variables in go.
//
// The implementation takes its cue from the upcoming "new" way of using
// variables in
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/resource_variable_ops.py
// but lacks many many missing details such as colocation, checking whether the
// variable is initialized, caching on another device etc. (see
// ResourceVariable._init_from_args in Python).
//
// Something along these lines probably should makes its way into the 'op'
// package, but for now we prototype here.
package variable

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Variable create a variable.
//
// Returns the operation that initializes the variable to initialValue,
// a handle to the variable to use for assignment operations, and
// an Output that produces the current value of the variable.
func Variable(scope *op.Scope, initialValue tf.Output) (init *tf.Operation, handle, value tf.Output) {
	// TODO: See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/resource_variable_ops.py
	// and the ResourceVariable class there for guidance on what this should really look like.
	scope = scope.SubScope("Variable")
	dtype := initialValue.DataType()
	handle = op.VarHandleOp(scope, dtype, initialValue.Shape())
	init = op.AssignVariableOp(scope.SubScope("Assign"), handle, initialValue)
	value = op.ReadVariableOp(scope.SubScope("Read"), handle, dtype)
	return init, handle, value
}
