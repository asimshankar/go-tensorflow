// Example of training the model created by model.py in Go.
//
// Usage: "go run train.go [--restore]"
// (See https://www.tensorflow.org/install/install_go for installation)
package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type Model struct {
	graph *tf.Graph
	sess  *tf.Session

	input  tf.Output
	target tf.Output
	output tf.Output

	initOp         *tf.Operation
	trainOp        *tf.Operation
	saveOp         *tf.Operation
	restoreOp      *tf.Operation
	checkpointFile tf.Output
}

func NewModel(graphDefFilename string) *Model {
	graphDef, err := ioutil.ReadFile(graphDefFilename)
	if err != nil {
		log.Fatal("Failed to read %q: %v", graphDefFilename, err)
	}
	graph := tf.NewGraph()
	if err = graph.Import(graphDef, ""); err != nil {
		log.Fatal("Invalid GraphDef?", err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}
	return &Model{
		graph: graph,
		sess:  sess,
		// All these names come from what model.py prints.
		input:          graph.Operation("input").Output(0),
		target:         graph.Operation("target").Output(0),
		output:         graph.Operation("output").Output(0),
		initOp:         graph.Operation("init"),
		trainOp:        graph.Operation("train"),
		saveOp:         graph.Operation("save/control_dependency"),
		restoreOp:      graph.Operation("save/restore_all"),
		checkpointFile: graph.Operation("save/Const").Output(0),
	}
}

func (m *Model) Init() {
	if _, err := m.sess.Run(nil, nil, []*tf.Operation{m.initOp}); err != nil {
		panic(err)
	}
}

func (m *Model) Restore(checkpointPrefix string) {
	t, err := tf.NewTensor(checkpointPrefix)
	if err != nil {
		panic(err)
	}
	feeds := map[tf.Output]*tf.Tensor{
		m.checkpointFile: t,
	}
	if _, err := m.sess.Run(feeds, nil, []*tf.Operation{m.restoreOp}); err != nil {
		panic(err)
	}
}

func (m *Model) Predict(batch [][][]float32) {
	batchTensor, err := tf.NewTensor(batch)
	if err != nil {
		panic(err)
	}
	feeds := map[tf.Output]*tf.Tensor{m.input: batchTensor}
	fetches := []tf.Output{m.output}
	results, err := m.sess.Run(feeds, fetches, nil)
	if err != nil {
		panic(err)
	}
	fetched := results[0].Value().([][][]float32)
	fmt.Println("Predictions:")
	for i := range batch {
		fmt.Printf("\tx = %v, predicted y = %v\n", batch[i], fetched[i])
	}
}

func (m *Model) RunTrainStep(inputBatch, targetBatch [][][]float32) {
	inputTensor, err := tf.NewTensor(inputBatch)
	if err != nil {
		panic(err)
	}
	targetTensor, err := tf.NewTensor(targetBatch)
	if err != nil {
		panic(err)
	}
	feeds := map[tf.Output]*tf.Tensor{
		m.input:  inputTensor,
		m.target: targetTensor,
	}
	if _, err = m.sess.Run(feeds, nil, []*tf.Operation{m.trainOp}); err != nil {
		panic(err)
	}
}

func (m *Model) Checkpoint(checkpointPrefix string) {
	t, err := tf.NewTensor(checkpointPrefix)
	if err != nil {
		panic(err)
	}
	feeds := map[tf.Output]*tf.Tensor{
		m.checkpointFile: t,
	}
	if _, err := m.sess.Run(feeds, nil, []*tf.Operation{m.saveOp}); err != nil {
		panic(err)
	}
}

func main() {
	var (
		graphDef            = "graph.pb"
		checkpointPrefix, _ = filepath.Abs(filepath.Join("checkpoints", "checkpoint"))
		restore             = directoryExists("checkpoints")
	)

	log.Print("Loading graph")
	model := NewModel(graphDef)

	if restore {
		log.Print("Restoring variables from checkpoint")
		model.Restore(checkpointPrefix)
	} else {
		log.Print("Initializing variables")
		model.Init()
	}

	testdata := [][][]float32{{{1}}, {{2}}, {{3}}}
	log.Print("Generating initial predictions")
	model.Predict(testdata)

	log.Print("Training for a few steps")
	for i := 0; i < 200; i++ {
		model.RunTrainStep(nextBatchForTraining())
	}

	log.Print("Updated predictions")
	model.Predict(testdata)

	log.Print("Saving checkpoint")
	model.Checkpoint(checkpointPrefix)
}

func directoryExists(dir string) bool {
	_, err := os.Stat(dir)
	return !os.IsNotExist(err)
}

func nextBatchForTraining() (inputs, targets [][][]float32) {
	const BATCH_SIZE = 10
	for i := 0; i < BATCH_SIZE; i++ {
		v := rand.Float32()
		inputs = append(inputs, [][]float32{{v}})
		targets = append(targets, [][]float32{{3*v + 2}})
	}
	return
}
