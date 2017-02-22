// Command word2vec implements https://www.tensorflow.org/tutorials/word2vec in Go.
//
// See: https://github.com/asimshankar/go-tensorflow/tree/master/word2vec/README.md
package main

import (
	"flag"
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

const (
	outOfVocabulary = "UNK"

	// Size of "context" around a target word.
	skipWindow = 1
	// Number of (example, labels) pairs to generate from a window.
	numSkips = 2
	// Total number of words (before, target_word, after) in a window.
	span = 2*skipWindow + 1
)

// Dataset is the data used for training.
type Dataset struct {
	// List of words in the text, with words represented as integer ids.
	Data []int32

	// Id2Word: Maps an integer id to the corresponding string
	Id2Word []string
}

func main() {
	var (
		flagSteps     = flag.Int("training_steps", 100000, "Number of steps to train for")
		flagBatchSize = flag.Int("training_batch_size", 128, "Batch size for a single training step")
		flagSeed      = flag.Int64("random_seed", time.Now().UTC().UnixNano(), "Random seed to use")
		flagGraphDef  = flag.String("graph", "word2vec_train_graph.pb", "Path to the GraphDef file written out by create_graph.py")
		flagDataset   = flag.String("dataset", "", "Path to the training dataset text file")
	)
	flag.Parse()
	rand.Seed(*flagSeed)
	log.Printf("Random seed set to: %v", *flagSeed)

	// Load the training graph and create a trainer
	graph := tf.NewGraph()
	graphDef, err := ioutil.ReadFile(*flagGraphDef)
	if err != nil {
		log.Fatal(err)
	}
	if err := graph.Import(graphDef, ""); err != nil {
		log.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer sess.Close()
	trainer, err := newTrainer(graph, sess)
	if err != nil {
		log.Fatal(err)
	}
	log.Println("Initialized trainer")

	// Print details about the embedding based on the shape of the
	// normalized_embeddings matrix in the graph.
	embeddingShape := graph.Operation("normalized_embeddings").Output(0).Shape()
	vocabularySize := int(embeddingShape.Size(0))
	log.Printf("Loaded model. Vocabulary size: %d, embedding space has %d dimensions", vocabularySize, embeddingShape.Size(1))

	// Build the dataset
	dataset, err := buildDataset(*flagDataset, vocabularySize)
	if err != nil {
		log.Fatal(err)
	}
	// Train
	batcher := newBatcher(dataset)
	var lossSum float32
	for step := 0; step < *flagSteps; step++ {
		loss, err := trainer.step(batcher.Next(*flagBatchSize))
		if err != nil {
			log.Fatalf("Training failed at step %d: %v", step, err)
		}
		lossSum += loss
		if step%2000 == 0 && step > 0 {
			log.Printf("Average loss at step %d: %v", step, lossSum/2000)
			lossSum = 0
		}
		if step%10000 == 0 && step > 0 {
			fmt.Print("After ", step, " steps: ")
			if err := sampleNeighbors(dataset.Id2Word, sess, graph); err != nil {
				log.Fatalf("Unable to calculate similarity for validation set after %d steps: %v", step, err)
			}
		}
	}

}

type trainer struct {
	sess                   *tf.Session
	examples, labels, loss tf.Output
	train                  *tf.Operation
}

func newTrainer(graph *tf.Graph, sess *tf.Session) (*trainer, error) {
	t := &trainer{
		sess:     sess,
		examples: graph.Operation("train_examples").Output(0),
		labels:   graph.Operation("train_labels").Output(0),
		train:    graph.Operation("train_op"),
		loss:     graph.Operation("loss").Output(0),
	}
	// Initialize the variables
	_, err := sess.Run(nil, nil, []*tf.Operation{graph.Operation("init")})
	return t, err
}

func (t *trainer) step(examples []int32, labels [][1]int32) (float32, error) {
	et, err := tf.NewTensor(examples)
	if err != nil {
		return 0, fmt.Errorf("invalid examples: %v", err)
	}
	lt, err := tf.NewTensor(labels)
	if err != nil {
		return 0, fmt.Errorf("invalid labels: %v", err)
	}
	fetched, err := t.sess.Run(
		map[tf.Output]*tf.Tensor{
			t.examples: et,
			t.labels:   lt,
		},
		[]tf.Output{t.loss},
		[]*tf.Operation{t.train},
	)
	if err != nil {
		return 0, err
	}
	return fetched[0].Value().(float32), nil
}

// sampleNeighbors prints the 8 closest neighbors of a random 10 of the 20 most
// frequent words. Words are assigned integer ids in frequency order, so the
// lowest 20 ids correspond to the most frequent words.
func sampleNeighbors(vocab []string, sess *tf.Session, graph *tf.Graph) error {
	words := make([]int32, 10)
	for i, w := range rand.Perm(20)[:len(words)] {
		words[i] = int32(w)
	}
	wordst, err := tf.NewTensor(words)
	if err != nil {
		return err
	}
	fetched, err := sess.Run(
		map[tf.Output]*tf.Tensor{graph.Operation("input").Output(0): wordst},
		[]tf.Output{graph.Operation("similarity").Output(0)},
		nil,
	)
	if err != nil {
		return err
	}
	similarity := fetched[0].Value().([][]float32)
	fmt.Println("Closest neighbors:")
	type neighbor struct {
		word  int
		score float32
	}
	var neighbors []neighbor
	for i, scores := range similarity {
		fmt.Printf("%10s --> ", vocab[words[i]])
		neighbors = neighbors[0:0]
		for w, s := range scores {
			neighbors = append(neighbors, neighbor{w, s})
		}
		sort.Slice(neighbors, func(i, j int) bool {
			return neighbors[i].score > neighbors[j].score
		})
		for _, n := range neighbors[1:9] {
			fmt.Printf("%q (%0.2f), ", vocab[n.word], n.score)
		}
		fmt.Println("...")
	}
	return nil
}

func buildDataset(filename string, vocabularySize int) (*Dataset, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dataset, err := BuildDataset(f, vocabularySize)
	if err != nil {
		return nil, err
	}
	log.Printf("Loaded dataset from %q", filename)
	log.Printf("Top 5 terms in vocabulary: %v", dataset.Id2Word[:5])
	sample := dataset.Data[:10]
	strs := make([]string, len(sample))
	for i, j := range sample {
		strs[i] = dataset.Id2Word[j]
	}
	log.Printf("Sample data: %v --> %s", sample, strings.Join(strs, " "))
	return dataset, nil
}

type batcher struct {
	dataset *Dataset
	idx     int
}

func newBatcher(dataset *Dataset) *batcher {
	return &batcher{dataset, 0}
}

func (b *batcher) Next(batchSize int) (examples []int32, labels [][1]int32) {
	examples = make([]int32, 0, batchSize)
	labels = make([][1]int32, 0, batchSize)
	// exclude keeps track of labels to avoid for a particular example.
	// labels should be avoided if they have already been selected.
	exclude := make([]int32, 0, numSkips)
	for len(examples) < batchSize {
		window := b.window()
		b.idx++
		const center = skipWindow
		exclude = exclude[0:0]
		for i := 0; i < numSkips; i++ {
			var l int32
			retry := true
			for retry {
				l = int32(rand.Intn(span))
				retry = l == center
				for _, x := range exclude {
					if l == x {
						retry = true
						break
					}
				}
			}
			exclude = append(exclude, l)
			examples = append(examples, window[center])
			labels = append(labels, [1]int32{window[l]})
		}
	}
	return
}

func (b *batcher) window() []int32 {
	if b.idx+span < len(b.dataset.Data) {
		return b.dataset.Data[b.idx : b.idx+span]
	}
	// edge of the dataset, copy the window out
	window := make([]int32, span)
	for i := 0; i < span; i++ {
		window[i] = b.dataset.Data[(b.idx+i)%len(b.dataset.Data)]
	}
	return window
}
