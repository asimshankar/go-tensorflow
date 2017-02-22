package main

import (
	"bufio"
	"container/heap"
	"fmt"
	"io"
	"sort"
)

func BuildDataset(raw io.ReadSeeker, vocabSize int) (*Dataset, error) {
	vocab, err := buildVocabulary(raw, vocabSize)
	if err != nil {
		return nil, err
	}
	if n, err := raw.Seek(0, 0); n != 0 || err != nil {
		return nil, fmt.Errorf("raw.Seek() returned (%v, %v), want (0, <nil>)", n, err)
	}
	data, err := applyVocabulary(raw, vocab)
	if err != nil {
		return nil, fmt.Errorf("failed to convert string dataset to list of integers: %v", err)
	}
	return &Dataset{
		Data:    data,
		Id2Word: reverseVocabulary(vocab),
	}, nil
}

func buildVocabulary(r io.Reader, size int) (map[string]int32, error) {
	var (
		s      = bufio.NewScanner(r)
		counts = make(map[string]int)
	)
	s.Split(bufio.ScanWords)
	for s.Scan() {
		counts[s.Text()]++
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if len(counts) <= size {
		size = len(counts)
	} else {
		// One slot in the dictionary is for all words that aren't in the dictionary
		size = size - 1
	}

	// Most common 'size' words
	h := &countHeap{
		words:  make([]string, 0, size),
		counts: counts,
	}
	heap.Init(h)
	outOfVocabularyCount := 0
	for w, _ := range counts {
		heap.Push(h, w)
		if h.Len() == size {
			p := heap.Pop(h).(string)
			outOfVocabularyCount += counts[p]
		}
	}

	// Sort the words in decreasing order of frequency
	sort.Slice(h.words, func(i, j int) bool {
		return counts[h.words[i]] > counts[h.words[j]]
	})
	vocab := make(map[string]int32)
	var id int32
	if outOfVocabularyCount > 0 {
		vocab[outOfVocabulary] = 0
		id = 1
	}
	for _, w := range h.words {
		vocab[w] = id
		id++
	}
	return vocab, nil
}

func applyVocabulary(r io.Reader, vocab map[string]int32) ([]int32, error) {
	s := bufio.NewScanner(r)
	s.Split(bufio.ScanWords)
	var ret []int32
	for s.Scan() {
		ret = append(ret, vocab[s.Text()])
	}
	return ret, s.Err()
}

func reverseVocabulary(vocab map[string]int32) []string {
	ret := make([]string, len(vocab))
	for w, id := range vocab {
		ret[id] = w
	}
	return ret
}

type countHeap struct {
	words  []string
	counts map[string]int
}

func (h countHeap) Len() int            { return len(h.words) }
func (h countHeap) Less(i, j int) bool  { return h.counts[h.words[i]] < h.counts[h.words[j]] }
func (h countHeap) Swap(i, j int)       { h.words[i], h.words[j] = h.words[j], h.words[i] }
func (h *countHeap) Push(x interface{}) { h.words = append(h.words, x.(string)) }
func (h *countHeap) Pop() interface{} {
	n := len(h.words)
	ret := h.words[n-1]
	h.words = h.words[0 : n-1]
	return ret
}
