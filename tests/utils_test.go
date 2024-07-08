package bm25

import (
	"testing"
)

func TestCountTermFreq(t *testing.T) {
	doc := []string{"hello", "world", "hello"}
	freq := countTermFreq("hello", doc)
	if freq != 2 {
		t.Errorf("countTermFreq() returned incorrect frequency: %d", freq)
	}
}

func TestTopNIndices(t *testing.T) {
	scores := []float64{0.5, 0.8, 0.2, 0.9}
	indices := topNIndices(scores, 2)
	expected := []int{3, 1}
	if !equalIntSlices(indices, expected) {
		t.Errorf("topNIndices() returned incorrect indices: %v", indices)
	}
}

func TestJoinTokens(t *testing.T) {
	tokens := []string{"hello", "world"}
	joined := joinTokens(tokens)
	if joined != "hello world" {
		t.Errorf("joinTokens() returned incorrect string: %s", joined)
	}
}

func TestMin(t *testing.T) {
	a, b := 3, 5
	min := min(a, b)
	if min != 3 {
		t.Errorf("min() returned incorrect value: %d", min)
	}
}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
