package bm25

import (
    "errors"
    "sort"
    "strings"
)

// CountTermFreq counts the frequency of a term in a document using the provided tokenizer function.
func CountTermFreq(term string, doc string, tokenizer func(string) []string) (int, error) {
    if term == "" {
        return 0, errors.New("term cannot be empty")
    }

    if doc == "" {
        return 0, errors.New("document cannot be empty")
    }

    if tokenizer == nil {
        return 0, errors.New("tokenizer function cannot be nil")
    }

    tokens := tokenizer(doc)
    freq := 0
    for _, t := range tokens {
        if t == term {
            freq++
        }
    }
    return freq, nil
}

// TopNIndices returns the indices of the top N scores in the given slice.
func TopNIndices(scores []float64, n int) ([]int, error) {
    if n <= 0 {
        return nil, errors.New("n must be a positive integer")
    }

    indices := make([]int, len(scores))
    for i := range indices {
        indices[i] = i
    }

    sort.Slice(indices, func(i, j int) bool {
        return scores[indices[i]] > scores[indices[j]]
    })

    return indices[:Min(n, len(indices))], nil
}

// JoinTokens joins the tokens in a document into a single string using the provided separator.
func JoinTokens(tokens []string, separator string) string {
    return strings.Join(tokens, separator)
}

// Min returns the minimum of two integers.
func Min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
