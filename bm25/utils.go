package bm25

import "sort"

// countTermFreq counts the frequency of a term in a document.
func countTermFreq(term string, doc []string) int {
	freq := 0
	for _, t := range doc {
		if t == term {
			freq++
		}
	}
	return freq
}

// topNIndices returns the indices of the top N scores in the given slice.
func topNIndices(scores []float64, n int) []int {
	indices := make([]int, len(scores))
	for i := range indices {
		indices[i] = i
	}

	sort.Slice(indices, func(i, j int) bool {
		return scores[indices[i]] > scores[indices[j]]
	})

	return indices[:min(n, len(indices))]
}

// joinTokens joins the tokens in a document into a single string.
func joinTokens(tokens []string) string {
	return strings.Join(tokens, " ")
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
