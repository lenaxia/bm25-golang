package bm25

import "sync"

// GetScoresBatched returns the BM25 scores for the given query using parallel computation with batching.
func (b *bm25Base) GetScoresBatched(query []string, bm25 BM25, batchSize int) ([]float64, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    if batchSize <= 0 {
        return nil, errors.New("batch size must be a positive integer")
    }

    var wg sync.WaitGroup
    scores := make([]float64, b.corpusSize)
    numBatches := (b.corpusSize + batchSize - 1) / batchSize
    wg.Add(numBatches)

    for i := 0; i < numBatches; i++ {
        start := i * batchSize
        end := min(start+batchSize, b.corpusSize)
        go func(start, end int) {
            defer wg.Done()
            for _, q := range query {
                qFreq := make([]float64, end-start)
                for j := start; j < end; j++ {
                    qFreq[j-start] = float64(countTermFreq(q, b.corpus[j]))
                }

                idf, err := b.IDF(q)
                if err != nil {
                    if b.logger != nil {
                        b.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
                    }
                    continue
                }

                for j := start; j < end; j++ {
                    docLen := b.docLengths[j]
                    k := computeK(bm25, docLen)
                    scores[j] += idf * computeScore(bm25, qFreq[j-start], k)
                }
            }
        }(start, end)
    }

    wg.Wait()
    return scores, nil
}

// GetBatchScoresBatched returns the BM25 scores for the given query and a subset of documents using parallel computation with batching.
func (b *bm25Base) GetBatchScoresBatched(query []string, docIDs []int, bm25 BM25, batchSize int) ([]float64, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    if len(docIDs) == 0 {
        return nil, errors.New("document IDs cannot be empty")
    }

    if batchSize <= 0 {
        return nil, errors.New("batch size must be a positive integer")
    }

    var wg sync.WaitGroup
    scores := make([]float64, len(docIDs))
    numBatches := (len(docIDs) + batchSize - 1) / batchSize
    wg.Add(numBatches)

    for i := 0; i < numBatches; i++ {
        start := i * batchSize
        end := min(start+batchSize, len(docIDs))
        go func(start, end int) {
            defer wg.Done()
            for _, q := range query {
                qFreq := make([]float64, end-start)
                for j := start; j < end; j++ {
                    docID := docIDs[j]
                    if docID < 0 || docID >= b.corpusSize {
                        if b.logger != nil {
                            b.logger.Printf("Invalid document ID: %d", docID)
                        }
                        continue
                    }
                    qFreq[j-start] = float64(countTermFreq(q, b.corpus[docID]))
                }

                idf, err := b.IDF(q)
                if err != nil {
                    if b.logger != nil {
                        b.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
                    }
                    continue
                }

                for j := start; j < end; j++ {
                    docID := docIDs[j]
                    if docID < 0 || docID >= b.corpusSize {
                        continue
                    }
                    docLen := b.docLengths[docID]
                    k := computeK(bm25, docLen)
                    scores[j-start] += idf * computeScore(bm25, qFreq[j-start], k)
                }
            }
        }(start, end)
    }

    wg.Wait()
    return scores, nil
}

// GetTopNBatched returns the top N documents for the given query using parallel computation with batching.
func (b *bm25Base) GetTopNBatched(query []string, n int, bm25 BM25, batchSize int) ([]string, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    if n <= 0 {
        if b.logger != nil {
            b.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
        }
        return []string{}, nil
    }

    if batchSize <= 0 {
        return nil, errors.New("batch size must be a positive integer")
    }

    scores, err := b.GetScoresBatched(query, bm25, batchSize)
    if err != nil {
        return nil, err
    }

    topNIndices := topNIndices(scores, n)

    topDocs := make([]string, len(topNIndices))
    for i, idx := range topNIndices {
        topDocs[i] = joinTokens(b.corpus[idx])
    }

    return topDocs, nil
}
