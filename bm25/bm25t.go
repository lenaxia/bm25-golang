package bm25

import (
    "errors"
    "log"
)

// BM25T is an implementation of the BM25T variant.
type BM25T struct {
    *bm25Base
    k1    float64
    b     float64
    delta float64
}

// NewBM25T creates a new instance of the BM25T struct.
func NewBM25T(corpus []string, tokenizer func(string) []string, k1 float64, b float64, delta float64, logger *log.Logger) (*BM25T, error) {
    if k1 < 0 {
        return nil, errors.New("k1 must be non-negative")
    }

    if b < 0 || b > 1 {
        return nil, errors.New("b must be between 0 and 1")
    }

    if delta < 0 {
        return nil, errors.New("delta must be non-negative")
    }

    base, err := NewBM25Base(corpus, tokenizer, logger)
    if err != nil {
        return nil, err
    }

    return &BM25T{
        bm25Base: base,
        k1:       k1,
        b:        b,
        delta:    delta,
    }, nil
}

// GetScores returns the BM25 scores for the given query.
func (t *BM25T) GetScores(query []string) ([]float64, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    scores := make([]float64, t.corpusSize)
    for _, q := range query {
        qFreq := make([]float64, t.corpusSize)
        for i, doc := range t.corpus {
            qFreq[i] = float64(countTermFreq(q, doc))
        }

        idf, err := t.IDF(q)
        if err != nil {
            if t.logger != nil {
                t.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
            }
            continue
        }

        for i, docLen := range t.docLengths {
            k := t.k1 * (1 - t.b + t.b*float64(docLen)/t.avgDocLen)
            scores[i] += idf * (t.delta + (qFreq[i] * (1 + k)) / (qFreq[i] + k))
        }
    }

    return scores, nil
}

// GetBatchScores returns the BM25 scores for the given query and a subset of documents.
func (t *BM25T) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    if len(docIDs) == 0 {
        return nil, errors.New("document IDs cannot be empty")
    }

    scores := make([]float64, len(docIDs))
    for _, q := range query {
        qFreq := make([]float64, len(docIDs))
        for i, docID := range docIDs {
            if docID < 0 || docID >= t.corpusSize {
                if t.logger != nil {
                    t.logger.Printf("Invalid document ID: %d", docID)
                }
                continue
            }
            qFreq[i] = float64(countTermFreq(q, t.corpus[docID]))
        }

        idf, err := t.IDF(q)
        if err != nil {
            if t.logger != nil {
                t.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
            }
            continue
        }

        for i, docID := range docIDs {
            if docID < 0 || docID >= t.corpusSize {
                continue
            }
            docLen := t.docLengths[docID]
            k := t.k1 * (1 - t.b + t.b*float64(docLen)/t.avgDocLen)
            scores[i] += idf * (t.delta + (qFreq[i] * (1 + k)) / (qFreq[i] + k))
        }
    }

    return scores, nil
}

// GetTopN returns the top N documents for the given query.
func (t *BM25T) GetTopN(query []string, n int) ([]string, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    if n <= 0 {
        if t.logger != nil {
            t.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
        }
        return []string{}, nil
    }

    scores, err := t.GetScores(query)
    if err != nil {
        return nil, err
    }

    topNIndices := topNIndices(scores, n)

    topDocs := make([]string, len(topNIndices))
    for i, idx := range topNIndices {
        topDocs[i] = joinTokens(t.corpus[idx])
    }

    return topDocs, nil
}
