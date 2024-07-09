package bm25

import (
    "errors"
    "log"
)

// BM25L is an implementation of the BM25L variant.
type BM25L struct {
    *bm25Base
    k1 float64
    b  float64
}

// NewBM25L creates a new instance of the BM25L struct.
func NewBM25L(corpus []string, tokenizer func(string) []string, k1 float64, b float64, logger *log.Logger) (*BM25L, error) {
    if k1 < 0 {
        return nil, errors.New("k1 must be non-negative")
    }

    if b < 0 || b > 1 {
        return nil, errors.New("b must be between 0 and 1")
    }

    base, err := NewBM25Base(corpus, tokenizer, logger)
    if err != nil {
        return nil, err
    }

    return &BM25L{
        bm25Base: base,
        k1:       k1,
        b:        b,
    }, nil
}

// GetScores returns the BM25 scores for the given query.
func (l *BM25L) GetScores(query []string) ([]float64, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    scores := make([]float64, l.corpusSize)
    for _, q := range query {
        qFreq := make([]float64, l.corpusSize)
        for i, doc := range l.corpus {
            docStr := JoinTokens(doc, " ")
            freq, _ := CountTermFreq(q, docStr, l.tokenizer) // Ignore the error for now
            qFreq[i] = float64(freq)
        }

        idf, err := l.IDF(q)
        if err != nil {
            if l.logger != nil {
                l.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
            }
            continue
        }

        for i, docLen := range l.docLengths {
            k := l.k1 * (1 - l.b + l.b*float64(docLen)/l.avgDocLen)
            scores[i] += idf * (qFreq[i] / (qFreq[i] + k))
        }
    }

    return scores, nil
}

// GetBatchScores returns the BM25 scores for the given query and a subset of documents.
func (l *BM25L) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
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
            if docID < 0 || docID >= l.corpusSize {
                if l.logger != nil {
                    l.logger.Printf("Invalid document ID: %d", docID)
                }
                continue
            }
            docStr := JoinTokens(l.corpus[docID], " ")
            freq, _ := CountTermFreq(q, docStr, l.tokenizer) // Ignore the error for now
            qFreq[i] = float64(freq)
        }

        idf, err := l.IDF(q)
        if err != nil {
            if l.logger != nil {
                l.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
            }
            continue
        }

        for i, docID := range docIDs {
            if docID < 0 || docID >= l.corpusSize {
                continue
            }
            docLen := l.docLengths[docID]
            k := l.k1 * (1 - l.b + l.b*float64(docLen)/l.avgDocLen)
            scores[i] += idf * (qFreq[i] / (qFreq[i] + k))
        }
    }

    return scores, nil
}

// GetTopN returns the top N documents for the given query.
func (l *BM25L) GetTopN(query []string, n int) ([]string, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    if n <= 0 {
        if l.logger != nil {
            l.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
        }
        return []string{}, nil
    }

    scores, err := l.GetScores(query)
    if err != nil {
        return nil, err
    }

    topNIndices, err := TopNIndices(scores, n)
    if err != nil {
        return nil, err
    }

    topDocs := make([]string, len(topNIndices))
    for i, idx := range topNIndices {
        topDocs[i] = JoinTokens(l.corpus[idx], " ")
    }

    return topDocs, nil
}
