# BM25 Golang Implementation

This repository provides a comprehensive implementation of various BM25 variants in the Go programming language. BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query. This implementation is inspired by and builds upon the work done by [Dorian Brown](https://github.com/dorianbrown/rank_bm25) in their Python implementation of BM25 algorithms.

## Table of Contents

- [BM25 Variants](#bm25-variants)
- [Installation](#installation)
- [Usage](#usage)
  - [Initializing](#initializing)
  - [Ranking Documents](#ranking-documents)
  - [Parallel and Batched Computation](#parallel-and-batched-computation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## BM25 Variants

This repository includes implementations of the following BM25 variants:

- Okapi BM25
- BM25L
- BM25+
- BM25-Adpt
- BM25T

These variants are based on the research paper ["A Study of Efficient and Robust IR Metrics"](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.8440&rep=rep1&type=pdf) by Luca Pinto, Diego Ceccarelli, and Claudio Lucchese, which provides an overview and benchmarks of each method.

## Installation

To use this BM25 implementation, you need to have Go installed on your system. You can download and install Go from the official website: [https://golang.org/dl/](https://golang.org/dl/)

Once you have Go installed, you can clone this repository and build the package:

```bash
git clone https://github.com/lenaxia/bm25-golang.git
cd bm25-golang
go build
```

## Usage

### Initializing

To initialize a BM25 instance, you need to provide a corpus of text documents and a tokenizer function. The tokenizer function is responsible for splitting a document into individual tokens (e.g., words).

Here's an example of how to initialize a BM25 instance using the `BM25Okapi` variant:

```go
import (
    "strings"
    "bm25-golang/bm25"
)

corpus := []string{
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?",
}

tokenizer := func(s string) []string {
    return strings.Split(s, " ")
}

bm25, err := bm25.NewBM25Okapi(corpus, tokenizer, nil)
if err != nil {
    // Handle error
}
```

In this example, we define a corpus of three text documents and a simple tokenizer function that splits the text on whitespace characters. We then create a new instance of `BM25Okapi` using the `NewBM25Okapi` function, passing in the corpus, tokenizer, and a logger (which can be `nil` if you don't need logging).

### Ranking Documents

Once you have initialized a BM25 instance, you can use it to rank documents based on their relevance to a given query. Here's an example:

```go
query := "windy London"
tokenizedQuery := tokenizer(query)

scores, err := bm25.GetScores(tokenizedQuery)
if err != nil {
    // Handle error
}

// Scores is now a slice of float64 values representing the relevance scores
// for each document in the corpus.
```

In this example, we define a query string `"windy London"` and tokenize it using the same tokenizer function we used for the corpus. We then call the `GetScores` method on the `BM25Okapi` instance, passing in the tokenized query. The `GetScores` method returns a slice of `float64` values representing the relevance scores for each document in the corpus.

Alternatively, you can use the `GetTopN` method to retrieve the top `N` most relevant documents:

```go
topN := 1
topDocs, err := bm25.GetTopN(tokenizedQuery, topN)
if err != nil {
    // Handle error
}

// topDocs is now a slice of strings containing the top N most relevant documents.
```

In this example, we call the `GetTopN` method on the `BM25Okapi` instance, passing in the tokenized query and the value `1` for `topN`. The `GetTopN` method returns a slice of strings containing the top `N` most relevant documents.

### Parallel and Batched Computation

This implementation also provides parallel and batched computation methods for improved performance when dealing with large corpora or many queries. These methods include:

- `GetScoresParallel`: Computes the BM25 scores for a given query using parallel computation.
- `GetBatchScoresParallel`: Computes the BM25 scores for a given query and a subset of documents using parallel computation.
- `GetTopNParallel`: Returns the top `N` documents for a given query using parallel computation.
- `GetScoresBatched`: Computes the BM25 scores for a given query using parallel computation with batching.
- `GetBatchScoresBatched`: Computes the BM25 scores for a given query and a subset of documents using parallel computation with batching.
- `GetTopNBatched`: Returns the top `N` documents for a given query using parallel computation with batching.

These methods follow a similar usage pattern as their non-parallel and non-batched counterparts, but they provide improved performance by leveraging Go's concurrency features and batching techniques.

## Examples

For more detailed examples and usage scenarios, please refer to the `examples/` directory in this repository.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. Make sure to follow the established coding conventions and provide appropriate tests for any new features or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This BM25 implementation in Go is inspired by and builds upon the work done by [Dorian Brown](https://github.com/dorianbrown/rank_bm25) in their Python implementation of BM25 algorithms. We would like to express our gratitude for their valuable contribution to the field of information retrieval and for providing a solid foundation for this Go implementation.
