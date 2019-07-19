package utils

import (
	"math"
)

type Vectorizer interface {
	Vectorize(docs [][]string) [][]float64
}

type UniqVectorizer interface {
	Vectorizer
	WordNum() int
}

var _ Vectorizer = (*bowVectorizer)(nil)
var _ Vectorizer = (*tfidfVectorizer)(nil)
var _ UniqVectorizer = (*word2IdVectorizer)(nil)

func makeWord2Id(docs [][]string) map[string]int {
	word2id := make(map[string]int)
	count := 0
	for _, doc := range docs {
		for _, word := range doc {
			if _, ok := word2id[word]; !ok {
				word2id[word] = count
				count++
			}
		}
	}
	return word2id
}

type bowVectorizer struct{}

func NewBowVectorizer() Vectorizer {
	return bowVectorizer{}
}

func (bv bowVectorizer) Vectorize(docs [][]string) [][]float64 {
	word2id := makeWord2Id(docs)

	results := make([][]float64, len(docs))
	for index, doc := range docs {
		result := make([]float64, len(word2id))
		for _, word := range doc {
			id, ok := word2id[word]
			if !ok {
				panic("なんかおかしい")
			}
			result[id] += 1
		}
		results[index] = result
	}
	return results
}

type tfidfVectorizer struct {
	minIdf float64
}

func NewTfidfVectorizer(minIdf float64) Vectorizer {
	return tfidfVectorizer{
		minIdf: minIdf,
	}
}

func countWords(doc []string) map[string]int {
	maps := make(map[string]int)
	for _, word := range doc {
		if _, ok := maps[word]; ok {
			maps[word] += 1
		} else {
			maps[word] = 1
		}
	}
	return maps
}

func in(word string, doc []string) bool {
	for _, t := range doc {
		if t == word {
			return true
		}
	}
	return false
}

func count(word string, docs [][]string) (count int) {
	for _, doc := range docs {
		if in(word, doc) {
			count++
		}
	}
	return
}

func (tv tfidfVectorizer) Vectorize(docs [][]string) [][]float64 {
	tf := func(word2id map[string]int, doc []string) []float64 {
		termCounts := make([]float64, len(word2id))
		for key, count := range countWords(doc) {
			id, ok := word2id[key]
			if !ok {
				panic("おかしいよ")
			}
			termCounts[id] = float64(count) / float64(len(doc))
		}
		return termCounts
	}

	idf := func(word2id map[string]int, docs [][]string) []float64 {
		idf := make([]float64, len(word2id))
		for word, id := range word2id {
			idf[id] = math.Log(float64(len(docs)) / float64(count(word, docs)))
		}
		return idf
	}

	word2id := makeWord2Id(docs)

	tfidfVec := make([][]float64, len(docs))
	idfVec := idf(word2id, docs)

	filteredIdf := make([]float64, 0)
	for i := 0; i < len(idfVec); i++ {
		if idfVec[i] > tv.minIdf {
			filteredIdf = append(filteredIdf, idfVec[i])
		}
	}

	for i := 0; i < len(docs); i++ {
		tfs := tf(word2id, docs[i])
		res := make([]float64, len(filteredIdf))
		current := 0
		for j := 0; j < len(tfs); j++ {
			if idfVec[j] > tv.minIdf {
				res[current] = tfs[j] * idfVec[j]
				current++
			}
		}
		tfidfVec[i] = res
	}
	return tfidfVec
}

type word2IdVectorizer struct {
	word2id map[string]int
}

func NewWord2IdVectorizer() UniqVectorizer {
	return &word2IdVectorizer{}
}

func (w *word2IdVectorizer) Vectorize(docs [][]string) [][]float64 {
	word2id := makeWord2Id(docs)
	w.word2id = word2id
	result := make([][]float64, len(docs))
	for i, ds := range docs {
		r := make([]float64, len(ds))
		for j, d := range ds {
			r[j] = float64(word2id[d])
		}
		result[i] = r
	}
	return result
}

func (w *word2IdVectorizer) WordNum() int {
	return len(w.word2id)
}
