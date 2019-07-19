package main

import (
	"testing"

	"github.com/kitagry/go-nlp/utils"
)

func TestLDA(t *testing.T) {
	lda := NewLDA(10, 0.1, 0.1)

	docs := []string{
		"I like to eat broccoli and bananas.",
		"I ate a banana and spinach smoothie for breakfast.",
		"Chinchillas and kittens are cute.",
		"My sister adopted cute kittens yesterday.",
		"My brother will kick a kitten tomorrow.",
		"Look at this cute hamster munching on a piece of chinchillas.",
	}

	word2IdVectorizer := utils.NewWord2IdVectorizer()
	preprocessdDocs := make([][]string, len(docs))
	for i := 0; i < len(docs); i++ {
		preprocessdDocs[i] = utils.Preprocessing(docs[i])
	}
	vecs := word2IdVectorizer.Vectorize(preprocessdDocs)
	lda.SetDocs(vecs, word2IdVectorizer.WordNum())

	lda.Inference()
}
