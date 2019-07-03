package main

import (
	"regexp"
	"strings"

	"github.com/aaaton/golem"
	"github.com/mdanzinger/stopwords"
)

func cleaningText(text string) string {
	re := regexp.MustCompile("(@|<.+?>|\\(.*\\))")
	return re.ReplaceAllString(text, "")
}

func removeStopWords(text string) string {
	return stopwords.CleanString(text, "en", false)
}

func tokenizeText(text string) []string {
	return strings.Split(text, " ")
}

var lemmatizer *golem.Lemmatizer

func lemmatizeWord(word string) string {
	return lemmatizer.Lemma(word)
}

// Prerocessing for text.
// 1. Remove special charactor.
// 2. Remove stop words.
// 3. Split texts.
// 4. Lemmatize words.
func Preprocessing(text string) []string {
	text = cleaningText(text)
	text = removeStopWords(text)
	if text[0] == ' ' {
		text = text[1:]
	}
	if text[len(text)-1] == ' ' {
		text = text[0 : len(text)-1]
	}
	texts := tokenizeText(text)
	for i := 0; i < len(texts); i++ {
		texts[i] = lemmatizeWord(texts[i])
	}
	return texts
}
