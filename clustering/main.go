package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/kitagry/go-nlp/utils"
)

func kMeans(vecs [][]float64, knum, nSample int) (result []int) {
	result = make([]int, len(vecs))

	// ランダムにグループ分け
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < len(result); i++ {
		result[i] = rand.Intn(knum)
	}

	writer := bufio.NewWriter(os.Stdout)
	for sample := 0; sample < nSample; sample++ {
		writer.WriteString(fmt.Sprintf("\r%d/%d", sample+1, nSample))
		writer.Flush()
		// 重心を求める
		centers := make([][]float64, knum)
		for i := 0; i < knum; i++ {
			centers[i] = make([]float64, len(vecs[0]))
		}

		for index, num := range result {
			for i := 0; i < len(vecs[0]); i++ {
				centers[num][i] += vecs[index][i]
			}
		}

		for i := 0; i < len(vecs); i++ {
			vec := vecs[i]
			minIndex := 0
			minDistance := -1.
			for i, center := range centers {
				dis, err := utils.CosineDistance(vec, center)
				if err != nil {
					fmt.Println(err)
					return
				}

				if minDistance < dis {
					minIndex = i
					minDistance = dis
				}
			}
			result[i] = minIndex
		}
	}
	writer.WriteString("\n")
	writer.Flush()

	return
}

func main() {
	f, err := os.Open("./movie.txt")
	if err != nil {
		panic(err)
	}

	sc := bufio.NewScanner(f)
	titles := make([]string, 0)
	docs := make([]string, 0)
	for sc.Scan() {
		t := sc.Text()

		texts := strings.SplitN(t, ",", 2)
		titles = append(titles, texts[0])
		docs = append(docs, texts[1])
	}

	fmt.Println("Preprocess documents.")
	preprocessedDocs := make([][]string, len(docs))
	for i := 0; i < len(docs); i++ {
		preprocessedDocs[i] = utils.Preprocessing(docs[i])
	}

	tfidfVectorizer := utils.NewTfidfVectorizer(0.1)
	vecs := tfidfVectorizer.Vectorize(preprocessedDocs)

	fmt.Println("KMeans")
	groupNum := 2
	groupIds := kMeans(vecs, groupNum, 100)
	groups := make(map[int][]string)

	for index, group := range groupIds {
		if _, ok := groups[group]; !ok {
			groups[group] = make([]string, 0)
		}

		groups[group] = append(groups[group], titles[index])
	}

	for _, group := range groups {
		fmt.Println(group)
	}
}
