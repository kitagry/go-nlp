package main

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type LDA struct {
	TopicNum int
	Alpha    float64
	Beta     float64

	nMZ *mat.Dense
	nZT *mat.Dense
	nZ  []float64
	zMN [][]int

	docIds    [][]float64
	vocabSize int
}

func NewLDA(topicNum int, alpha, beta float64) LDA {
	return LDA{
		TopicNum: topicNum,
		Alpha:    alpha,
		Beta:     beta,
	}
}

func (lda *LDA) SetDocs(docIds [][]float64, vocabularySize int) {
	lda.docIds = docIds
	M := len(docIds)
	lda.vocabSize = vocabularySize

	lda.zMN = make([][]int, len(docIds))
	lda.nMZ = mat.NewDense(M, lda.TopicNum, initializeArray(M*lda.TopicNum, lda.Alpha))
	lda.nZT = mat.NewDense(lda.TopicNum, lda.vocabSize, initializeArray(lda.TopicNum*lda.vocabSize, lda.Beta))
	lda.nZ = initializeArray(lda.TopicNum, float64(lda.vocabSize)*lda.Beta)

	rand.Seed(time.Now().UnixNano())
	for m, docId := range docIds {
		zN := make([]int, len(docId))
		for i := 0; i < len(docId); i++ {
			zN[i] = rand.Intn(lda.TopicNum)
		}
		lda.zMN[m] = zN

		for i, t := range docId {
			z := zN[i]
			lda.nMZ.Set(m, z, lda.nMZ.At(m, z)+1)
			lda.nZT.Set(z, int(t), lda.nZT.At(z, int(t))+1)
			lda.nZ[z] += 1
		}
	}
}

func (lda *LDA) Inference() {
	for m, docId := range lda.docIds {
		zN := lda.zMN[m]
		for n, t := range docId {
			z := zN[n]
			lda.nMZ.Set(m, z, lda.nMZ.At(m, z)-1)
			lda.nZT.Set(z, int(t), lda.nZT.At(z, int(t))-1)
			lda.nZ[z] -= 1

			// トピックの事後確率からサンプリングを行う
			vd := mat.NewVecDense(lda.TopicNum, nil)
			vd.MulElemVec(lda.nZT.ColView(int(t)), lda.nMZ.RowView(m))
			subVec := mat.NewVecDense(lda.TopicNum, lda.nZ)
			vd.SubVec(vd, subVec)
			vd.DivElemVec(vd, mat.NewVecDense(lda.TopicNum, initializeArray(lda.TopicNum, sum(vd.RawVector().Data))))
			c := distuv.NewCategorical(vd.RawVector().Data, nil)
			newZ := int(c.Rand())

			zN[n] = newZ
			lda.nMZ.Set(m, newZ, lda.nMZ.At(m, newZ)+1)
			lda.nZT.Set(newZ, int(t), lda.nZT.At(newZ, int(t))+1)
			lda.nZ[newZ] += 1
		}
	}
}

func initializeArray(size int, scalar float64) []float64 {
	array := make([]float64, size)
	for i := 0; i < size; i++ {
		array[i] = scalar
	}
	return array
}

func sum(arr []float64) float64 {
	res := 0.
	for _, ar := range arr {
		res += ar
	}
	return res
}

func main() {
}
