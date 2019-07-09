package utils

import (
	"errors"
	"math"
)

func EuclideanDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("配列の長さは同じにしてください")
	}

	count := 0.
	for i := 0; i < len(a); i++ {
		count += math.Pow(a[i]-b[i], 2)
	}
	return math.Sqrt(count), nil
}

func inner(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("配列の長さは同じにしてください")
	}
	result := 0.
	for i := 0; i < len(a); i++ {
		result += a[i] * b[i]
	}
	return result, nil
}

func norm(a []float64) float64 {
	result := 0.
	for _, p := range a {
		result += math.Pow(p, 2)
	}
	return math.Sqrt(result)
}

func CosineDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("配列の長さは同じにしてください")
	}
	inner, _ := inner(a, b)
	normA := norm(a)
	normB := norm(b)
	return inner / (normA * normB), nil
}
