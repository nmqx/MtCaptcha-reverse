package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	"time"
)

const (
	CharMap    = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
	ServerPort = ":9091"
)

var CharToIntMap map[rune]int

func init() {
	CharToIntMap = make(map[rune]int)
	for i, c := range CharMap {
		CharToIntMap[c] = i
	}
}

//  MTCaptcha full js reversal of the logic

func foldBase64IntArray(a1 []int, foldCount int) []int {
	a2 := make([]int, len(a1))
	for i := 0; i < len(a1); i++ {
		a2[i] = a1[len(a1)-1-i]
	}

	a3 := make([]int, len(a1))
	copy(a3, a1)

	offset := 0
	x, y, z := 0, 0, 0

	for i := 0; i < foldCount; i++ {
		offset++
		for x = 0; x < len(a1); x++ {
			val := math.Floor(float64(a3[x]+a2[(x+offset)%len(a2)]) * 73.0 / 8.0)
			a3[x] = (int(val) + y + z) % 64
			z = y / 2
			y = a3[x] / 2
		}
	}
	return a3
}

func hashIntAry(arr []int) int {
	var hash int32 = 0
	for _, val := range arr {
		hash = (hash << 5) - hash + int32(val)
	}
	finalHash := int(hash)
	if finalHash < 0 {
		finalHash *= -1
	}
	return finalHash
}

func generateHypothesis3(fseed string, fslots int, fdepth int) string {
	buffer := make([]int, len(fseed))
	for i, c := range fseed {
		if val, ok := CharToIntMap[c]; ok {
			buffer[i] = val
		} else {
			buffer[i] = 0
		}
	}

	var sb strings.Builder
	for i := 0; i < fslots; i++ {
		buffer = foldBase64IntArray(buffer, 31)
		temp := foldBase64IntArray(buffer, fdepth)
		hashed := hashIntAry(temp)

		val := hashed % 4096
		c1 := val >> 6
		c2 := val & 63

		sb.WriteByte(CharMap[c1])
		sb.WriteByte(CharMap[c2])
	}
	return sb.String()
}

//   API / local server

type CalculateRequest struct {
	Fseed  string `json:"fseed"`
	Fslots int    `json:"fslots"`
	Fdepth int    `json:"fdepth"`
}

type CalculateResponse struct {
	Fa    string `json:"fa"`
	Error string `json:"error,omitempty"`
}

func calculateHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var req CalculateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		json.NewEncoder(w).Encode(CalculateResponse{Error: "Invalid JSON"})
		return
	}

	startTime := time.Now()
	fa := ""
	if req.Fseed != "" && req.Fslots > 0 {
		fa = generateHypothesis3(req.Fseed, req.Fslots, req.Fdepth)
	}
	duration := time.Since(startTime)
	fmt.Printf("Calculated FA: %s (Time: %v)\n", fa, duration)

	json.NewEncoder(w).Encode(CalculateResponse{Fa: fa})
}

func main() {
	http.HandleFunc("/calculate", calculateHandler)

	fmt.Printf("MTCaptcha Calculator Service listening on %s\n", ServerPort)
	fmt.Printf("Endpoint: POST /calculate (Body: {fseed, fslots, fdepth})\n")

	if err := http.ListenAndServe(ServerPort, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
