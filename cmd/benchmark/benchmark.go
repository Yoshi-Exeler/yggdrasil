package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"
	"yggdrasil/pkg/engine"

	chess "github.com/Yoshi-Exeler/chesslib"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func main() {
	// Setup Profiling
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	SearchPosition("5B2/PP1k2P1/p3pr1p/7p/1p2p3/8/3K2Rn/4r3 w - - 0 1")
	return
	fmt.Println("----BEGIN YGGDRASIL BENCHMARK----")
	// Benchmark Evaluation Functions
	benchmarkStaticEval()
	benchmarkStaticEval2()
	benchmarkStaticEval3()
	fmt.Println("----END  YGGDRASIL  BENCHMARK----")
}

func SearchPosition(fenStr string) {
	fen, _ := chess.FEN(fenStr)
	game := chess.NewGame(fen)
	eng := engine.NewEngine(game, chess.Black)
	eng.Search()
}

func benchmarkStaticEval() {
	fmt.Println("[EVAL1] Begin Setup")
	// Create the Game from the base position FEN
	fenStr := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	fen, _ := chess.FEN(fenStr)
	game := chess.NewGame(fen)
	// Generate the Valid Moves from the First Position
	moves := game.ValidMoves()
	// Generate Positions for those moves
	var positions []*chess.Position
	for _, mv := range moves {
		alloc := *game.Position()
		alloc.Update(mv)
		positions = append(positions, &alloc)
	}
	// Generate a List of 100 Mil Randomly Selected Positions from the list
	var randomSelection []*chess.Position
	for i := 0; i < 100000000; i++ {
		randomSelection = append(randomSelection, positions[rand.Intn(len(positions))])
	}
	// Begin Timing Execution now
	fmt.Println("[EVAL1] Setup Completed, Evaluating 100.000.000 Positions")
	start := time.Now()
	// Iterate over the List and Evaluate Each Position
	for i := 0; i < len(randomSelection); i++ {
		go randomSelection[i].Board().Evaluate(chess.Black)
	}
	// Stop Timing Execution
	end := time.Now()
	stime := uint(end.Sub(start))
	sms := stime / 1000 / 1000
	ops := (100000000 / sms) * 1000
	// Print results
	fmt.Printf("[EVAL1] Finished Benchmarking\n")
	fmt.Printf("[EVAL1] 100.000.000 Operations completed in %vms\n", sms)
	fmt.Printf("[EVAL1] That Makes %v Static Evaluations per second\n", ops)
}

func benchmarkStaticEval2() {
	fmt.Println("[EVAL2] Begin Setup")
	// Create the Game from the base position FEN
	fenStr := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	fen, _ := chess.FEN(fenStr)
	game := chess.NewGame(fen)
	// Generate the Valid Moves from the First Position
	moves := game.ValidMoves()
	// Generate Positions for those moves
	var positions []*chess.Position
	for _, mv := range moves {
		alloc := *game.Position()
		alloc.Update(mv)
		positions = append(positions, &alloc)
	}
	// Generate a List of 100 Mil Randomly Selected Positions from the list
	var randomSelection []*chess.Position
	for i := 0; i < 100000000; i++ {
		randomSelection = append(randomSelection, positions[rand.Intn(len(positions))])
	}
	// Begin Timing Execution now
	fmt.Println("[EVAL2] Setup Completed, Evaluating 100.000.000 Positions")
	start := time.Now()
	// Iterate over the List and Evaluate Each Position
	for i := 0; i < len(randomSelection); i++ {
		go randomSelection[i].Board().EvaluateFast()
	}
	// Stop Timing Execution
	end := time.Now()
	stime := uint(end.Sub(start))
	sms := stime / 1000 / 1000
	ops := (100000000 / sms) * 1000
	// Print results
	fmt.Printf("[EVAL2] Finished Benchmarking\n")
	fmt.Printf("[EVAL2] 100.000.000 Operations completed in %vms\n", sms)
	fmt.Printf("[EVAL2] That Makes %v Static Evaluations per second\n", ops)
}

func benchmarkStaticEval3() {
	fmt.Println("[EVAL3] Begin Setup")
	// Create the Game from the base position FEN
	fenStr := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	fen, _ := chess.FEN(fenStr)
	game := chess.NewGame(fen)
	// Generate the Valid Moves from the First Position
	moves := game.ValidMoves()
	// Generate Positions for those moves
	var positions []*chess.Position
	for _, mv := range moves {
		alloc := *game.Position()
		alloc.Update(mv)
		positions = append(positions, &alloc)
	}
	// Generate a List of 100 Mil Randomly Selected Positions from the list
	var randomSelection []*chess.Position
	for i := 0; i < 100000000; i++ {
		randomSelection = append(randomSelection, positions[rand.Intn(len(positions))])
	}
	// Begin Timing Execution now
	fmt.Println("[EVAL3] Setup Completed, Evaluating 100.000.000 Positions")
	start := time.Now()
	// Iterate over the List and Evaluate Each Position
	for i := 0; i < len(randomSelection); i++ {
		go randomSelection[i].Board().EvaluateFast()
	}
	// Stop Timing Execution
	end := time.Now()
	stime := uint(end.Sub(start))
	sms := stime / 1000 / 1000
	ops := (100000000 / sms) * 1000
	// Print results
	fmt.Printf("[EVAL3] Finished Benchmarking\n")
	fmt.Printf("[EVAL3] 100.000.000 Operations completed in %vms\n", sms)
	fmt.Printf("[EVAL3] That Makes %v Static Evaluations per second\n", ops)
}
