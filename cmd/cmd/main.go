package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
	"unicode"
	"yggdrasil/pkg/engine"

	chess "github.com/Yoshi-Exeler/chesslib"
)

var game *chess.Game
var reader *bufio.Reader
var wmove = true
var clear map[string]func() //create a map for storing clear funcs

func main() {
	// Create STDIN Reader
	reader = bufio.NewReader(os.Stdin)
	// Create the Game
	fenStr := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	//fenStr := "8/8/5k2/8/8/1pp4K/1r5P/7R b - - 0 1"
	fen, _ := chess.FEN(fenStr)
	game = chess.NewGame(fen)
	// Enter Game Loop
	for {
		Turn()
	}

}

// Turn will cause the active player to take a turn
func Turn() {
	//CallClear()
	fmt.Println(game.Position().Board().Draw())
	if len(game.Moves()) > 0 {
		if wmove {
			fmt.Print("Black Played: ")
		} else {
			fmt.Print("White Played: ")
		}
		fmt.Println(game.Moves()[len(game.Moves())-1])
		fmt.Println("Board Evaluation (Black Perspective):", math.Round(float64(engine.EvaluatePosition(game.Position(), chess.Black))*100)/100)
		fmt.Println(engine.NewEngine(game, chess.Black).GetOpeningName())
		if game.Position().Status() == chess.Checkmate {
			if wmove {
				fmt.Print("Black Wins by Checkmate!")
				os.Exit(0)
			} else {
				fmt.Print("White Wins by Checkmate!")
				os.Exit(0)
			}
		}
		if game.Position().Status() == chess.Stalemate {
			fmt.Println("The Game is a Draw")
			os.Exit(0)
		}
	}
	if wmove {
		//MakeRandomMove()

		//tree := engine.NewEngine(game, chess.White)
		//mv := tree.SearchMinimax(4)
		//game.Move(mv)

		wmove = !wmove
	w_begin_input:
		inp := SpaceMap(ReadSTDIN())
		err := game.MoveStr(inp)
		if err != nil {
			fmt.Printf("Your input was invalid, error: %v\n", err)
			goto w_begin_input
		}

	} else {
		start := time.Now()
		eng := engine.NewEngine(game, chess.Black)
		mv := eng.Search()
		end := time.Now()
		stime := uint(end.Sub(start))
		sSec := stime / 1000 / 1000
		fmt.Printf("Search Completed in %vms", sSec)
		game.Move(mv)
		wmove = !wmove
	}
}

func SpaceMap(str string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return -1
		}
		return r
	}, str)
}

func MakeRandomMove() {
	rand.Seed(time.Now().Unix()) // initialize global pseudo random generator
	game.Move(game.ValidMoves()[rand.Intn(len(game.ValidMoves()))])
}

// ReadSTDIN will prompt the user and read stdin
func ReadSTDIN() string {
	text, _ := reader.ReadString('\n')
	return text
}

func init() {
	clear = make(map[string]func()) //Initialize it
	clear["linux"] = func() {
		cmd := exec.Command("clear") //Linux example, its tested
		cmd.Stdout = os.Stdout
		cmd.Run()
	}
	clear["windows"] = func() {
		cmd := exec.Command("cmd", "/c", "cls") //Windows example, its tested
		cmd.Stdout = os.Stdout
		cmd.Run()
	}
}

func CallClear() {
	value, ok := clear[runtime.GOOS] //runtime.GOOS -> linux, windows, darwin etc.
	if ok {                          //if we defined a clear func for that platform:
		value() //we execute it
	} else { //unsupported platform
		panic("Your platform is unsupported! I can't clear terminal screen :(")
	}
}
