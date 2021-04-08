package engine

import (
	"fmt"
	"sort"

	chess "github.com/Yoshi-Exeler/chesslib"
)

// GetOpeningMove returns an opening theory move from ECO if one exists
func (e *Engine) GetOpeningMove() *chess.Move {
	prevMoves := e.Game.Moves()
	moveIndex := len(prevMoves)
	opennings := e.ECO.Possible(e.Game.Moves())
	sort.Sort(byOpeningLength(opennings))
	for _, op := range opennings {
		moves := op.Game().Moves()
		if len(moves) > moveIndex {
			usable := true
			for idx, mv := range prevMoves {
				if moves[idx].String() != mv.String() {
					usable = false
				}
			}
			if usable {
				fmt.Println("[YGG] Playing the", op.Title())
				fmt.Println("[YGG] Next move", moves[moveIndex])
				return moves[moveIndex]
			}
		}
		break
	}
	fmt.Println("[YGG] No move was found in ECO")
	return nil
}

// TODO: integrate this into GetOpeningMove to save a lookup
// GetOpeningName returns the current theory name
func (e *Engine) GetOpeningName() string {
	prevMoves := e.Game.Moves()
	moveIndex := len(prevMoves)
	opennings := e.ECO.Possible(e.Game.Moves())
	sort.Sort(byOpeningLength(opennings))
	for _, op := range opennings {
		moves := op.Game().Moves()
		if len(moves) > moveIndex {
			usable := true
			for idx, mv := range prevMoves {
				if moves[idx].String() != mv.String() {
					usable = false
				}
			}
			if usable {
				return op.Title()
			}
		}
		break
	}
	return ""
}
