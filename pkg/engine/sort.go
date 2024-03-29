package engine

import (
	chess "github.com/Yoshi-Exeler/chesslib"
	opening "github.com/Yoshi-Exeler/chesslib/opening"
)

type byMVVLVA struct {
	Nodes          []*Node
	Worker         *Worker
	TPV            *chess.Move
	Depth          int
	DepthRemaining int
	Alpha          int16
	Beta           int16
	Max            bool
	Quiescence     bool
}

func (a byMVVLVA) Len() int      { return len(a.Nodes) }
func (a byMVVLVA) Swap(i, j int) { a.Nodes[i], a.Nodes[j] = a.Nodes[j], a.Nodes[i] }
func (a byMVVLVA) Less(i, j int) bool {
	// Check for previously established pvs
	if !a.Quiescence && SameMove(a.Nodes[i].Value, a.TPV) && !SameMove(a.Nodes[j].Value, a.TPV) {
		return true
	}
	// Killer Moves have second highest priority
	if a.Worker.IsKillerMove(a.Nodes[i].Value, a.Nodes[i].Depth) && !a.Worker.IsKillerMove(a.Nodes[j].Value, a.Nodes[j].Depth) {
		return true
	}
	// Promotions will be searched next
	if a.Nodes[i].Value.Promo() != chess.NoPieceType && a.Nodes[j].Value.Promo() == chess.NoPieceType {
		return true
	}
	// If both moves are captures, sort them using MVVLVA
	if a.Nodes[i].Value.HasTag(chess.Capture) && a.Nodes[j].Value.HasTag(chess.Capture) {
		return a.Worker.captureValue(a.Nodes[i].Value) > a.Worker.captureValue(a.Nodes[j].Value)
	}
	// If one move is a capture search it first
	if a.Nodes[i].Value.HasTag(chess.Check) && !a.Nodes[j].Value.HasTag(chess.Check) {
		return true
	}
	// if a move is a check, search it before positionals
	if a.Nodes[i].Value.HasTag(chess.Capture) && !a.Nodes[j].Value.HasTag(chess.Capture) {
		return true
	}

	return false
}

// captureValue returns the material change caused by the capture
func (w *Worker) captureValue(move *chess.Move) int16 {
	// Get the Victim and Attacker Pieces
	victim := w.Simulation.Board().Piece(move.S2)
	attacker := w.Simulation.Board().Piece(move.S1)
	// Get the Values for the Pieces
	victimValue := abs(pieceValues[victim])
	attackerValue := abs(pieceValues[attacker])
	// Calculate the Capture Differential and return it
	return victimValue - attackerValue
}

type byOpeningLength []*opening.Opening

func (a byOpeningLength) Len() int           { return len(a) }
func (a byOpeningLength) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byOpeningLength) Less(i, j int) bool { return len(a[i].PGN()) > len(a[j].PGN()) }
