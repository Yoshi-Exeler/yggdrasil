package engine

import (
	"yggdrasil/pkg/transposition"

	chess "github.com/Yoshi-Exeler/chesslib"
	opening "github.com/Yoshi-Exeler/chesslib/opening"
)

type byMVVLVA struct {
	Nodes      []*Node
	Worker     *Worker
	PV         []*chess.Move
	Depth      int
	Alpha      int16
	Beta       int16
	Max        bool
	Quiescence bool
}

func (a byMVVLVA) Len() int      { return len(a.Nodes) }
func (a byMVVLVA) Swap(i, j int) { a.Nodes[i], a.Nodes[j] = a.Nodes[j], a.Nodes[i] }
func (a byMVVLVA) Less(i, j int) bool {
	/*
		// TODO: This makes the search slower, prolly doesnt work...
		if !a.Quiescence {
			// The Principal Variation from previous iterations must be searched first
			if IsPVMove(a.Nodes[i], a.PV, a.Depth) && !IsPVMove(a.Nodes[j], a.PV, a.Depth) {
				return true
			}
		}
	*/
	// Killer Moves have second highest priority
	if a.Worker.IsKillerMove(a.Nodes[i].Value, a.Nodes[i].Depth) && !a.Worker.IsKillerMove(a.Nodes[j].Value, a.Nodes[j].Depth) {
		return true
	}
	/*
		// TODO: this also makes the search slower why tho????
		if !a.Quiescence {
			// Next we search hash moves, since they are basically free information
			if a.Worker.HasTransposition(a.Nodes[i], a) && !a.Worker.HasTransposition(a.Nodes[j], a) {
				return true
			}
		}
	*/
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

func IsPVMove(nd *Node, pv []*chess.Move, depth int) bool {
	if nd.Depth > int8(len(pv)) {
		return false
	}
	return SameMove(pv[depth], nd.Value)
}

// TODO: check that the transposition actually can be used e.g. depth <= entry.depth
func (w *Worker) HasTransposition(nd *Node, a byMVVLVA) bool {
	// If we already know whether or not this node has a transposition, just return
	if nd.TableChecked {
		return nd.EntryUsable
	}
	// Obtain the Node hash as efficiently as possible
	nodeHash := uint64(0)
	// if the hash has been computed before, just reuse it
	if nd.Hash != 0 {
		nodeHash = nd.Hash
	} else {
		// otherwise compute the hash
		snap := w.Simulation
		w.Simulation = w.Simulation.Update(nd.Value)
		h := w.Simulation.Hash()
		nd.Hash = h
		w.Simulation = snap
		nodeHash = h
	}
	// Query the Transposition table
	entry := w.Engine.SharedTable.Query(nodeHash)
	result := canUseEntry(entry, a.Alpha, a.Beta, a.Depth, a.Max)
	// Remember the result so we dont recompute this inside of one search
	nd.TableChecked = true
	nd.EntryUsable = result
	return result
}

func canUseEntry(entry *transposition.Entry, alpha int16, beta int16, depth int, max bool) bool {
	// If the Transposition Table Query Produced a Hit
	if entry != nil {
		// Check that we are Min/Max aligned and this entry contains usefull information
		if depth <= entry.Depth && max == entry.Max {
			// If this entry contains an exact score
			if entry.Exact {
				// We skip all further processing of this node and return the stored result
				return true
			}
			// If this entry is an upper bound that raises or matches the current upper bound
			if !entry.Exact && entry.Alpha && entry.Score >= alpha {
				// We skip all further processing of this node and return the stored result
				return true
			}
			// If this entry is a lower bound that lowers or matches the current lower bound
			if !entry.Exact && !entry.Alpha && entry.Score <= beta {
				// We skip all further processing of this node and return the stored result
				return true
			}
		}
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
