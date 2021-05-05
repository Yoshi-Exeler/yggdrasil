package engine

import (
	"fmt"
	"sort"
	"sync"
	"time"

	transposition "yggdrasil/pkg/transposition"

	chess "github.com/Yoshi-Exeler/chesslib"
	opening "github.com/Yoshi-Exeler/chesslib/opening"
)

/* Possible Optimizations
*  Improve Hash table Housekeeping
*  Endgame Eval and Sequence Database
*  https://www.duo.uio.no/bitstream/handle/10852/53769/master.pdf?sequence=1&isAllowed=y
*  Implement Null-Move Pruning
*  Implement Futility Pruning at the Frontier (depth==1)
*  Scope out the Viability of SEE (Static Exchange Evaluation) in this framework
*  Implement an AlphaBeta improvement (MTD(f),PVS,BNS)
*  Benchmark NegaMax vs Minimax
*  Reduce the Code as far as possible
*  Refractor API's
*  Expand testing Framework
*  Expand Benchmarks
 */

/* Known Problems & Bugs
* Sometimes lower depth sequences are playend in Searchmode 2 & 3
* Transposition sort must check remaining depth (!!!)
 */

// Engine is the Minimax Engine
type Engine struct {
	SearchMode         uint8 // 1 = SyncFull, 2 = SyncIterative, 3 = AMP
	Depth              int
	ProcessingTime     time.Duration
	UseOpeningTheory   bool
	Game               *chess.Game
	Origin             chess.Position
	ECO                *opening.BookECO
	Color              chess.Color
	SharedTable        *transposition.Table
	PVTable            *PVT
	TableHits          uint
	GeneratedNodes     uint
	QGeneratedNodes    uint
	EvaluatedNodes     uint
	Visited            uint
	QVisited           uint
	LoadedFromPosCache uint
	NMP                uint
	DeltaP             uint
}

type Worker struct {
	Simulation  *chess.Position
	Evaluation  int16
	Origin      chess.Position
	Root        *Node
	Depth       int
	Engine      *Engine
	KillerMoves map[int8][2]*chess.Move
	KMTFlip     bool
	Stop        bool
}

// Node is a node in the Engine
type Node struct {
	Parent        *Node
	Depth         int8
	BestChild     *Node
	BestChildEval int16
	Value         *chess.Move
	StatusEval    int16
	StatusChecked bool
	Hash          uint64
	TableChecked  bool
	EntryUsable   bool
}

// Snapshot represents a snapshot of an evaluation state
type Snapshot struct {
	Position   chess.Position
	Evaluation int16
}

// NewEngine returns a new Engine from the specified game
func NewEngine(game *chess.Game, clr chess.Color) *Engine {
	return &Engine{UseOpeningTheory: false, Depth: 1, SearchMode: 3, ProcessingTime: time.Second * 15, PVTable: NewPVT(), SharedTable: &transposition.Table{Table: &sync.Map{}}, ECO: opening.NewBookECO(), Game: game, Color: clr, Origin: *game.Position()}
}

// Search will Search will produce a move to play next
func (e *Engine) Search() *chess.Move {
	// Check that we can actually play a move here
	if statusIsEnd(e.Origin.Status()) {
		fmt.Println("Aborting search, no possible moves exist")
		return nil
	}
	// if the opening theory database is enabled
	if e.UseOpeningTheory {
		// Check the ECO-Database for an Opening Move
		move := e.GetOpeningMove()
		// if a move was found return it
		if move != nil {
			return move
		}
		// if no move was found, stop checking the Database from now on
		e.UseOpeningTheory = false
	}
	// Declare the best node and score
	var bestNode *Node
	var bestScore int16
	// Switch the SearchMode
	switch e.SearchMode {
	case 1:
		// TODO: optimize recast int()
		// SynchronousFull will search Synchronously until the specified depth is reached
		bestNode, bestScore = e.SearchSynchronousFull(e.Game, (e.Depth))
	case 2:
		// SynchronousIterative will search Synchronously using Iterative Deepening until the
		// specified processing time has elapsed
		bestNode, bestScore = e.SearchSynchronousIterative(e.Game, e.ProcessingTime)
	case 3:
		// SynchronousIterative will search Synchronously using Iterative Deepening until the
		// specified processing time has elapsed
		bestNode, bestScore = e.SearchSMP(e.Game, e.ProcessingTime, 7)
	}
	// Get the Origin Move of the Best Leaf, i.e. the move to play
	origin := bestNode.getSequence(e)[0]
	// Log some information about the search we just completed
	fmt.Printf("Principal Variation:%v\nMinimax Value:%v\nDepth:%v\n", bestNode.getSequence(e), bestScore, bestNode.Depth)
	fmt.Printf("Generated:%v Visited:%v Evaluated:%v LoadedFromCache:%v\n", e.GeneratedNodes, e.Visited, e.EvaluatedNodes, e.LoadedFromPosCache)
	fmt.Printf("QGenerated:%v QVisited:%v TranspositionHits:%v DeltaP:%v\n", e.QGeneratedNodes, e.QVisited, e.TableHits, e.DeltaP)
	if bestScore > 10000 {
		fmt.Printf("Checkmate for black in %v ply\n", abs(bestScore-10100))
	}
	if bestScore < -10000 {
		fmt.Printf("Checkmate for white in %v ply\n", abs(bestScore+10100))
	}
	return origin
}

// SearchSynchronousFull will fully search the tree to the Specified Depth
func (e *Engine) SearchSynchronousFull(game *chess.Game, depth int) (*Node, int16) {
	w := &Worker{Engine: e, Simulation: game.Position(), Origin: *game.Position(), Depth: depth, Root: &Node{Depth: 0, StatusEval: MinScore, StatusChecked: true}}
	w.Evaluation = w.Simulation.Board().EvaluateFastI16()
	w.KillerMoves = make(map[int8][2]*chess.Move)
	return w.MinimaxPruning(w.Root, MinScore, MaxScore, depth, true, true)
}

// SearchSynchronousIterative
func (e *Engine) SearchSynchronousIterative(game *chess.Game, processingTime time.Duration) (*Node, int16) {
	currentDepth := 1
	bestNode := &Node{}
	bestScore := MinScore
	w := &Worker{Engine: e, Simulation: game.Position(), Origin: *game.Position(), Root: &Node{Depth: 0, StatusEval: MinScore, StatusChecked: true}}
	w.KillerMoves = make(map[int8][2]*chess.Move)
	go func() {
		time.Sleep(processingTime)
		w.Stop = true
	}()
	start := time.Now()
	for {
		w.Evaluation = w.Simulation.Board().EvaluateFastI16()
		nd, score := w.MinimaxPruning(w.Root, MinScore, MaxScore, currentDepth, true, true)
		if w.Stop {
			break
		}
		elapsed := time.Since(start)
		fmt.Printf("Completed depth=%v	target=%v	elapsed=%v	nodes=%v\n", currentDepth, score, elapsed, w.Engine.GeneratedNodes+w.Engine.QGeneratedNodes)
		bestNode = nd
		bestScore = score
		// Update the Principal Variation Table
		e.PVTable.Update(bestNode.getSequence(e), currentDepth-1)
		currentDepth++
	}
	return bestNode, bestScore
}

func (e *Engine) SearchSMP(game *chess.Game, processingTime time.Duration, workers uint8) (*Node, int16) {
	maxdepth := 0
	stop := false
	bestNode := &Node{}
	bestScore := MinScore
	work := func() {
		currentDepth := 1
		w := &Worker{Engine: e, Simulation: game.Position(), Origin: *game.Position(), Root: &Node{Depth: 0, StatusEval: MinScore, StatusChecked: true}}
		w.KillerMoves = make(map[int8][2]*chess.Move)
		go func() {
			time.Sleep(processingTime)
			w.Stop = true
			stop = true
		}()
		start := time.Now()
		for {
			w.Evaluation = w.Simulation.Board().EvaluateFastI16()
			nd, score := w.MinimaxPruning(w.Root, MinScore, MaxScore, currentDepth, true, true)
			if w.Stop {
				break
			}
			elapsed := time.Since(start)
			fmt.Printf("Completed depth=%v	target=%v	elapsed=%v	nodes=%v\n", currentDepth, score, elapsed, w.Engine.GeneratedNodes+w.Engine.QGeneratedNodes)
			if currentDepth > maxdepth {
				maxdepth = currentDepth
				bestNode = nd
				bestScore = score
				// Update the Principal Variation Table
				e.PVTable.Update(bestNode.getSequence(e), currentDepth)
			}
			currentDepth++
		}
	}
	for i := 0; i < int(workers); i++ {
		go work()
	}
	for {
		if stop {
			break
		}
		time.Sleep(time.Millisecond * 25)
	}
	return bestNode, bestScore
}

// MakeMove will make the Specified move in the Workers internal Simulation and incrementally update the Evaluation
func (w *Worker) MakeMove(node *Node) *Snapshot {
	w.Engine.EvaluatedNodes++
	// Make our snapshot to return
	ret := Snapshot{Position: *w.Simulation, Evaluation: w.Evaluation}
	// Snapshot the Previous board state
	preMoveBoard := *w.Simulation.Board()
	// Update the Current Simualtion state
	w.Simulation = w.Simulation.Update(node.Value)
	// Get the Piece Being Moved
	piece := preMoveBoard.Piece(node.Value.S1)
	// Check if a piece was Captured
	if node.Value.HasTag(chess.Capture) {
		// Get the Captured Piece
		capturedPiece := preMoveBoard.Piece(node.Value.S2)
		// Remove its evaluation from the current Evaluation
		w.Evaluation -= pieceValues[capturedPiece]
		// Remove its Positional Evaluation
		if capturedPiece == chess.WhitePawn {
			w.Evaluation -= whitePawnPositionalValuesI16[node.Value.S2]
		} else if capturedPiece == chess.BlackPawn {
			w.Evaluation -= blackPawnPositionalValuesI16[node.Value.S2]
		} else if capturedPiece == chess.BlackBishop || capturedPiece == chess.BlackKnight || capturedPiece == chess.BlackRook || capturedPiece == chess.BlackQueen {
			w.Evaluation -= blackPiecePositionalValuesI16[node.Value.S2]
		} else if capturedPiece == chess.WhiteBishop || capturedPiece == chess.WhiteKnight || capturedPiece == chess.WhiteRook || capturedPiece == chess.WhiteQueen {
			w.Evaluation -= whitePiecePositionalValuesI16[node.Value.S2]
		}
	}
	// King moves dont have positional evaluation
	if piece == chess.BlackKing || piece == chess.WhiteKing {
		return &ret
	}
	// Otherwise we incrementally Update the Evaluation
	// Get the Status of the resulting board
	status := w.Simulation.Status()
	// Check for Checkmate
	if status == chess.Checkmate {
		if w.Simulation.Turn() == chess.Black {
			// Override the Evaluation with the Checkmate Value for White
			w.Evaluation = -10000 - (100 - int16(node.Depth))
		}
		// Override the Evaluation with the Checkmate Value for Black
		w.Evaluation = 10000 + (100 - int16(node.Depth))
	}
	// Get the Positional Differential caused by the move
	if piece == chess.WhitePawn {
		w.Evaluation += whitePawnPositionalValuesI16[node.Value.S2] - whitePawnPositionalValuesI16[node.Value.S1]
		// Check if a Promotion Happened
		prom := node.Value.Promo()
		if prom != chess.NoPieceType {
			// Add the Piece that was Promoted into to the Evaluation and remove the Pawn that was used
			w.Evaluation += pieceValues[prom] - pieceValues[chess.WhitePawn]
		}
	} else if piece == chess.BlackPawn {
		w.Evaluation += blackPawnPositionalValuesI16[node.Value.S2] - blackPawnPositionalValuesI16[node.Value.S1]
		// Check if a Promotion Happened
		prom := node.Value.Promo()
		if prom != chess.NoPieceType {
			// Add the Piece that was Promoted into to the Evaluation and remove the Pawn that was used
			w.Evaluation += pieceValues[prom+6] - pieceValues[chess.BlackPawn]
		}
	} else if piece == chess.BlackBishop || piece == chess.BlackKnight || piece == chess.BlackRook || piece == chess.BlackQueen {
		w.Evaluation += blackPiecePositionalValuesI16[node.Value.S2] - blackPiecePositionalValuesI16[node.Value.S1]
	} else if piece == chess.WhiteBishop || piece == chess.WhiteKnight || piece == chess.WhiteRook || piece == chess.WhiteQueen {
		w.Evaluation += whitePiecePositionalValuesI16[node.Value.S2] - whitePiecePositionalValuesI16[node.Value.S1]
	}
	// Return the Restore value
	return &ret
}

// Unmake move will revert a move and decrement the Evaluation accordingly
func (w *Worker) UnmakeMove(snap *Snapshot) {
	w.Simulation = &snap.Position
	w.Evaluation = snap.Evaluation
}

// IsQuietMove returns wether a move is a Quiet (a pure positional move)
func (w *Worker) IsQuietMove(mv *chess.Move) bool {
	return !mv.HasTag(chess.Capture) && !mv.HasTag(chess.Check) && mv.Promo() == chess.NoPieceType
}

// SaveKillerMove will save the specified move to the KillerMove table, ensuring no duplicates
func (w *Worker) SaveKillerMove(move *chess.Move, depth int8) {
	if w.KillerMoves[depth][0] == nil {
		w.KillerMoves[depth] = [2]*chess.Move{move, w.KillerMoves[depth][1]}
		return
	}
	if w.KillerMoves[depth][1] == nil {
		w.KillerMoves[depth] = [2]*chess.Move{w.KillerMoves[depth][0], move}
		return
	}
	// Check that this move is not currently in the KMT
	if !SameMove(w.KillerMoves[depth][0], move) && !SameMove(w.KillerMoves[depth][1], move) {
		// Add it to a slot in the KMT
		if w.KMTFlip {
			w.KillerMoves[depth] = [2]*chess.Move{move, w.KillerMoves[depth][1]}
		} else {
			w.KillerMoves[depth] = [2]*chess.Move{w.KillerMoves[depth][0], move}
		}
		w.KMTFlip = !w.KMTFlip
	}
}

// IsKillerMove queries the KMT to check whether or not a move is a killer move
func (w *Worker) IsKillerMove(mv *chess.Move, depth int8) bool {
	if w.KillerMoves[depth][0] != nil {
		if SameMove(w.KillerMoves[depth][0], mv) {
			return true
		}
	}
	if w.KillerMoves[depth][1] != nil {
		if SameMove(w.KillerMoves[depth][1], mv) {
			return true
		}
	}
	return false
}

// QuiescenseSearch will Search the Specified Position with a limited SubSearch to mitigate the Horizon effect
// by extending the Search until a Stable(Quiet) Position can be statically evaluated or the Entire Branch can
// be failed by cutof (soft or hard fail). The Expected Branching factor in a Quiescence Search is around 7 in
// the midgame so each search should not be too expensive.
func (w *Worker) QuiescenseSearch(node *Node, alpha int16, beta int16, max bool) int16 {
	// Increment the Nodes Visited by QuiescenceSearch
	w.Engine.QVisited++
	// Check if the Current Node is a Terminating node
	nseval := w.NodeStatusScore(node, max)
	if nseval > MinScore {
		// if this node is a terminating node, return its evaluation
		return nseval
	}
	// if we are in a maximizing branch
	if max {
		// Check if the Standing Pat Causes Beta Cutof
		if w.Evaluation >= beta {
			// return beta (fail hard)
			return beta
		}
		// Check if the standing pat is greater than alpha
		if w.Evaluation > alpha {
			// raise alpha
			alpha = w.Evaluation
		}
		// Check if alpha can even be improved
		if w.Evaluation < alpha-DeltaMax {
			return alpha
		}
		// Generate the Unstable Children of the Node, make sure not to influence the sim
		unstable := node.GetUnstableLeaves(w, max, node.IsCheck())
		// Sort the Unstable nodes using MVVLVA to increase search speed
		sort.Sort(byMVVLVA{Nodes: unstable, Worker: w, PV: w.Engine.PVTable.GetPV(), Quiescence: true})
		// Iterate over the Unstable children of the node
		for _, child := range unstable {
			// Make sure not to leak a pointer to the Iterator
			alloc := *child
			// Only if the current Move is a Capture, we do delta pruning
			if alloc.Value.HasTag(chess.Capture) && alloc.Value.Promo() == chess.NoPieceType {
				// Get the Piece Being Captured
				capturedPiece := w.Simulation.Board().Piece(alloc.Value.S2)
				// Check that the Value of the Captured Piece Plus Delta is greater than alpha (Delta Pruning)
				if (w.Evaluation - pieceValues[capturedPiece] + Delta) < alpha {
					w.Engine.DeltaP++
					continue
				}
			}
			// Simulate the Move of the current Child
			snapshot := w.MakeMove(&alloc)
			// Set the Depth of the Child Node
			alloc.Depth = node.Depth + 1
			// Call QuiescenseSearch recursively
			score := w.QuiescenseSearch(&alloc, alpha, beta, !max)
			// Restore the Snapshot, we cannot just invert the move and update the simulation beacuse of promotions
			w.UnmakeMove(snapshot)
			// Check if the current child causes beta cuttof
			if score >= beta {
				// fail hard
				return beta
			}
			// Check if the Score of the current child is greater than alpha
			if score > alpha {
				// raise alpha
				alpha = score
			}
		}
		// if we are in a minimizing branch, we must treat alpha as the lower bound and beta as the upper bound
	} else {
		// Check if the Standing Pat Causes Alpha Cuttof
		if w.Evaluation <= alpha {
			// fail soft
			return alpha
		}
		// Check if the standing pat lowers beta
		if w.Evaluation < beta {
			// lower beta
			beta = w.Evaluation
		}
		// Check if beta can even be lowered
		if w.Evaluation > beta+DeltaMax {
			return beta
		}
		// Generate the Unstable Children of the Node
		unstable := node.GetUnstableLeaves(w, max, node.IsCheck())
		// Sort the Unstable nodes using MVVLVA to increase search speed
		sort.Sort(byMVVLVA{Nodes: unstable, Worker: w, PV: w.Engine.PVTable.GetPV(), Quiescence: true})
		// Iterate over the Unstable children of the node
		for _, child := range unstable {
			// Make sure not to leak a pointer to the Iterator
			alloc := *child
			// Only if the current Move is a Capture, we do delta pruning
			if alloc.Value.HasTag(chess.Capture) && alloc.Value.Promo() == chess.NoPieceType {
				// Get the Piece Being Captured
				capturedPiece := w.Simulation.Board().Piece(alloc.Value.S2)
				// Check that the Value of the Captured Piece Plus Delta is greater than alpha (Delta Pruning)
				if (w.Evaluation - pieceValues[capturedPiece] - Delta) > beta {
					w.Engine.DeltaP++
					continue
				}
			}
			// Simulate the Move of the current Child
			snapshot := w.MakeMove(&alloc)
			// Set the Depth of the Child Node
			alloc.Depth = node.Depth + 1
			// Call QuiescenseSearch recursively
			score := w.QuiescenseSearch(&alloc, alpha, beta, !max)
			// Restore the Snapshot, we cannot just invert the move and update the simulation beacuse of promotions
			w.UnmakeMove(snapshot)
			// Check if the Current child causes Alpha cuttof
			if score <= alpha {
				// fail soft
				return alpha
			}
			// Check if the Score lowers beta
			if score < beta {
				// lower beta
				beta = score
			}
		}
	}
	// Return the Appropriate upper bound
	if max {
		return alpha
	} else {
		return beta
	}

}

// MinimaxPruning will Dynamically build a Searchtree starting with the node specified in the Initial Call
// as the Root node. The Tree will be Searched with the Minimax Algorithm. Alpha/Beta Pruning will be used
// to significantly reduce the Required workload. Quiescence Search and an In-Check Search Extension will
// be used to mitigate the Horizon effect to a reasonable degree.
func (w *Worker) MinimaxPruning(node *Node, alpha int16, beta int16, depth int, max bool, nmp bool) (*Node, int16) {
	// Immediately Exit if our worker was stopped, returns will be ignored
	if w.Stop {
		return node, 0
	}
	// Query the Transposition Table
	entry := w.Engine.SharedTable.Query(node.getHash(w))
	// If the Transposition Table Query Produced a Hit
	if entry != nil {
		// Check that we are Min/Max aligned and this entry contains usefull information
		if depth <= entry.Depth && max == entry.Max {
			// If this entry contains an exact score
			if entry.Exact {
				w.Engine.TableHits++
				// We skip all further processing of this node and return the stored result
				return node, entry.Score
			}
			// If this entry is an upper bound that raises or matches the current upper bound
			if !entry.Exact && entry.Alpha && entry.Score >= alpha {
				w.Engine.TableHits++
				// We skip all further processing of this node and return the stored result
				return node, entry.Score
			}
			// If this entry is a lower bound that lowers or matches the current lower bound
			if !entry.Exact && !entry.Alpha && entry.Score <= beta {
				w.Engine.TableHits++
				// We skip all further processing of this node and return the stored result
				return node, entry.Score
			}
		}
	}
	// Increment the Nodes visited by the Search
	w.Engine.Visited++
	// Check if the Current Node is a Terminating node i.e. Stalemate or Checkmate
	nseval := w.NodeStatusScore(node, max)
	if nseval > MinScore {
		// If this node is a Terminating Node return its evaluation
		return node, nseval
	}
	// If we are at or below the target depth and not in check, Evaluate this Node using Quiescence Search if neccessary
	if depth <= 0 && !node.IsCheck() {
		// Perform Quiescence Search to get a score
		score := w.QuiescenseSearch(node, alpha, beta, max)
		// Commit the Result to the Transposition Table as an exact score with depth 0 (evaluation table entry)
		//w.Engine.SharedTable.Commit(w.Simulation.Hash(), transposition.Entry{Score: score, Exact: true, Max: max, Alpha: false, Depth: 0})
		return node, score
	}

	// If we are in a maximizing Branch
	if max {
		best := MinScore
		bestNode := &Node{}
		// Generate the Children of the current node
		leaves := node.GenerateLeaves(w)
		// Sort them by their Expected impact on the Evaluation (Captures and Checks first)
		sort.Sort(byMVVLVA{Nodes: leaves, Worker: w, PV: w.Engine.PVTable.GetPV(), Depth: int(node.Depth), Alpha: alpha, Beta: beta, Max: max})
		// Iterate over the Children of the Current Node
		for _, child := range leaves {
			// Prevent Leaking the Loop iterator
			alloc := *child
			// Simulate the Move of the current Child
			snapshot := w.MakeMove(&alloc)
			// Recursively Call Minimax for each child
			nod, ev := w.MinimaxPruning(&alloc, alpha, beta, depth-1, false, true)
			// Restore the Snapshot, we cannot just invert the move and update the simulation beacuse of promotions
			w.UnmakeMove(snapshot)
			// if the Current Child is better than the Previous best, it becomes the new Best
			if ev > best {
				best = ev
				bestNode = nod
			}
			// Raise Alpha if the Current Child is Greater than Alpha
			alpha = i16max(alpha, ev)
			// Check if the Current Child Causes Pruning, in which case we can stop the Iteration immediately
			if beta <= alpha {
				// Save this Move to the Killer Move Table to improve move ordering in sibiling nodes
				w.SaveKillerMove(alloc.Value, alloc.Depth)
				break
			}
		}
		// Check if this is an upper bound on the minimax value
		if best == alpha {
			// Commit the result to the transposition table
			w.Engine.SharedTable.Commit(node.getHash(w), transposition.Entry{Score: best, Exact: false, Max: max, Alpha: true, Depth: int(node.Depth)})
		} else {
			// Commit the result to the transposition table
			w.Engine.SharedTable.Commit(node.getHash(w), transposition.Entry{Score: best, Exact: true, Max: max, Alpha: false, Depth: int(node.Depth)})
		}
		return bestNode, best
	}
	// If we are in a minimizing Branch
	worst := MaxScore
	worstNode := &Node{}
	// Generate the Children of the current node
	leaves := node.GenerateLeaves(w)
	// Sort them by their Expected impact on the Evaluation (Captures and Checks first)
	sort.Sort(byMVVLVA{Nodes: leaves, Worker: w, PV: w.Engine.PVTable.GetPV(), Depth: int(node.Depth), Alpha: alpha, Beta: beta, Max: max})
	// Iterate over the Children of the Current Node
	for _, child := range leaves {
		// Prevent Leaking the Loop iterator
		alloc := *child
		// Simulate the Move of the current Child
		snapshot := w.MakeMove(&alloc)
		// Recursively Call Minimax for each child
		nod, ev := w.MinimaxPruning(child, alpha, beta, depth-1, true, true)
		// Restore the Snapshot, we cannot just invert the move and update the simulation beacuse of promotions
		w.UnmakeMove(snapshot)
		// if the Current Child is worse than the Previous worst, it becomes the new worst
		if ev < worst {
			worst = ev
			worstNode = nod
		}
		// Lower Beta if the Current Child is Lower than Alpha
		beta = i16min(beta, ev)
		// Check if the Current Child Causes Pruning, in which case we can stop the Iteration immediately
		if beta <= alpha {
			// Save this Move to the Killer Move Table to improve move ordering in sibiling nodes
			w.SaveKillerMove(alloc.Value, alloc.Depth)
			break
		}
	}
	// Check if this is an upper bound on the minimax value
	if worst == beta {
		// Commit the result to the transposition table
		w.Engine.SharedTable.Commit(node.getHash(w), transposition.Entry{Score: worst, Exact: false, Max: max, Alpha: false, Depth: int(node.Depth)})
	} else {
		// Commit the result to the transposition table
		w.Engine.SharedTable.Commit(node.getHash(w), transposition.Entry{Score: worst, Exact: true, Max: max, Alpha: false, Depth: int(node.Depth)})
	}
	return worstNode, worst
}

// NodeStatusScore Returns the Status evaluation of this node
func (w *Worker) NodeStatusScore(n *Node, inv bool) int16 {
	// if this is the root dont check
	if n.Value == nil {
		return MinScore
	}
	// if this has a status already calculated, return it
	if n.StatusChecked {
		return n.StatusEval
	}
	// Evaluate the Status of the node using the specified inversion
	n.StatusEval = statusEval(w.Simulation.Status(), inv, int(n.Depth))
	// Set a flag that the status was evaluated so we dont evaluate again
	n.StatusChecked = true
	return n.StatusEval
}

// statusIsEnd will return wether or not the specified method ends the game
func statusIsEnd(s chess.Method) bool {
	return (s == chess.Checkmate || s == chess.Stalemate)
}

// statusEval will return the status score of the node, MinScore if this is a regular node
// 1000 or -1000 for Checkmate, 0 for Stalemate
func statusEval(s chess.Method, inv bool, depth int) int16 {
	if s == chess.Checkmate {
		if inv {
			return -10000 + (-100 + int16(depth))
		}
		return 10000 + (100 - int16(depth))
	}
	if s == chess.Stalemate {
		return 0
	}
	return MinScore
}

// reverseSequence will generate the inverse of the Specified Sequence
func reverseSequence(seq []*chess.Move) []*chess.Move {
	for i, j := 0, len(seq)-1; i < j; i, j = i+1, j-1 {
		seq[i], seq[j] = seq[j], seq[i]
	}
	return seq
}

// IsCheck returns if a node is a checking node
func (n *Node) IsCheck() bool {
	if n == nil || n.Value == nil {
		return false
	}
	return n.Value.HasTag(chess.Check)
}

// GetUnstableLeaves Returns the list of Immediate Recaptures that became availabe through the last move
func (n *Node) GetUnstableLeaves(w *Worker, inv bool, check bool) []*Node {
	// Return an empty list if the Node is a terminating node, this could be optimized by only calling this from a safe ctx
	nseval := w.NodeStatusScore(n, inv)
	if nseval > MinScore {
		return make([]*Node, 0)
	}
	leaves := n.GenerateLeaves(w)
	w.Engine.GeneratedNodes -= uint(len(leaves))
	unstable := []*Node{}
	for _, nd := range leaves {
		if check || (nd.Value.HasTag(chess.Capture) || nd.Value.HasTag(chess.Check)) || nd.Value.Promo() != chess.NoPieceType {
			w.Engine.QGeneratedNodes++
			unstable = append(unstable, nd)
		}
	}
	return unstable
}

// GenerateLeaves will generate the leaves of the specified node
func (n *Node) GenerateLeaves(w *Worker) []*Node {
	// Get the Valid Moves From this Position
	vmoves := w.Simulation.ValidMoves()
	// Initialize node collection
	nds := []*Node{}
	// Convert the Moves to Nodes and add them to the Collection
	for _, mov := range vmoves {
		w.Engine.GeneratedNodes++
		nds = append(nds, &Node{Parent: n, Value: mov, Depth: n.Depth + 1})
	}
	return nds
}

// getSequence will return the full sequence of moves required to get to this node including the move of this node
func (n *Node) getSequence(e *Engine) []*chess.Move {
	seq := make([]*chess.Move, 0)
	cnode := n
	for {
		// Add the Value of the Current Node to the Sequence
		seq = append(seq, cnode.Value)
		// Stop if there is no parent
		if cnode.Parent == nil || cnode.Parent.Value == nil {
			break
		}
		// Go to the Parent
		cnode = cnode.Parent
	}
	rev := reverseSequence(seq)
	return rev
}

// getHash returns the hash of a node, using hash caching
func (n *Node) getHash(w *Worker) uint64 {
	if n.Hash != 0 {
		return n.Hash
	}
	newHash := w.Simulation.Hash()
	n.Hash = newHash
	return newHash
}

// EvalStatic will evaluate a board
func EvaluatePosition(pos *chess.Position, clr chess.Color) int16 {
	// Calculate the Status of this Position
	status := pos.Status()
	// Calculate the Score of this Position
	score := pos.Board().EvaluateFastI16()
	// Check for Checkmate
	if status == chess.Checkmate {
		if pos.Turn() == clr {
			return -10000
		}
		return 10000
	}
	// Check for Stalemate
	if status == chess.Stalemate {
		return 0
	}
	return score
}
