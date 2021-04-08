package engine

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	chess "github.com/Yoshi-Exeler/chesslib"
	opening "github.com/Yoshi-Exeler/chesslib/opening"
)

/* Possible Optimizations
*  Multithreading = ~5x Performance Increase
*  Improve Hash table Housekeeping
*  LazySMP --> Search on many threads at the same time with a shared hashtable
*  Endgame Eval and Sequence Database
*  Improve Move ordering using MVVLVA
*  Save Moves that Cause Pruning, Order after Caputres
*  https://www.duo.uio.no/bitstream/handle/10852/53769/master.pdf?sequence=1&isAllowed=y
*  Remove all References from the Tree, only save the parents of the best node and the node currently being evaluated
*  Implement Delta Pruning in Quiescence Search
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
* Engine misses mate in one and instead tries to play mate in four 3k4/1pp2pr1/3P3p/7b/p7/P3q3/5R2/5K2 b - - 7 40
 */

// Engine is the Minimax Engine
type Engine struct {
	SearchMode         uint8 // 1 = SyncFull, 2 = SyncIterative, 3 = AMP
	Depth              int
	ProcessingTime     time.Duration
	UseOpeningTheory   bool
	Game               *chess.Game
	Origin             chess.Position
	SharedCache        *sync.Map
	ECO                *opening.BookECO
	Color              chess.Color
	GeneratedNodes     uint
	QGeneratedNodes    uint
	EvaluatedNodes     uint
	Visited            uint
	QVisited           uint
	LoadedFromPosCache uint
	FrontierUnstable   uint
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
	QDepth        int8
	BestChild     *Node
	BestChildEval int16
	Increment     int16
	Value         *chess.Move
	StatusEval    int16
	StatusChecked bool
}

// Snapshot represents a snapshot of an evaluation state
type Snapshot struct {
	Position   chess.Position
	Evaluation int16
}

// NewEngine returns a new Engine from the specified game
func NewEngine(game *chess.Game, clr chess.Color) *Engine {
	return &Engine{UseOpeningTheory: false, Depth: 1, SearchMode: 2, ProcessingTime: time.Second * 15, ECO: opening.NewBookECO(), Game: game, Color: clr, Origin: *game.Position(), SharedCache: &sync.Map{}}
}

// Search will Search will produce a move to play next
func (e *Engine) Search() *chess.Move {
	// Check that we can actually play a move here
	if statusIsEnd(e.Origin.Status()) {
		fmt.Println("[YGG] Aborting search, no possible moves exist")
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
	}
	// Get the Origin Move of the Best Leaf, i.e. the move to play
	origin := bestNode.getSequence(e)[0]
	// Log some information about the search we just completed
	fmt.Printf("Sequence:%v\nTarget:%v\nOrigin:%v D:%v QD:%v\n", bestNode.getSequence(e), math.Round(float64(bestScore)*100)/100, origin, bestNode.Depth, bestNode.QDepth)
	fmt.Printf("Generated:%v Visited:%v Evaluated:%v LoadedFromCache:%v\n", e.GeneratedNodes, e.Visited, e.EvaluatedNodes, e.LoadedFromPosCache)
	fmt.Printf("QGenerated:%v QVisited:%v FrontierUnstable:%v\n", e.QGeneratedNodes, e.QVisited, e.FrontierUnstable)
	return origin
}

// SearchSynchronousFull will fully search the tree to the Specified Depth
func (e *Engine) SearchSynchronousFull(game *chess.Game, depth int) (*Node, int16) {
	w := &Worker{Engine: e, Simulation: game.Position(), Origin: *game.Position(), Depth: depth, Root: &Node{Depth: 0, QDepth: 0, StatusEval: MinScore, StatusChecked: true}}
	w.Evaluation = w.Simulation.Board().EvaluateFastI16()
	w.KillerMoves = make(map[int8][2]*chess.Move)
	return w.MinimaxPruning(w.Root, MinScore, MaxScore, depth, true)
}

// SearchSynchronousIterative
func (e *Engine) SearchSynchronousIterative(game *chess.Game, processingTime time.Duration) (*Node, int16) {
	currentDepth := 1
	bestNode := &Node{}
	bestScore := MinScore
	w := &Worker{Engine: e, Simulation: game.Position(), Origin: *game.Position(), Root: &Node{Depth: 0, QDepth: 0, StatusEval: MinScore, StatusChecked: true}}
	w.KillerMoves = make(map[int8][2]*chess.Move)
	go func() {
		time.Sleep(processingTime)
		w.Stop = true
	}()
	start := time.Now()
	for {
		w.Evaluation = w.Simulation.Board().EvaluateFastI16()
		nd, score := w.MinimaxPruning(w.Root, MinScore, MaxScore, currentDepth, true)
		if w.Stop {
			break
		}
		elapsed := time.Since(start)
		fmt.Printf("Completed depth=%v target=%v elapsed=%v\n", currentDepth, score, elapsed)
		bestNode = nd
		bestScore = score
		currentDepth++
	}
	return bestNode, bestScore
}

// MakeMove will make the Specified move in the Workers internal Simulation and incrementally update the Evaluation
func (w *Worker) MakeMove(node *Node) *Snapshot {
	w.Engine.EvaluatedNodes++
	// Make our snapshot to return
	ret := Snapshot{Position: *w.Simulation, Evaluation: w.Evaluation}
	// Store the Last Evaluation
	lastEvaluation := w.Evaluation
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
	// Hash the Current Position
	posHash := w.Simulation.Hash()
	// Get the Value for this Hash
	storedV, err := w.ReadFromCache(posHash)
	// If we have a stored value for this Hash use it
	if err == nil {
		w.Engine.LoadedFromPosCache++
		w.Evaluation = storedV
		return &ret
	}
	// Otherwise we incrementally Update the Evaluation
	// Get the Status of the resulting board
	status := w.Simulation.Status()
	// Check for Checkmate
	if status == chess.Checkmate {
		if w.Simulation.Turn() == chess.Black {
			// Override the Evaluation with the Checkmate Value for White
			w.Evaluation = -10000
		}
		// Override the Evaluation with the Checkmate Value for Black
		w.Evaluation = 10000
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
			w.Evaluation += pieceValues[prom] - pieceValues[chess.BlackPawn]
		}
	} else if piece == chess.BlackBishop || piece == chess.BlackKnight || piece == chess.BlackRook || piece == chess.BlackQueen {
		w.Evaluation += blackPiecePositionalValuesI16[node.Value.S2] - blackPiecePositionalValuesI16[node.Value.S1]
	} else if piece == chess.WhiteBishop || piece == chess.WhiteKnight || piece == chess.WhiteRook || piece == chess.WhiteQueen {
		w.Evaluation += whitePiecePositionalValuesI16[node.Value.S2] - whitePiecePositionalValuesI16[node.Value.S1]
	}
	// Store the Value in the Transposition Table
	w.CommitToCache(posHash, w.Evaluation)
	// Store this node's increment
	node.Increment = lastEvaluation - w.Evaluation
	// Return the Restore value
	return &ret
}

// Unmake move will revert a move and decrement the Evaluation accordingly
func (w *Worker) UnmakeMove(snap *Snapshot) {
	w.Simulation = &snap.Position
	w.Evaluation = snap.Evaluation
}

func (w *Worker) ReadFromCache(hash [16]byte) (int16, error) {
	val, ok := w.Engine.SharedCache.Load(hash)
	if !ok {
		return 0, fmt.Errorf("cache acess failed")
	}
	ret, ok := val.(int16)
	if !ok {
		return 0, fmt.Errorf("type assertion failed")
	}
	return ret, nil
}

func (w *Worker) CommitToCache(hash [16]byte, eval int16) {
	w.Engine.SharedCache.Store(hash, eval)
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
	// Get the Standing Pat Score using Static Evaluation
	standingPat := w.Evaluation
	// if we are in a maximizing branch
	if max {
		// Check if the Standing Pat Causes Beta Cutof
		if standingPat >= beta {
			// return beta (fail hard)
			return beta
		}
		// Check if the standing pat is greater than alpha
		if standingPat > alpha {
			// raise alpha
			alpha = standingPat
		}
		// Generate the Unstable Children of the Node, make sure not to influence the sim
		unstable := node.GetUnstableLeaves(w, max, node.IsCheck())
		// Sort the Unstable nodes using MVVLVA to increase search speed
		sort.Sort(byMVVLVA{Nodes: unstable, Worker: w})
		// Iterate over the Unstable children of the node
		for _, child := range unstable {
			// Make sure not to leak a pointer to the Iterator
			alloc := *child
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
		if standingPat <= alpha {
			// fail soft
			return alpha
		}
		// Check if the standing pat lowers beta
		if standingPat < beta {
			// lower beta
			beta = standingPat
		}
		// Generate the Unstable Children of the Node
		unstable := node.GetUnstableLeaves(w, max, node.IsCheck())
		// Sort the Unstable nodes using MVVLVA to increase search speed
		sort.Sort(byMVVLVA{Nodes: unstable, Worker: w})
		// Iterate over the Unstable children of the node
		for _, child := range unstable {
			// Make sure not to leak a pointer to the Iterator
			alloc := *child
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
func (w *Worker) MinimaxPruning(node *Node, alpha int16, beta int16, depth int, max bool) (*Node, int16) {
	// Immediately Exit if our worker was stopped, returns will be ignored
	if w.Stop {
		return node, 0
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
		return node, node.EvaluateWithQuiescence(w, alpha, beta, depth, max)
	}
	// If we are in a maximizing Branch
	if max {
		best := MinScore
		bestNode := &Node{}
		// Generate the Children of the current node
		leaves := node.GenerateLeaves(w)
		// Sort them by their Expected impact on the Evaluation (Captures and Checks first)
		sort.Sort(byMVVLVA{Nodes: leaves, Worker: w})
		// Iterate over the Children of the Current Node
		for _, child := range leaves {
			// Prevent Leaking the Loop iterator
			alloc := *child
			// Simulate the Move of the current Child
			snapshot := w.MakeMove(&alloc)
			// Recursively Call Minimax for each child
			nod, ev := w.MinimaxPruning(&alloc, alpha, beta, depth-1, false)
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
		return bestNode, best
	}
	// If we are in a minimizing Branch
	worst := MaxScore
	worstNode := &Node{}
	// Generate the Children of the current node
	leaves := node.GenerateLeaves(w)
	// Sort them by their Expected impact on the Evaluation (Captures and Checks first)
	sort.Sort(byMVVLVA{Nodes: leaves, Worker: w})
	// Iterate over the Children of the Current Node
	for _, child := range leaves {
		// Prevent Leaking the Loop iterator
		alloc := *child
		// Simulate the Move of the current Child
		snapshot := w.MakeMove(&alloc)
		// Recursively Call Minimax for each child
		nod, ev := w.MinimaxPruning(child, alpha, beta, depth-1, true)
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
	return worstNode, worst
}

// EvaluateWithQuiescence will return the Evaluation of the Node, using QuiescenseSearch if applicable
func (n *Node) EvaluateWithQuiescence(w *Worker, alpha int16, beta int16, depth int, max bool) int16 {
	score := int16(0)
	// Check wether or not this node is stable
	unstableLeaves := n.GetUnstableLeaves(w, max, false)
	// if it is unstable begin Quiescence
	if len(unstableLeaves) > 0 {
		w.Engine.FrontierUnstable++
		score = w.QuiescenseSearch(n, alpha, beta, max)
		// if the node is stable perform Static Evaluation
	} else {
		// Otherwise Use the static evaluation
		score = w.Evaluation
	}
	return score
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
