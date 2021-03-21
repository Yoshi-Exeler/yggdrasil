package engine

import (
	"fmt"
	"math"
	"sort"
	"time"

	chess "rxchess"
	opening "rxchess/opening"
)

/* Possible Optimizations
*  Multithreading = ~5x Performance Increase
*  SimulatePositions with different parents by only going back to the First Common Ancestor ~1.5x Performance Increase
*  Improve Hash table Housekeeping
*  LazySMP --> Search on many threads at the same time with a shared hashtable
*  Endgame Eval and Sequence Database
*  Iterative Deepening in combination with the Hashtable and Pruning
*  Caputre Heuristic Most Valueable Victim Least Valuable Attacker
*  Save Moves that Cause Pruning, Search after Caputres
* https://www.duo.uio.no/bitstream/handle/10852/53769/master.pdf?sequence=1&isAllowed=y
*
 */

// MaxScore is Bigger than the Maximum Score Reachable
const MaxScore = float32(3000)

// MinScore is Smaller than the Minimum Score Reachable
const MinScore = float32(-3000)

// Engine is the Minimax Engine
type Engine struct {
	SearchMode         uint8 // 1 = SyncFull, 2 = SyncIterative, 3 = AMP
	Depth              uint8
	ProcessingTime     time.Duration
	UseOpeningTheory   bool
	Game               *chess.Game
	ECO                *opening.BookECO
	Origin             chess.Position
	Simulation         *chess.Position
	Root               *Node
	Color              chess.Color
	EvaluationCache    map[[16]byte]float32
	GeneratedNodes     uint
	QGeneratedNodes    uint
	EvaluatedNodes     uint
	QEvaluatedNodes    uint
	Visited            uint
	QVisited           uint
	LoadedFromPosCache uint
	FrontierUnstable   uint
}

// Node is a node in the Engine
type Node struct {
	Parent          *Node
	Depth           uint8
	QDepth          uint8
	QNode           *Node
	Value           *chess.Move
	Leaves          []*Node
	LeavesGenerated bool
	StatusEval      float32
	StatusChecked   bool
}

// NewEngine returns a new Engine from the specified game
func NewEngine(g *chess.Game, clr chess.Color) *Engine {
	pos := *g.Position()
	return &Engine{UseOpeningTheory: true, Depth: 5, SearchMode: 1, ECO: opening.NewBookECO(), Game: g, Color: clr, Origin: *g.Position(), Simulation: &pos, Root: &Node{Value: nil, Depth: 0}, EvaluationCache: make(map[[16]byte]float32)}
}

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

// ByMoveVariance will Sort by Captures and Checks, which have the highest potential score variance
type byMoveVariance []*Node

func (a byMoveVariance) Len() int      { return len(a) }
func (a byMoveVariance) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a byMoveVariance) Less(i, j int) bool {
	if a[i].Value.HasTag(chess.Check) && !a[j].Value.HasTag(chess.Check) {
		return true
	}
	if a[i].Value.HasTag(chess.Capture) && !a[j].Value.HasTag(chess.Capture) {
		return true
	}
	return false
}

type byOpeningLength []*opening.Opening

func (a byOpeningLength) Len() int           { return len(a) }
func (a byOpeningLength) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byOpeningLength) Less(i, j int) bool { return len(a[i].PGN()) > len(a[j].PGN()) }

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
	var bestScore float32
	// Switch the SearchMode
	switch e.SearchMode {
	case 1:
		// TODO: optimize recast int()
		// SynchronousFull will search Synchronously until the specified depth is reached
		bestNode, bestScore = e.SearchSynchronousFull(int(e.Depth))
	case 2:
		// SynchronousIterative will search Synchronously using Iterative Deepening until the
		// specified processing time has elapsed
		bestNode, bestScore = e.SearchSynchronousIterative(e.ProcessingTime)
	case 3:
		// AMP will search using Asynchronous Iterative Deepening
		bestNode, bestScore = e.SearchAMP(e.ProcessingTime)
	}
	// Get the Origin Move of the Best Leaf, i.e. the move to play
	origin := bestNode.getSequence(e)[0]
	// Log some information about the search we just completed
	fmt.Printf("[YGG] Sequence:%v\nTarget:%v\nOrigin:%v D:%v QD:%v LD:%v\n", bestNode.getSequence(e), math.Round(float64(bestScore)*100)/100, origin, bestNode.Depth, bestNode.QDepth, u8Max(bestNode.Depth, bestNode.QDepth))
	fmt.Printf("[YGG] Generated:%v Visited:%v Evaluated:%v LoadedFromCache:%v\n", e.GeneratedNodes, e.Visited, e.EvaluatedNodes, e.LoadedFromPosCache)
	fmt.Printf("[YGG] QGenerated:%v QVisited:%v QEvaluated:%v FrontierUnstable:%v\n", e.QGeneratedNodes, e.QVisited, e.QEvaluatedNodes, e.FrontierUnstable)
	return origin
}

func u8Max(a uint8, b uint8) uint8 {
	if a > b {
		return a
	}
	return b
}

// SearchSynchronousFull will fully search the tree to the Specified Depth
func (e *Engine) SearchSynchronousFull(depth int) (*Node, float32) {
	return e.MinimaxPruning(e.Root, MinScore, MaxScore, depth, true)
}

// SearchSynchronousIterative
func (e *Engine) SearchSynchronousIterative(processingTime time.Duration) (*Node, float32) {
	return nil, 0
}

// SearchAMP will search the position using Asymmetric Multi Processing
func (e *Engine) SearchAMP(processingTime time.Duration) (*Node, float32) {
	return nil, 0
}

// QuiescenseSearch will Search the Specified Position with a limited SubSearch to mitigate the Horizon effect
// by extending the Search until a Stable(Quiet) Position can be statically evaluated or the Entire Branch can
// be failed by cutof (soft or hard fail). The Expected Branching factor in a Quiescence Search is around 7 in
// the midgame so each search should not be too expensive.
func (e *Engine) QuiescenseSearch(node *Node, alpha float32, beta float32, max bool) float32 {
	// Increment the Nodes Visited by QuiescenceSearch
	e.QVisited++
	// Check if the Current Node is a Terminating node
	nseval := e.NodeStatusScore(node, max)
	if nseval > MinScore {
		// if this node is a terminating node, return its evaluation
		return nseval
	}
	// Get the Standing Pat Score using Static Evaluation
	standingPat := e.EvaluateStaticPosition(e.Simulation)
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
		tmp := *e.Simulation
		unstable := node.GetUnstableLeaves(e, max)
		e.Simulation = &tmp
		// Iterate over the Unstable children of the node
		for _, child := range unstable {
			// Make sure not to leak a pointer to the Iterator
			alloc := *child
			// Snapshot the State of the Simulation
			snapshot := *e.Simulation
			// Simulate the Move of the current child
			e.Simulation = e.Simulation.Update(alloc.Value)
			// Call QuiescenseSearch recursively
			score := e.QuiescenseSearch(&alloc, alpha, beta, !max)
			// Restore the Snapshot, we cannot just invert the move and update the simulation beacuse of promotions
			e.Simulation = &snapshot
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
		tmp := *e.Simulation
		unstable := node.GetUnstableLeaves(e, max)
		e.Simulation = &tmp
		// Iterate over the Unstable children of the node
		for _, child := range unstable {
			// Make sure not to leak a pointer to the Iterator
			alloc := *child
			// Snapshot the State of the Simulation
			snapshot := *e.Simulation
			// Simulate the Move of the current child
			e.Simulation = e.Simulation.Update(alloc.Value)
			// Call Q recursively
			score := e.QuiescenseSearch(&alloc, alpha, beta, !max)
			// Restore the Snapshot, we cannot just invert the move and update the simulation beacuse of promotions
			e.Simulation = &snapshot
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
func (e *Engine) MinimaxPruning(node *Node, alpha float32, beta float32, depth int, max bool) (*Node, float32) {
	// Increment the Nodes visited by the Search
	e.Visited++
	// Check if the Current Node is a Terminating node i.e. Stalemate or Checkmate
	nseval := e.NodeStatusScore(node, max)
	if nseval > MinScore {
		// If this node is a Terminating Node return its evaluation
		return node, nseval
	}
	// If we are at or below the target depth and not in check, Evaluate this Node using Quiescence Search if neccessary
	if depth <= 0 && !node.IsCheck() {
		return node, node.EvaluateWithQuiescence(e, alpha, beta, depth, max)
	}
	// If we are in a maximizing Branch
	if max {
		best := MinScore
		bestNode := &Node{}
		// Generate the Children of the current node
		leaves := node.GetLeaves(e)
		// Sort them by their Expected impact on the Evaluation (Captures and Checks first)
		sort.Sort(byMoveVariance(leaves))
		// Iterate over the Children of the Current Node
		for _, child := range leaves {
			// Recursively Call Minimax for each child
			nod, ev := e.MinimaxPruning(child, alpha, beta, depth-1, !max)
			// if the Current Child is better than the Previous best, it becomes the new Best
			if ev > best {
				best = ev
				bestNode = nod
			}
			// Raise Alpha if the Current Child is Greater than Alpha
			alpha = f32max(alpha, ev)
			// Check if the Current Child Causes Pruning, in which case we can stop the Iteration immediately
			if beta <= alpha {
				break
			}
		}
		return bestNode, best
	}
	// If we are in a minimizing Branch
	worst := MaxScore
	worstNode := &Node{}
	// Generate the Children of the current node
	leaves := node.GetLeaves(e)
	// Sort them by their Expected impact on the Evaluation (Captures and Checks first)
	sort.Sort(byMoveVariance(leaves))
	// Iterate over the Children of the Current Node
	for _, child := range leaves {
		// Recursively Call Minimax for each child
		nod, ev := e.MinimaxPruning(child, alpha, beta, depth-1, !max)
		// if the Current Child is worse than the Previous worst, it becomes the new worst
		if ev < worst {
			worst = ev
			worstNode = nod
		}
		// Lower Beta if the Current Child is Lower than Alpha
		beta = f32min(beta, ev)
		// Check if the Current Child Causes Pruning, in which case we can stop the Iteration immediately
		if beta <= alpha {
			break
		}
	}
	return worstNode, worst
}

// EvaluateStaticPosition will evaluate a Position statically, if an evaluation exists in the Hashtable, it is loaded instead
func (e *Engine) EvaluateStaticPosition(pos *chess.Position) float32 {
	// Hash the Current Position
	posHash := pos.Hash()
	// Get the Value for this Hash
	storedV := e.EvaluationCache[posHash]
	// If we have a stored value for this Hash return it
	if storedV != 0 {
		e.LoadedFromPosCache++
		return storedV
	}
	score := float32(0)
	// Perform Static Evaluation
	score = EvaluatePosition(e.Simulation, e.Color)
	// Increment the Evaluations Performed in Quiescence Search
	e.QEvaluatedNodes++
	// Save this Evaluation to the Cache
	e.EvaluationCache[posHash] = score
	return score
}

// EvaluateWithQuiescence will return the Evaluation of the Node, using QuiescenseSearch if applicable
func (n *Node) EvaluateWithQuiescence(e *Engine, alpha float32, beta float32, depth int, max bool) float32 {
	// Simulate the State of this Node
	e.simulateToNode(n)
	// Hash the Current Position
	posHash := e.Simulation.Hash()
	// Get the Value for this Hash
	storedV := e.EvaluationCache[posHash]
	// If we have a stored value for this Hash return it
	if storedV != 0 {
		e.LoadedFromPosCache++
		return storedV
	}
	score := float32(0)
	// Check wether or not this node is stable
	tmp := *e.Simulation
	unstableLeaves := n.GetUnstableLeaves(e, max)
	e.Simulation = &tmp
	// if it is unstable begin Quiescence
	if len(unstableLeaves) > 0 {
		e.FrontierUnstable++
		score = e.QuiescenseSearch(n, alpha, beta, max)
		// if the node is stable perform Static Evaluation
	} else {
		e.EvaluatedNodes++
		score = EvaluatePosition(e.Simulation, e.Color)
	}
	// Save this Evaluation to the Cache
	e.EvaluationCache[posHash] = score
	return score
}

// NodeStatusScore Returns the Status evaluation of this node
func (e *Engine) NodeStatusScore(n *Node, inv bool) float32 {
	// if this is the root dont check
	if n.Value == nil {
		return MinScore
	}
	// if this has a status already calculated, return it
	if n.StatusChecked {
		return n.StatusEval
	}
	// Simulate to the Node
	e.simulateToNode(n)
	// Evaluate the Status of the node using the specified inversion
	n.StatusEval = statusEval(e.Simulation.Status(), inv)
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
func statusEval(s chess.Method, inv bool) float32 {
	if s == chess.Checkmate {
		if inv {
			return -1000.0
		}
		return 1000.0
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

// GetLeaves will return the Leaves of the node and generate them if they dont exist yet
func (n *Node) GetLeaves(e *Engine) []*Node {
	if n.LeavesGenerated {
		return n.Leaves
	}
	return n.GenerateLeaves(e)
}

// IsCheck returns if a node is a checking node
func (n *Node) IsCheck() bool {
	if n == nil || n.Value == nil {
		return false
	}
	return n.Value.HasTag(chess.Check)
}

// GetUnstableLeaves Returns the list of Immediate Recaptures that became availabe through the last move
func (n *Node) GetUnstableLeaves(e *Engine, inv bool) []*Node {
	// Return an empty list if the Node is a terminating node, this could be optimized by only calling this from a safe ctx
	nseval := e.NodeStatusScore(n, inv)
	if nseval > MinScore {
		return make([]*Node, 0)
	}
	leaves := n.GetLeaves(e)
	e.GeneratedNodes -= uint(len(leaves))
	unstable := []*Node{}
	for _, nd := range leaves {
		if nd.Value.HasTag(chess.Capture) || nd.Value.HasTag(chess.Check) {
			e.QGeneratedNodes++
			unstable = append(unstable, nd)
		}
	}
	n.Leaves = unstable
	n.LeavesGenerated = true
	return unstable
}

// GenerateLeaves will generate the leaves of the specified node
func (n *Node) GenerateLeaves(e *Engine) []*Node {
	// Get the Sequence of the Parent node we were called on
	e.simulateToNode(n)
	// Get the Valid Moves From this Position
	vmoves := e.Simulation.ValidMoves()
	// Initialize node collection
	nds := []*Node{}
	// Convert the Moves to Nodes and add them to the Collection
	for _, mov := range vmoves {
		e.GeneratedNodes++
		nds = append(nds, &Node{Parent: n, Value: mov, Depth: n.Depth + 1})
	}
	// Set the Leaves of MV
	n.Leaves = nds
	return nds
}

func (e *Engine) simulateToNode(n *Node) {
	// Reset the Simulation to the Origin
	e.resetSimulation()
	// If this is the Root node we stop now, nothing todo
	if n.Value == nil {
		return
	}
	// Get the Sequence Required to get to this node
	seq := n.getSequence(e)
	// Apply those move to get to the Current Node
	for _, ms := range seq {
		e.Simulation = e.Simulation.Update((ms))
	}
}

// resetSimulation will reset the Simulated game to the original state
func (e *Engine) resetSimulation() {
	npos := e.Origin
	e.Simulation = &npos
}

// getSequence will return the full sequence of moves required to get to this node including the move of this node
func (n *Node) getSequence(e *Engine) []*chess.Move {
	seq := make([]*chess.Move, 0)
	cnode := n
	for {
		// Add the Value of the Current Node to the Sequence
		seq = append(seq, cnode.Value)
		// Stop if there is no parent
		if cnode.Parent == nil || cnode.Parent == e.Root {
			break
		}
		// Go to the Parent
		cnode = cnode.Parent
	}
	rev := reverseSequence(seq)
	return rev
}

// EvalStatic will evaluate a board
func EvaluatePosition(pos *chess.Position, clr chess.Color) float32 {
	// Calculate the Status of this Position
	status := pos.Status()
	// Calculate the Score of this Position
	score := pos.Board().Evaluate(clr)
	// Check for Checkmate
	if status == chess.Checkmate {
		if pos.Turn() == clr {
			return -1000.0
		}
		return 1000.0
	}
	// Check for Stalemate
	if status == chess.Stalemate {
		return 0
	}
	return score
}

func f32min(a float32, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func f32max(a float32, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
