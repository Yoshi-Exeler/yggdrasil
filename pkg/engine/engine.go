package engine

import (
	"fmt"
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

var TxReset = uint(0)
var CnReset = 0
var TxGetSeq = uint(0)
var CnGetSeq = 0
var TxSTN = uint(0)
var CnSTN = 0
var TxEval = uint(0)
var CnEval = 0

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
	GeneratedNodes     uint
	EvaluatedNodes     uint
	Visited            uint
	LoadedFromPosCache uint
	FrontierUnstable   uint
	QEvaluatedNodes    uint
	CurrentBestNode    *Node
	EvaluationCache    map[[16]byte]float32
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
	return &Engine{UseOpeningTheory: false, Depth: 1, SearchMode: 1, ECO: opening.NewBookECO(), Game: g, Color: clr, Origin: *g.Position(), Simulation: &pos, Root: &Node{Value: nil, Depth: 0}, EvaluationCache: make(map[[16]byte]float32)}
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

// ResetStats will reset the Statistics of the Engine
func (e *Engine) ResetStats() {
	e.Visited = 0
	e.GeneratedNodes = 0
	e.EvaluatedNodes = 0
	e.LoadedFromPosCache = 0
}

// Search will Search the Tree using Minimax
func (e *Engine) Search() *chess.Move {
	// Check that we can actually play a move here
	if statusIsEnd(e.Origin.Status()) {
		fmt.Println("[YGG] Aborting search, no possible moves exist")
		return nil
	}
	if e.UseOpeningTheory {
		// Check the ECO for an Opening Move
		move := e.GetOpeningMove()
		if move != nil {
			return move
		}
		e.UseOpeningTheory = false
	}
	var bestNode *Node
	var bestScore float32
	switch e.SearchMode {
	case 1:
		bestNode, bestScore = e.SearchSynchronousFull(int(e.Depth))
	case 2:
		bestNode, bestScore = e.SearchSynchronousIterative(e.ProcessingTime)
	case 3:
		bestNode, bestScore = e.SearchAMP(e.ProcessingTime)
	}
	// Get the Origin Move of the Best Leaf
	origin := bestNode.getSequence(e)[0]
	if bestNode.QNode != nil {
		qorigin := bestNode.QNode.getSequence(e)
		fmt.Println("[QSEQ:]", qorigin)
		e.simulateToNode(bestNode.QNode)
		fmt.Println(e.Simulation.Board().Draw())
	}

	fmt.Printf("[YGG] Sequence:%v\nTarget:%v\nOrigin:%v D:%v QD:%v LD:%v\n", bestNode.getSequence(e), bestScore, origin, bestNode.Depth, bestNode.QDepth, u8Max(bestNode.Depth, bestNode.QDepth))
	fmt.Printf("[YGG] Generated:%v Visited:%v Evaluated:%v LoadedFromCache:%v\n", e.GeneratedNodes, e.Visited, e.EvaluatedNodes, e.LoadedFromPosCache)
	fmt.Printf("[YGG] FrontierUnstable:%v QuiescEvals:%v\n", e.FrontierUnstable, e.QEvaluatedNodes)
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

func (n *Node) SubSearch(e *Engine, processingTime time.Duration) (*Node, float32) {
	return nil, 0
}

// SearchAMP will search the position using Asymmetric Multi Processing
func (e *Engine) SearchAMP(processingTime time.Duration) (*Node, float32) {
	firstLayer := e.Root.GenerateLeaves(e)
	for _, nd := range firstLayer {
		go nd.SubSearch(e, processingTime)
	}
	return nil, 0
}

// GetColor offers the Option to invert colors
func (e *Engine) GetColor(inv bool) chess.Color {
	if inv {
		if e.Color == chess.Black {
			return chess.White
		}
		return chess.Black
	}
	return e.Color
}

// GetInverseMove returns the Inverse of the Value of the Node
func (n *Node) GetInverseMove() *chess.Move {
	return &chess.Move{S1: n.Value.S2, S2: n.Value.S1}
}

func (e *Engine) Q(node *Node, alpha float32, beta float32, max bool) float32 {
	// Increment the Processed Nodes
	e.QEvaluatedNodes++
	// Check for Termination
	nseval := e.NodeStatusScore(node, max)
	if nseval > MinScore {
		return nseval
	}
	// Get the Standing Pat
	standingPat := e.EvaluateStaticPosition(e.Simulation)
	// if we are in a max branch
	if max {
		// Check if the Standing Pat Causes Beta Cuttof
		if standingPat >= beta {
			fmt.Printf("SP(%v)>=BETA(%v)    [%v]\n", standingPat, beta, node.Value)
			fmt.Println(e.Simulation.Board().Draw())
			return beta
		}
		// Check if the standing pat raises alpha
		if standingPat > alpha {
			fmt.Printf("SP(%v)>ALPHA(%v)    [%v]\n", standingPat, alpha, node.Value)
			fmt.Println(e.Simulation.Board().Draw())
			alpha = standingPat
		}
		// Generate the Unstable Children of the Node, make sure not to influence the sim
		tmp := *e.Simulation
		unstable := node.GetUnstableLeaves(e, max)
		e.Simulation = &tmp
		// Iterate over the Unstable children of the node
		for _, child := range unstable {
			// Dealloc the Iterator
			alloc := *child
			// Simulate the Move of the current child
			fmt.Println("PRE_UPDATE", alloc)
			fmt.Println("Board\n", e.Simulation.Board().Draw())
			e.Simulation = e.Simulation.Update(alloc.Value)
			// Call Q recursively
			score := e.Q(&alloc, alpha, beta, !max)
			// Unsimulate the Move
			e.Simulation = e.Simulation.Update(alloc.GetInverseMove())
			// Check if the move causes beta cuttof
			if score >= beta {
				//fmt.Printf("SC(%v)>=BETA(%v)    [%v]\n", standingPat, beta, alloc.Value)
				//fmt.Println(e.Simulation.Board().Draw())
				return beta
			}
			// Check if the Score raises alpha
			if score > alpha {
				//fmt.Printf("SC(%v)>=ALPHA(%v)    [%v]\n", standingPat, alpha, alloc.Value)
				//fmt.Println(e.Simulation.Board().Draw())
				alpha = score
			}
		}
		// if we are in a min branch
	} else {
		// Check if the Standing Pat Causes Beta Cuttof
		if standingPat <= alpha {
			//fmt.Printf("SP(%v)<=ALPHA(%v)    [%v]\n", standingPat, alpha, node.Value)
			//fmt.Println(e.Simulation.Board().Draw())
			return alpha
		}
		// Check if the standing pat raises alpha
		if standingPat < beta {
			//fmt.Printf("SP(%v)<BETA(%v)    [%v]\n", standingPat, beta, node.Value)
			//fmt.Println(e.Simulation.Board().Draw())
			beta = standingPat
		}
		// Generate the Unstable Children of the Node
		tmp := *e.Simulation
		unstable := node.GetUnstableLeaves(e, max)
		e.Simulation = &tmp
		// Iterate over the Unstable children of the node
		for _, child := range unstable {
			// Dealloc the Iterator
			alloc := *child
			//fmt.Println("PRE_UPDATE", alloc)
			//fmt.Println("Board\n", e.Simulation.Board().Draw())
			// Simulate the Move of the current child
			e.Simulation = e.Simulation.Update(alloc.Value)
			// Call Q recursively
			score := e.Q(&alloc, alpha, beta, !max)
			// Unsimulate the Move
			e.Simulation = e.Simulation.Update(alloc.GetInverseMove())
			// Check if the move causes beta cuttof
			if score <= alpha {
				//fmt.Printf("SC(%v)<=ALPHA(%v)    [%v]\n", standingPat, alpha, alloc.Value)
				//fmt.Println(e.Simulation.Board().Draw())
				return alpha
			}
			// Check if the Score raises alpha
			if score < beta {
				//fmt.Printf("SC(%v)<BETA(%v)    [%v]\n", standingPat, beta, alloc.Value)
				//fmt.Println(e.Simulation.Board().Draw())
				beta = score
			}
		}
	}
	if max {
		//fmt.Printf("Return ALPHA(%v)    [%v]\n", alpha, node.Value)
		//fmt.Println(e.Simulation.Board().Draw())
		return alpha
	} else {
		//fmt.Printf("Return BETA(%v)    [%v]\n", alpha, node.Value)
		//fmt.Println(e.Simulation.Board().Draw())
		return beta
	}

}

// MinimaxPruning will traverse the Tree using the Minimax Algorithm with Alpa/Beta Pruning
func (e *Engine) MinimaxPruning(node *Node, alpha float32, beta float32, depth int, max bool) (*Node, float32) {
	e.Visited++
	// Test for Ceckmate and Stalemate
	nseval := e.NodeStatusScore(node, max)
	if nseval > MinScore {
		return node, nseval
	}
	// If we are at or below depth 0 and not in check, Evaluate this Node
	if depth <= 0 && !node.IsCheck() {
		// Perform Quiescent Evaluation
		return node, node.Quiescence(e, alpha, beta, depth, max)
	}
	if max {
		best := MinScore
		bestNode := &Node{}
		leaves := node.GetLeaves(e)
		sort.Sort(byMoveVariance(leaves))
		for _, child := range leaves {
			nod, ev := e.MinimaxPruning(child, alpha, beta, depth-1, !max)
			if ev > best {
				best = ev
				bestNode = nod
			}
			alpha = f32max(alpha, ev)
			if beta <= alpha {
				break
			}
		}
		return bestNode, best
	}
	worst := MaxScore
	worstNode := &Node{}
	leaves := node.GetLeaves(e)
	sort.Sort(byMoveVariance(leaves))
	for _, child := range leaves {
		nod, ev := e.MinimaxPruning(child, alpha, beta, depth-1, !max)
		if ev < worst {
			worst = ev
			worstNode = nod
		}
		beta = f32min(beta, ev)
	}
	return worstNode, worst
}

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
	// Save this Evaluation to the Cache
	e.EvaluatedNodes++
	e.EvaluationCache[posHash] = score
	return score
}

func (n *Node) EvaluateStatic(e *Engine) float32 {
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
	// Perform Static Evaluation
	score = EvaluatePosition(e.Simulation, e.Color)
	// Save this Evaluation to the Cache
	e.EvaluatedNodes++
	e.EvaluationCache[posHash] = score
	return score
}

// Evaluate will return the Evaluation of the Node, using QuiescenseSearch if applicable
func (n *Node) Quiescence(e *Engine, alpha float32, beta float32, depth int, max bool) float32 {
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
		fmt.Println("BEGIN_Q_", n.Value)
		fmt.Println(e.Simulation.Board().Draw())
		score = e.Q(n, alpha, beta, max)
		// if the node is stable perform Static Evaluation
	} else {
		score = EvaluatePosition(e.Simulation, e.Color)
	}

	// Save this Evaluation to the Cache
	e.EvaluatedNodes++
	e.EvaluationCache[posHash] = score
	return score
}

// NodeStatusScore Returns the Status evaluation of this node
func (e *Engine) NodeStatusScore(n *Node, inv bool) float32 {
	// if this is the root dont check
	if n.Value == nil {
		return MinScore
	}
	// if this has a status
	if n.StatusChecked {
		return n.StatusEval
	}
	e.resetSimulation()
	e.simulateToNode(n)
	n.StatusEval = statusEval(e.Simulation.Status(), inv)
	n.StatusChecked = true
	return n.StatusEval
}

// statusIsEnd will return wether or not the specified method ends the game
func statusIsEnd(s chess.Method) bool {
	return (s == chess.Checkmate || s == chess.Stalemate)
}

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
	unstable := []*Node{}
	for _, nd := range leaves {
		if nd.Value.HasTag(chess.Capture) || nd.Value.HasTag(chess.Check) {
			unstable = append(unstable, nd)
		}
	}
	n.Leaves = unstable
	n.LeavesGenerated = true
	return unstable
}

// GenerateLeaves will generate the leaves of the specified node
func (n *Node) GenerateLeaves(e *Engine) []*Node {
	// Reset the Simulation
	e.resetSimulation()
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
	start := time.Now()
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
	end := time.Now()
	TxSTN += uint(end.Sub(start))
	CnSTN++
}

// resetSimulation will reset the Simulated game to the original state
func (e *Engine) resetSimulation() {
	start := time.Now()
	npos := e.Origin
	e.Simulation = &npos
	end := time.Now()
	TxReset += uint(end.Sub(start))
	CnReset++
}

// getSequence will return the full sequence of moves required to get to this node including the move of this node
func (n *Node) getSequence(e *Engine) []*chess.Move {
	start := time.Now()
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
	end := time.Now()
	TxGetSeq += uint(end.Sub(start))
	CnGetSeq++
	return rev
}

// EvalStatic will evaluate a board
func EvaluatePosition(pos *chess.Position, clr chess.Color) float32 {
	start := time.Now()
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
	end := time.Now()
	TxEval += uint(end.Sub(start))
	CnEval++
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
