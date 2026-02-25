#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  NEURAL NETWORK + MCTS SUDOKU SOLVER  — Built from scratch  ║
║  NN  : 729→2048→1024→1024→256→729  ReLU + grouped Softmax   ║
║  Loss: cross-entropy + constraint penalty (row/col/box)      ║
║  Curriculum: 50 clues → 42 → 35  (easy to hard over epochs) ║
║  MCTS: starvation-diet UCT — invoked only when stalled       ║
║  Guard: regression-proof supervisor protects best checkpoint ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np, random, time, os, pickle, math

# ── ANSI ──────────────────────────────────────────────────────────
RED="\033[91m";GRN="\033[92m";YLW="\033[93m";BLU="\033[94m"
MAG="\033[95m";CYN="\033[96m";WHT="\033[97m";BLD="\033[1m";DIM="\033[2m";RST="\033[0m"
FILL="█"; EMPTY="░"

WEIGHTS_PATH = "model_weights.pkl"
BEST_PATH    = "model_best.pkl"

# ── Architecture: 5,092,569 parameters ────────────────────────────
ARCH = [729, 2048, 1024, 1024, 256, 729]

def bar(val, w=28):
    f=int(w*val); pct=val*100
    col=RED if pct<40 else (YLW if pct<75 else GRN)
    return f"{col}{FILL*f}{EMPTY*(w-f)}{RST} {BLD}{pct:5.1f}%{RST}"

def header():
    tp=sum(ARCH[i]*ARCH[i+1]+ARCH[i+1] for i in range(len(ARCH)-1))
    print(f"\n{BLD}{CYN}{'='*64}{RST}")
    print(f"{BLD}{CYN}  🧠 NN + MCTS SUDOKU SOLVER{RST}")
    print(f"{BLD}{CYN}  Net : {' -> '.join(str(l) for l in ARCH)}  ({tp:,} params){RST}")
    print(f"{BLD}{CYN}  Loss: CE + constraint penalty (row/col/box){RST}")
    print(f"{BLD}{CYN}  Curr: 55→47→39→32→26 clues · 5 phases · aug×4{RST}")
    print(f"{BLD}{CYN}  MCTS: forced-guess · sharp rewards · deep rollouts{RST}")
    print(f"{BLD}{CYN}  Guard: regression-proof supervisor{RST}")
    print(f"{BLD}{CYN}{'='*64}{RST}\n")

# ══════════════════════════════════════════════════════════════════
#  CONSTRAINT ENGINE  (bitmask Sudoku)
# ══════════════════════════════════════════════════════════════════

FULL9 = 0x1FF

def _bit(d):       return 1 << (d - 1)
def _popcount(x):  return bin(x).count('1')
def _iter_bits(m):
    d = 1
    while m:
        if m & 1: yield d
        m >>= 1; d += 1

_PEERS = []
_UNITS = []

for _i in range(81):
    _r, _c = _i//9, _i%9
    _br, _bc = (_r//3)*3, (_c//3)*3
    _ps = set()
    for _j in range(9): _ps.add(_r*9+_j); _ps.add(_j*9+_c)
    for _dr in range(3):
        for _dc in range(3): _ps.add((_br+_dr)*9+(_bc+_dc))
    _ps.discard(_i)
    _PEERS.append(tuple(sorted(_ps)))

for _r in range(9):
    _UNITS.append(tuple(_r*9+_c for _c in range(9)))
for _c in range(9):
    _UNITS.append(tuple(_r*9+_c for _r in range(9)))
for _br in range(3):
    for _bc in range(3):
        _UNITS.append(tuple((_br*3+_dr)*9+(_bc*3+_dc) for _dr in range(3) for _dc in range(3)))

# Numpy unit arrays for vectorised constraint ops
_UNITS_NP = [np.array(u, dtype=np.int32) for u in _UNITS]


class SudokuState:
    __slots__ = ('board', 'cands', '_uns', 'nn_p')

    def __init__(self):
        self.board = [0]*81
        self.cands = [FULL9]*81
        self._uns  = 81
        self.nn_p  = None

    @classmethod
    def from_puzzle(cls, puzzle_9x9, nn_probs=None):
        s = cls()
        s.nn_p = nn_probs
        ok = True
        for i in range(81):
            v = int(puzzle_9x9[i//9][i%9])
            if v > 0:
                ok = s._assign(i, v) and ok
        if not ok: s._uns = -1
        return s

    def copy(self):
        s        = SudokuState.__new__(SudokuState)
        s.board  = self.board[:]
        s.cands  = self.cands[:]
        s._uns   = self._uns
        s.nn_p   = self.nn_p
        return s

    def _assign(self, cell, digit):
        if self.board[cell] != 0:
            return self.board[cell] == digit
        m = _bit(digit)
        if not (self.cands[cell] & m):
            return False
        self.board[cell] = digit
        self.cands[cell] = 0
        self._uns -= 1
        for p in _PEERS[cell]:
            if self.board[p] == 0:
                self.cands[p] &= ~m
                if self.cands[p] == 0:
                    return False
        return True

    def propagate(self):
        changed = True
        while changed:
            changed = False
            for i in range(81):
                if self.board[i] != 0: continue
                c = self.cands[i]
                if c == 0:             return False
                if c & (c-1) == 0:
                    if not self._assign(i, c.bit_length()): return False
                    changed = True
            for unit in _UNITS:
                for d in range(1, 10):
                    m = _bit(d); cnt = 0; last = -1
                    for ci in unit:
                        if self.board[ci] == d: cnt = -1; break
                        if self.board[ci] == 0 and (self.cands[ci] & m):
                            cnt += 1; last = ci
                    if   cnt == -1: continue
                    elif cnt ==  0: return False
                    elif cnt ==  1:
                        if not self._assign(last, d): return False
                        changed = True
        return True

    def is_solved(self): return self._uns == 0

    def remaining_candidates(self):
        return sum(_popcount(self.cands[i]) for i in range(81) if self.board[i]==0)

    def mrv_cell(self):
        best=-1; bmin=10
        for i in range(81):
            if self.board[i]!=0: continue
            c=_popcount(self.cands[i])
            if c==0: return -1,0
            if c<bmin: bmin=c; best=i
            if c==1: break
        return best, bmin

    def candidates_of(self, cell):
        return list(_iter_bits(self.cands[cell]))

    def lcv_order(self, cell, cands):
        nn = self.nn_p
        scored = []
        for d in cands:
            m = _bit(d)
            elim = sum(1 for p in _PEERS[cell]
                       if self.board[p]==0 and (self.cands[p] & m))
            nn_prob = float(nn[cell][d-1]) if nn is not None else 0.0
            scored.append((elim, -nn_prob, d))
        scored.sort()
        return [d for _,_,d in scored]


# ══════════════════════════════════════════════════════════════════
#  VALIDITY CHECK
# ══════════════════════════════════════════════════════════════════

def is_valid_solution(board_9x9, puzzle_9x9):
    b = board_9x9
    if np.any(b == 0):
        return False
    mask = puzzle_9x9 > 0
    if not np.all(b[mask] == puzzle_9x9[mask]):
        return False
    target = set(range(1, 10))
    for i in range(9):
        if set(b[i, :]) != target:   return False
        if set(b[:, i]) != target:   return False
    for br in range(3):
        for bc in range(3):
            if set(b[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten()) != target:
                return False
    return True


# ══════════════════════════════════════════════════════════════════
#  CONSTRAINT LOSS  (differentiable global consistency penalty)
# ══════════════════════════════════════════════════════════════════

def constraint_loss_and_grad(pred):
    """
    Penalise violations of Sudoku row / column / box constraints.

    For every unit (row, col, box) and every digit d:
        ideal: exactly one of the 9 cells in that unit holds digit d
        -> sum_{c in unit} P(cell=c, digit=d)  should equal 1.0

    Loss  = mean over all 27 units x 9 digits of (sum - 1)^2
    Grad  w.r.t. P(c,d)  = sum over units containing c of
                            2 * (sum_{c' in unit} P(c',d) - 1) / (27 * batch)

    pred  : (batch, 729)  -- softmax output from network
    return: (scalar_loss, grad same shape as pred)
    """
    batch   = pred.shape[0]
    p       = pred.reshape(batch, 81, 9)     # (B, 81, 9)
    grad    = np.zeros_like(p)
    total   = 0.0
    n_units = len(_UNITS_NP)
    norm    = float(batch * n_units)         # normalisation

    for unit in _UNITS_NP:
        unit_p    = p[:, unit, :]            # (B, 9, 9)  cells x digits
        digit_sum = unit_p.sum(axis=1)       # (B, 9)     sum over cells per digit
        residual  = digit_sum - 1.0          # (B, 9)     deviation from ideal
        total    += np.mean(residual ** 2)
        # Gradient: distribute 2*residual back to every cell in this unit
        g_unit    = 2.0 * residual / norm    # (B, 9)
        grad[:, unit, :] += g_unit[:, np.newaxis, :]

    return total / n_units, grad.reshape(batch, 729)


# ══════════════════════════════════════════════════════════════════
#  MCTS
# ══════════════════════════════════════════════════════════════════

MCTS_SIMS      = 48
MCTS_DEPTH_CAP = 5
ROLLOUT_DEPTH  = 4
DEEP_ROLLOUTS  = 2
UCT_C          = 0.5

_S_SOLVED = 1.0
_S_CONTRA = -1.0
_S_SCALE  = 81 * 9


def _score(state):
    if state.is_solved():  return _S_SOLVED
    rem = state.remaining_candidates()
    if rem == 0:           return _S_CONTRA
    return -rem / _S_SCALE


class _Node:
    __slots__ = ('digit','visits','total','prior')
    def __init__(self, digit, prior):
        self.digit  = digit
        self.visits = 1
        self.total  = prior
        self.prior  = prior


def _rollout_bounded(state):
    s = state.copy()
    if not s.propagate(): return _S_CONTRA
    if s.is_solved():     return _S_SOLVED
    for _ in range(ROLLOUT_DEPTH):
        cell, n = s.mrv_cell()
        if cell == -1:  return _score(s)
        if n == 0:      return _S_CONTRA
        cands = s.lcv_order(cell, s.candidates_of(cell))
        if not cands:   return _S_CONTRA
        if not s._assign(cell, cands[0]): return _S_CONTRA
        if not s.propagate():             return _S_CONTRA
        if s.is_solved():                 return _S_SOLVED
    return _score(s)


def _rollout_deep(state):
    s = state.copy()
    if not s.propagate(): return _S_CONTRA
    if s.is_solved():     return _S_SOLVED
    for _ in range(81):
        cell, n = s.mrv_cell()
        if cell == -1:  return _score(s)
        if n == 0:      return _S_CONTRA
        cands = s.lcv_order(cell, s.candidates_of(cell))
        if not cands:   return _S_CONTRA
        if not s._assign(cell, cands[0]): return _S_CONTRA
        if not s.propagate():             return _S_CONTRA
        if s.is_solved():                 return _S_SOLVED
    return _score(s)


def _forced_guess(state, cell, cands):
    if len(cands) != 2:
        return None
    a, b = cands[0], cands[1]
    trial = state.copy()
    if trial._assign(cell, a) and trial.propagate():
        return None
    return b


def mcts_decide(state, cell, cands):
    nn = state.nn_p
    n  = len(cands)
    nodes = []
    for d in cands:
        prior = float(nn[cell][d-1]) if nn is not None else (1.0/n)
        nodes.append(_Node(d, prior))
    s_prior = sum(nd.prior for nd in nodes) + 1e-9
    for nd in nodes: nd.prior /= s_prior; nd.total = nd.prior
    N = len(nodes)
    for sim_i in range(MCTS_SIMS):
        lnN  = math.log(N + 1)
        pick = max(nodes, key=lambda nd:
            (nd.total/nd.visits) + UCT_C*math.sqrt(lnN/nd.visits))
        s2 = state.copy()
        ok = s2._assign(cell, pick.digit) and s2.propagate()
        if not ok:
            score = _S_CONTRA
        elif s2.is_solved():
            score = _S_SOLVED
        else:
            score = _rollout_deep(s2) if sim_i < DEEP_ROLLOUTS else _rollout_bounded(s2)
        pick.visits += 1
        pick.total  += score
        N           += 1
    return max(nodes, key=lambda nd: nd.total/nd.visits).digit


def solve_mcts(state, depth=0):
    if not state.propagate(): return None
    if state.is_solved():     return state
    cell, n_cands = state.mrv_cell()
    if cell == -1:   return state if state.is_solved() else None
    if n_cands == 0: return None
    cands = state.lcv_order(cell, state.candidates_of(cell))
    if not cands: return None
    forced = _forced_guess(state, cell, cands)
    if forced is not None:
        cands = [forced]
    elif depth <= MCTS_DEPTH_CAP and n_cands >= 2:
        best_d = mcts_decide(state, cell, cands)
        cands  = [best_d] + [d for d in cands if d != best_d]
    for digit in cands:
        s2 = state.copy()
        if s2._assign(cell, digit):
            result = solve_mcts(s2, depth + 1)
            if result is not None:
                return result
    return None


def nn_guided_solve(puzzle_9x9, nn_probs_81x9):
    nn_probs_copy = np.array(nn_probs_81x9, dtype=np.float32)
    state = SudokuState.from_puzzle(puzzle_9x9, nn_probs_copy)
    if state._uns < 0:
        return None
    result = solve_mcts(state)
    if result is None or not result.is_solved():
        return None
    board = np.zeros((9, 9), dtype=int)
    for i in range(81):
        board[i//9][i%9] = result.board[i]
    return board


# ══════════════════════════════════════════════════════════════════
#  NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════

class NeuralNet:
    def __init__(self, layers):
        self.layers   = layers
        self.weights  = []
        self.biases   = []
        self.best_acc = 0.0
        for i in range(len(layers)-1):
            scale = np.sqrt(2.0/layers[i])
            self.weights.append(np.random.randn(layers[i],layers[i+1]).astype(np.float32)*scale)
            self.biases.append(np.zeros(layers[i+1], dtype=np.float32))
        self.mW=[np.zeros_like(w) for w in self.weights]
        self.vW=[np.zeros_like(w) for w in self.weights]
        self.mb=[np.zeros_like(b) for b in self.biases]
        self.vb=[np.zeros_like(b) for b in self.biases]
        self.t=0

    def relu(self,x):    return np.maximum(0,x)
    def relu_d(self,x):  return (x>0).astype(np.float32)

    def softmax_g(self,x):
        x=x.reshape(-1,81,9); x=x-x.max(axis=2,keepdims=True)
        ex=np.exp(x)
        return (ex/(ex.sum(axis=2,keepdims=True)+1e-8)).reshape(-1,729)

    def forward(self,X):
        self.cache=[X]; A=X
        for i,(W,b) in enumerate(zip(self.weights,self.biases)):
            Z=A@W+b
            A=self.relu(Z) if i<len(self.weights)-1 else self.softmax_g(Z)
            self.cache.append((Z,A))
        return A

    def loss(self, pred, tgt, constraint_alpha=0.0):
        """Cross-entropy + optional constraint penalty."""
        pr = pred.reshape(-1,81,9)
        tr = tgt.reshape(-1,81,9)
        ce = -np.sum(tr*np.log(pr+1e-8)) / (pred.shape[0]*81)
        if constraint_alpha > 0:
            cl, _ = constraint_loss_and_grad(pred)
            return ce + constraint_alpha * cl
        return ce

    def backward(self, X, tgt, lr, constraint_alpha=0.0):
        """
        Back-prop with optional constraint penalty.

        The constraint gradient is added directly to dA (the gradient
        w.r.t. the final softmax output) before propagating backwards.
        This is the correct insertion point because the constraint is
        defined purely on the output probabilities.

        constraint_alpha is ramped up by the caller over training.
        """
        batch = X.shape[0]
        pred  = self.cache[-1][1]
        gW    = [None]*len(self.weights)
        gb    = [None]*len(self.biases)

        # Cross-entropy gradient w.r.t. softmax output
        dA = (pred - tgt) / (batch * 81)

        # Add constraint penalty gradient at output layer
        if constraint_alpha > 0:
            _, c_grad = constraint_loss_and_grad(pred)
            dA = dA + constraint_alpha * c_grad

        for i in reversed(range(len(self.weights))):
            Ap = self.cache[i] if i==0 else self.cache[i][1]
            gW[i] = Ap.T @ dA
            gb[i] = dA.sum(axis=0)
            if i > 0:
                dA = (dA @ self.weights[i].T) * self.relu_d(self.cache[i][0])

        self.t += 1; b1,b2,eps = 0.9, 0.999, 1e-8
        for i in range(len(self.weights)):
            self.mW[i] = b1*self.mW[i] + (1-b1)*gW[i]
            self.vW[i] = b2*self.vW[i] + (1-b2)*gW[i]**2
            self.weights[i] -= lr*(self.mW[i]/(1-b1**self.t)) / \
                               (np.sqrt(self.vW[i]/(1-b2**self.t))+eps)
            self.mb[i] = b1*self.mb[i] + (1-b1)*gb[i]
            self.vb[i] = b2*self.vb[i] + (1-b2)*gb[i]**2
            self.biases[i] -= lr*(self.mb[i]/(1-b1**self.t)) / \
                              (np.sqrt(self.vb[i]/(1-b2**self.t))+eps)

    def _pack(self):
        return dict(layers=self.layers,
                    weights=[w.copy() for w in self.weights],
                    biases=[b.copy() for b in self.biases],
                    mW=self.mW, vW=self.vW, mb=self.mb, vb=self.vb,
                    t=self.t, best_acc=self.best_acc)

    def save(self,path):
        with open(path,"wb") as f: pickle.dump(self._pack(),f)
        kb=os.path.getsize(path)/1024
        print(f"  {GRN}💾 {BLD}{path}{RST}{GRN}  ({kb:.0f} KB)  acc={self.best_acc*100:.2f}%{RST}")

    @classmethod
    def load(cls,path):
        with open(path,"rb") as f: d=pickle.load(f)
        m=cls(d["layers"])
        m.weights=d["weights"]; m.biases=d["biases"]
        m.mW=d["mW"]; m.vW=d["vW"]; m.mb=d["mb"]; m.vb=d["vb"]
        m.t=d["t"]; m.best_acc=d.get("best_acc",0.0)
        return m

    def predict_probs(self, puzzle_9x9):
        x = encode(puzzle_9x9).reshape(1,-1)
        return self.forward(x)[0].reshape(81,9)

    def nn_cell_accuracy(self, X, Y_boards):
        pred = self.forward(X)
        pb   = np.array([decode(p) for p in pred])
        ca   = np.sum(pb==Y_boards) / Y_boards.size
        pa   = np.sum(np.all(pb==Y_boards, axis=(1,2))) / len(Y_boards)
        return ca, pa


# ══════════════════════════════════════════════════════════════════
#  SUPERVISOR
# ══════════════════════════════════════════════════════════════════

class Supervisor:
    def __init__(self,best_path=BEST_PATH):
        self.best_path=best_path; self.best=0.0
        if os.path.exists(best_path):
            try:
                with open(best_path,"rb") as f:
                    self.best=pickle.load(f).get("best_acc",0.0)
                print(f"  {CYN}🛡  previous best = {BLD}{self.best*100:.2f}%{RST}")
            except Exception: pass
        else:
            print(f"  {DIM}🛡  no prior best -- starting fresh{RST}")

    def evaluate(self,model,run_acc):
        print(f"\n  {BLD}{MAG}{'-'*52}{RST}")
        print(f"  {BLD}{MAG}  🛡  SUPERVISOR DECISION{RST}")
        print(f"  {BLD}{MAG}{'-'*52}{RST}")
        print(f"  This run  : {BLD}{run_acc*100:.2f}%{RST}")
        print(f"  Best ever : {BLD}{self.best*100:.2f}%{RST}")
        model.best_acc=run_acc; model.save(WEIGHTS_PATH)
        if run_acc>self.best:
            delta=run_acc-self.best; self.best=run_acc; model.save(self.best_path)
            print(f"  {GRN}{BLD}  NEW BEST +{delta*100:.2f}%  -> '{self.best_path}' updated{RST}")
        else:
            gap=self.best-run_acc
            print(f"  {YLW}{BLD}  REGRESSION BLOCKED (-{gap*100:.2f}% vs best){RST}")
            print(f"  {DIM}  '{self.best_path}' left untouched{RST}")
        print(f"  {BLD}{MAG}{'-'*52}{RST}\n")
        return self.best


# ══════════════════════════════════════════════════════════════════
#  DATA + ENCODE / DECODE
# ══════════════════════════════════════════════════════════════════

def generate_sudoku(clues=35):
    base=3; side=base*base
    def pattern(r,c): return (base*(r%base)+r//base+c)%side
    def sh(s): return random.sample(s,len(s))
    rB=range(base)
    rows=[g*base+r for g in sh(rB) for r in sh(rB)]
    cols=[g*base+c for g in sh(rB) for c in sh(rB)]
    nums=sh(range(1,side+1))
    board=np.array([[nums[pattern(r,c)] for c in cols] for r in rows])
    puzzle=board.copy(); cells=list(range(81)); random.shuffle(cells)
    for cell in cells[:81-clues]: puzzle[cell//9][cell%9]=0
    return puzzle, board

def augment_sudoku(puzzle, solution):
    """
    Apply a random valid Sudoku symmetry transform.
    Every transform in this set is guaranteed to produce a valid Sudoku,
    so the augmented (puzzle, solution) pair is always correct by construction.

    Transforms applied in sequence (all independent, all composable):

    1. Digit relabelling  — permute the 9 digit symbols (9! = 362880 variants)
       e.g. swap all 1s<->7s everywhere; constraints are invariant under relabelling.

    2. Row shuffle within bands  — each 3-row band's rows can be permuted freely
       (3! per band = 6, three bands = 216 variants per band combo).
       Row order within a band does not affect row/col/box validity.

    3. Column shuffle within stacks  — same logic applied to columns
       (216 variants).

    4. Band shuffle  — the three horizontal bands can be reordered (3! = 6).
       Box structure is preserved because entire 3-row bands move together.

    5. Stack shuffle  — same for vertical stacks (3! = 6).

    6. Transpose  — reflect across the main diagonal (50% chance).
       Rows become columns; all constraints still hold.

    Together these cover the full isomorphism group of Sudoku under relabelling,
    giving up to ~10^12 distinct-but-equivalent puzzles from a single board.
    In practice we sample one random transform per call, so the effective
    dataset multiplier equals the augment_factor passed to make_dataset().
    """
    s = solution.copy().astype(int)
    p = puzzle.copy().astype(int)

    # ── 1. Digit relabelling ──────────────────────────────────────
    perm = np.array(random.sample(range(1, 10), 9), dtype=int)  # perm[d-1] = new digit for d
    s_new = np.zeros_like(s); p_new = np.zeros_like(p)
    for orig in range(1, 10):
        s_new[s == orig] = perm[orig - 1]
        p_new[p == orig] = perm[orig - 1]
    s, p = s_new, p_new

    # ── 2. Row shuffle within each band ──────────────────────────
    for band in range(3):
        row_perm = random.sample(range(3), 3)
        block_s = s[band*3:(band+1)*3].copy()
        block_p = p[band*3:(band+1)*3].copy()
        s[band*3:(band+1)*3] = block_s[row_perm]
        p[band*3:(band+1)*3] = block_p[row_perm]

    # ── 3. Column shuffle within each stack ──────────────────────
    for stack in range(3):
        col_perm = random.sample(range(3), 3)
        block_s = s[:, stack*3:(stack+1)*3].copy()
        block_p = p[:, stack*3:(stack+1)*3].copy()
        s[:, stack*3:(stack+1)*3] = block_s[:, col_perm]
        p[:, stack*3:(stack+1)*3] = block_p[:, col_perm]

    # ── 4. Band shuffle ───────────────────────────────────────────
    band_perm = random.sample(range(3), 3)
    s = np.vstack([s[b*3:(b+1)*3] for b in band_perm])
    p = np.vstack([p[b*3:(b+1)*3] for b in band_perm])

    # ── 5. Stack shuffle ──────────────────────────────────────────
    stack_perm = random.sample(range(3), 3)
    s = np.hstack([s[:, b*3:(b+1)*3] for b in stack_perm])
    p = np.hstack([p[:, b*3:(b+1)*3] for b in stack_perm])

    # ── 6. Transpose (50%) ────────────────────────────────────────
    if random.random() < 0.5:
        s = s.T.copy()
        p = p.T.copy()

    return p, s


def encode(board):
    enc=np.zeros(729,dtype=np.float32)
    for i in range(9):
        for j in range(9):
            v=board[i,j]
            if v>0: enc[(i*9+j)*9+(v-1)]=1.0
    return enc

def decode(pred):
    pred=pred.reshape(81,9); board=np.zeros((9,9),dtype=int)
    for i in range(81): board[i//9][i%9]=np.argmax(pred[i])+1
    return board

def make_dataset(n, clues=35, augment_factor=1):
    """
    Generate n base puzzles then augment each one augment_factor times.
    Total samples = n * augment_factor.

    augment_factor=1  : no augmentation (base puzzles only).
    augment_factor=k  : each base puzzle produces k-1 extra augmented variants
                        for a total of n*k samples.

    Augmented variants are interspersed with originals so any prefix-shuffle
    in the training loop sees mixed diversity from the start.
    """
    total = n * augment_factor
    print(f"  {DIM}Generating {n} base × aug{augment_factor} "
          f"= {total} samples  ({clues} clues, {81-clues} blanks)...{RST}",
          end="", flush=True)
    X, Y, Yb, Pz = [], [], [], []
    for _ in range(n):
        p, s = generate_sudoku(clues)
        X.append(encode(p)); Y.append(encode(s)); Yb.append(s.copy()); Pz.append(p.copy())
        for _ in range(augment_factor - 1):
            pa, sa = augment_sudoku(p, s)
            X.append(encode(pa)); Y.append(encode(sa)); Yb.append(sa); Pz.append(pa)
    print(f"  {GRN}done{RST}")
    return (np.array(X, np.float32), np.array(Y, np.float32),
            np.array(Yb), np.array(Pz))


# ══════════════════════════════════════════════════════════════════
#  CURRICULUM PHASES
# ══════════════════════════════════════════════════════════════════

def make_curriculum_phases(n_base=1500, aug=4):
    """
    Five difficulty tiers with augmented datasets.

    n_base   : base puzzles per phase before augmentation.
    aug      : augmentation factor — each base puzzle becomes `aug` training examples.
    total    : n_base * aug samples per phase.

    Why 5 phases instead of 3?
      The gap between 50 clues (trivial) and 26 clues (near-expert) is too large
      to cross in one jump.  Fine-grained steps give the model smooth footholds.

    Why augmentation?
      Sudoku has a huge isomorphism group (digit relabelling × row/col/band/stack
      shuffles × transpose).  Each base puzzle can produce thousands of valid
      distinct variants.  With aug=4, 1500 base puzzles become 6000 training
      samples that the model cannot just memorise — they look structurally different
      while remaining drawn from the same difficulty distribution.

    Difficulty ladder:
      Phase 1 — EASY     55 clues (26 blanks)  basic digit placement
      Phase 2 — WARM     47 clues (34 blanks)  first real constraint interactions
      Phase 3 — MEDIUM   39 clues (42 blanks)  full constraint reasoning required
      Phase 4 — HARD     32 clues (49 blanks)  search almost always needed
      Phase 5 — EXPERT   26 clues (55 blanks)  near-minimum clue count, matches test
    """
    phases = []
    tier_clues = [55, 47, 39, 32, 26]
    tier_names = ["EASY", "WARM", "MEDIUM", "HARD", "EXPERT"]
    for clues, label in zip(tier_clues, tier_names):
        print(f"\n  Phase {label} ({clues} clues, {81-clues} blanks):")
        phases.append(make_dataset(n_base, clues, augment_factor=aug))
    return phases


# ══════════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════════

def print_side_by_side(puzzle, predicted, solution, clue_mask, method_tag=""):
    tag = f" [{method_tag}]" if method_tag else ""
    print(f"\n  {BLD}{CYN}{'='*76}{RST}")
    print(f"  {BLD}{YLW}  {'PUZZLE':^24}    {'NN+MCTS'+tag:^26}    {'SOLUTION':^24}{RST}")
    print(f"  {BLD}{CYN}{'='*76}{RST}")
    for i in range(9):
        if i%3==0 and i>0:
            print(f"  {DIM}{'-'*23}  {'-'*23}  {'-'*23}{RST}")
        rp=rn=rs=""
        for j in range(9):
            if j%3==0 and j>0:
                rp+=f"{DIM}|{RST}"; rn+=f"{DIM}|{RST}"; rs+=f"{DIM}|{RST}"
            pv=puzzle[i,j]; nv=predicted[i,j]; sv=solution[i,j]
            rp+=f" {CYN}{pv}{RST} " if pv>0 else f" {DIM}.{RST} "
            rn+=(f" {CYN}{nv}{RST} " if clue_mask[i,j]
                 else f" {GRN if nv==sv else RED}{nv}{RST} ")
            rs+=f" {WHT}{sv}{RST} "
        print(f"  {rp}    {rn}    {rs}")
    print(f"  {BLD}{CYN}{'='*76}{RST}")


# ══════════════════════════════════════════════════════════════════
#  TRAINING — single phase
# ══════════════════════════════════════════════════════════════════

def train_phase(model, X_tr, Y_tr, Yb_tr, X_val, Y_val, Yb_val,
                epochs, batch_size, lr,
                c_alpha_start=0.0, c_alpha_end=0.0,
                phase_label=""):
    """
    Single-phase training loop.

    c_alpha (constraint penalty weight) linearly interpolates from
    c_alpha_start to c_alpha_end over the phase's epochs.

    Why linear ramp? Adding a large penalty at epoch 1 distorts gradients
    before the network has learned any structure. The warmup lets CE
    establish a sensible weight landscape first, then constraint pressure
    breaks the locally-correct-but-globally-wrong plateau.
    """
    n = X_tr.shape[0]
    print(f"\n{BLD}{MAG}  TRAINING  {phase_label}{RST}  "
          f"{DIM}({n} puzzles . {epochs} ep . batch={batch_size} . "
          f"lr={lr:.4f} . alpha={c_alpha_start:.3f}->{c_alpha_end:.3f}){RST}\n")

    bst_acc = 0.0; bst_W = None; bst_b = None

    for ep in range(1, epochs+1):
        # Linearly ramp constraint alpha within this phase
        if epochs > 1:
            c_alpha = c_alpha_start + (c_alpha_end - c_alpha_start) * (ep-1) / (epochs-1)
        else:
            c_alpha = c_alpha_end

        idx = np.random.permutation(n); Xs, Ys = X_tr[idx], Y_tr[idx]
        eloss = 0.0
        for s in range(0, n, batch_size):
            xb = Xs[s:s+batch_size]; yb = Ys[s:s+batch_size]
            pred = model.forward(xb)
            eloss += model.loss(pred, yb, constraint_alpha=c_alpha)
            model.backward(xb, yb, lr, constraint_alpha=c_alpha)
        eloss /= max(n // batch_size, 1)

        if ep % 5 == 0 or ep == 1:
            ca, pa = model.nn_cell_accuracy(X_val, Yb_val)
            imp = ca > bst_acc
            if imp:
                bst_acc = ca
                bst_W = [w.copy() for w in model.weights]
                bst_b = [b.copy() for b in model.biases]
            b   = bar(ca, 26)
            st  = f"{YLW}*{RST}" if imp else " "
            rg  = f" {RED}v{RST}" if ca < bst_acc - 0.005 and ep > 5 else ""
            al  = f"{DIM}a={c_alpha:.3f}{RST}"
            print(f"  Ep {BLD}{ep:3d}{RST}/{epochs} {al}"
                  f"  Loss:{YLW}{eloss:7.4f}{RST}"
                  f"  Cell:{b}{st}"
                  f"  Solved:{GRN}{pa*100:5.1f}%{RST}{rg}")
        else:
            print(f"  Ep {ep:3d}/{epochs}  Loss:{DIM}{eloss:7.4f}{RST}", end="\r")

    if bst_W:
        model.weights = bst_W; model.biases = bst_b
        print(f"\n  {DIM}Restored in-phase best ({bst_acc*100:.2f}%){RST}")

    return bst_acc


# ══════════════════════════════════════════════════════════════════
#  TRAINING — curriculum orchestrator
# ══════════════════════════════════════════════════════════════════

def train_curriculum(model, curriculum_phases, X_val, Y_val, Yb_val,
                     batch_size=64):
    """
    Five-phase curriculum trainer.

    Phase 1 — EASY     (55 clues) 30 ep  lr=0.002   alpha 0.00->0.03
      Establish basic digit-placement on near-complete boards.
      Very light constraint pressure — just enough to prevent pure CE overfit.

    Phase 2 — WARM     (47 clues) 35 ep  lr=0.0015  alpha 0.03->0.07
      First real constraint interactions. LR reduced slightly.
      Alpha climbs to break early local minima.

    Phase 3 — MEDIUM   (39 clues) 40 ep  lr=0.001   alpha 0.07->0.12
      Majority of training budget. Full constraint pressure engaging.
      Largest phase by epoch count — model does most of its learning here.

    Phase 4 — HARD     (32 clues) 40 ep  lr=0.0005  alpha 0.12->0.17
      Near test difficulty. Fine-tuning with strong global consistency signal.
      LR quartered to prevent catastrophic forgetting of Phase 3 structure.

    Phase 5 — EXPERT   (26 clues) 30 ep  lr=0.0002  alpha 0.17->0.20
      Expert-difficulty; fewer clues than standard test set.
      Training on harder puzzles means test puzzles feel 'easy' to the model.
      Very small LR — polishing, not relearning.
    """
    configs = [
        # (phase_idx, epochs, lr,      alpha_start, alpha_end, label)
        (0, 30, 0.002,  0.00, 0.03, "Phase 1 -- EASY   (55 clues)"),
        (1, 35, 0.0015, 0.03, 0.07, "Phase 2 -- WARM   (47 clues)"),
        (2, 40, 0.001,  0.07, 0.12, "Phase 3 -- MEDIUM (39 clues)"),
        (3, 40, 0.0005, 0.12, 0.17, "Phase 4 -- HARD   (32 clues)"),
        (4, 30, 0.0002, 0.17, 0.20, "Phase 5 -- EXPERT (26 clues)"),
    ]
    best_overall = 0.0
    for idx, epochs, lr, ca_s, ca_e, label in configs:
        X_tr, Y_tr, Yb_tr, _ = curriculum_phases[idx]
        acc = train_phase(model, X_tr, Y_tr, Yb_tr,
                          X_val, Y_val, Yb_val,
                          epochs=epochs, batch_size=batch_size,
                          lr=lr, c_alpha_start=ca_s, c_alpha_end=ca_e,
                          phase_label=label)
        best_overall = max(best_overall, acc)
        print(f"  {GRN}Phase done -- best cell acc this phase: {acc*100:.2f}%{RST}")

    return best_overall


# ══════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate_and_show(model, X_te, Yb_te, Pz_te, n_show=3):
    """
    Unambiguous evaluation.

    All arrays entering results[] are independent .copy()s.
    is_valid_solution() is the single source of truth for 'solved'.
    The display loop reads the exact objects the metrics loop wrote.
    """
    nn_ca, nn_pa = model.nn_cell_accuracy(X_te, Yb_te)
    preds        = model.forward(X_te)          # (N, 729) float probs

    print(f"\n  {DIM}Running NN+MCTS solver on {len(Pz_te)} test puzzles...{RST}", flush=True)

    mcts_solved       = 0
    mcts_cell_correct = 0
    mcts_total_cells  = 0
    results           = []   # (puzzle_board, pred_board, solution_board, is_solved, method)

    t0 = time.time()
    for k in range(len(Pz_te)):
        puzzle_board   = Pz_te[k].copy()            # (9,9) int, owned
        solution_board = Yb_te[k].copy()            # (9,9) int, owned
        nn_prob        = preds[k].reshape(81,9).copy()   # (81,9) float, owned

        mcts_board = nn_guided_solve(puzzle_board, nn_prob)

        if mcts_board is not None:
            pred_board = mcts_board          # fresh array from nn_guided_solve
            method     = "MCTS"
            is_solved  = is_valid_solution(pred_board, puzzle_board)
            if is_solved:
                mcts_solved += 1
        else:
            pred_board = decode(preds[k]).copy()
            method     = "NN"
            is_solved  = is_valid_solution(pred_board, puzzle_board)

        mcts_cell_correct += int(np.sum(pred_board == solution_board))
        mcts_total_cells  += 81
        results.append((puzzle_board, pred_board, solution_board, is_solved, method))

    mcts_ca = mcts_cell_correct / mcts_total_cells
    mcts_pa = mcts_solved / len(Pz_te)
    elapsed = time.time() - t0

    print(f"\n{BLD}{CYN}{'='*64}{RST}")
    print(f"{BLD}{CYN}  EVALUATION RESULTS{RST}")
    print(f"{BLD}{CYN}{'='*64}{RST}")
    print(f"\n  {'':22}  {'Cell Acc':>12}  {'Puzzles Solved':>16}")
    print(f"  {'-'*54}")
    print(f"  {'NN alone':22}  {bar(nn_ca,20)}  {GRN}{nn_pa*100:6.1f}%{RST}")
    print(f"  {'NN + MCTS':22}  {bar(mcts_ca,20)}  {GRN}{mcts_pa*100:6.1f}%{RST}")
    print(f"\n  {DIM}MCTS solver ran {len(Pz_te)} puzzles in {elapsed:.1f}s  "
          f"({elapsed/len(Pz_te)*1000:.0f} ms/puzzle){RST}")

    lift_ca  = (mcts_ca - nn_ca) * 100
    lift_pa  = (mcts_pa - nn_pa) * 100
    lift_col = GRN if lift_ca >= 0 else RED
    print(f"\n  {BLD}Lift from MCTS:{RST}"
          f"  Cell Acc {lift_col}{lift_ca:+.2f}%{RST}"
          f"  |  Puzzles Solved {lift_col}{lift_pa:+.1f}%{RST}")

    print(f"\n{BLD}{CYN}{'='*64}{RST}")
    print(f"{BLD}{CYN}  SAMPLE PREDICTIONS  (green=correct  red=wrong){RST}")
    print(f"{BLD}{CYN}{'='*64}{RST}")

    for k in range(min(n_show, len(results))):
        # Unpack the exact arrays evaluated above -- no re-computation
        puzzle_board, pred_board, solution_board, is_solved, method = results[k]

        clue_mask       = (puzzle_board > 0)
        n_empty         = int(np.sum(~clue_mask))
        n_blank_correct = int(np.sum((pred_board == solution_board) & ~clue_mask))
        mc_col          = GRN if method == "MCTS" else YLW

        if is_solved:
            status = f"{GRN}{BLD}SOLVED (valid){RST}"
        else:
            pct    = n_blank_correct / max(n_empty, 1) * 100
            status = (f"{RED}Partial -- "
                      f"{n_blank_correct}/{n_empty} blanks correct ({pct:.0f}%){RST}")

        print(f"\n  {BLD}Puzzle #{k+1}  --  {status}  "
              f"[{mc_col}{BLD}{method}{RST}]  {DIM}({n_empty} blanks){RST}")
        print_side_by_side(puzzle_board, pred_board, solution_board, clue_mask, method)

    return nn_ca, nn_pa, mcts_ca, mcts_pa


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42); random.seed(42)
    header()

    # ── [1/5] Curriculum datasets ──────────────────────────────────
    print(f"{BLD}{BLU}  [1/5] Curriculum datasets{RST}")
    print(f"  {DIM}5 tiers · 1500 base puzzles × aug4 = 6000 samples each:{RST}")
    curriculum_phases = make_curriculum_phases(n_base=1500, aug=4)

    # Validation: 400 puzzles at 28 clues (harder than HARD, easier than EXPERT)
    # Test: 100 puzzles at 26 clues (matches EXPERT training difficulty)
    print(f"\n  {DIM}Validation (400 puzzles, 28 clues) and test (100 puzzles, 26 clues):{RST}")
    X_val, Y_val, Yb_val, _     = make_dataset(400, 28, augment_factor=1)
    X_te,  _,     Yb_te,  Pz_te = make_dataset(100, 26, augment_factor=1)

    # ── [2/5] Model ────────────────────────────────────────────────
    print(f"\n{BLD}{BLU}  [2/5] Model{RST}")
    if os.path.exists(WEIGHTS_PATH):
        print(f"  {YLW}Resuming from '{WEIGHTS_PATH}'{RST}")
        model   = NeuralNet.load(WEIGHTS_PATH); resumed = True
        tp      = sum(w.size+b.size for w,b in zip(model.weights,model.biases))
        print(f"  {GRN}Loaded  params={tp:,}  step={model.t}  "
              f"prev_acc={model.best_acc*100:.2f}%{RST}")
    else:
        model   = NeuralNet(ARCH); resumed = False
        tp      = sum(w.size+b.size for w,b in zip(model.weights,model.biases))
        print(f"  {GRN}Fresh model  params={tp:,}{RST}")
    print(f"  {DIM}Layers: {' -> '.join(str(l) for l in model.layers)}{RST}")

    # ── [3/5] Supervisor ───────────────────────────────────────────
    print(f"\n{BLD}{BLU}  [3/5] Supervisor{RST}")
    sv = Supervisor(BEST_PATH)

    # ── [4/5] Training ─────────────────────────────────────────────
    print(f"\n{BLD}{BLU}  [4/5] Curriculum training{RST}")
    if resumed:
        print(f"  {DIM}(fine-tuning -- Adam state preserved){RST}")
    t0      = time.time()
    run_acc = train_curriculum(model, curriculum_phases,
                               X_val, Y_val, Yb_val,
                               batch_size=64)
    print(f"\n  {GRN}Done in {time.time()-t0:.1f}s  run_acc={run_acc*100:.2f}%{RST}")

    atb = sv.evaluate(model, run_acc)

    # ── [5/5] Evaluation ───────────────────────────────────────────
    print(f"{BLD}{BLU}  [5/5] Evaluating best checkpoint{RST}")
    em = NeuralNet.load(BEST_PATH) if os.path.exists(BEST_PATH) else model
    nn_ca, nn_pa, mcts_ca, mcts_pa = evaluate_and_show(em, X_te, Yb_te, Pz_te, n_show=3)

    print(f"\n{BLD}{CYN}{'='*64}{RST}")
    print(f"{BLD}{GRN}  Complete{RST}")
    print(f"  Run best        : {BLD}{run_acc*100:.2f}%{RST}")
    print(f"  All-time best   : {BLD}{atb*100:.2f}%{RST}")
    print(f"  NN cell acc     : {BLD}{nn_ca*100:.2f}%{RST}")
    print(f"  NN puzzles      : {BLD}{nn_pa*100:.2f}%{RST}")
    print(f"  MCTS cell acc   : {BLD}{mcts_ca*100:.2f}%{RST}")
    print(f"  MCTS solved     : {BLD}{mcts_pa*100:.2f}%{RST}")
    print(f"{BLD}{CYN}{'='*64}{RST}\n")

if __name__=="__main__":
    main()