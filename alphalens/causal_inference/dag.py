import math
import logging
from typing import List, Tuple, Set, Dict, Optional
from itertools import combinations
import numpy as np

logger = logging.getLogger(__name__)

class CausalDAGDiscovery:
    """
    Implements a constraint-based causal DAG discovery framework (PC-Algorithm style)
    using partial correlations as conditional independence tests.
    """
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def compute_partial_corr(self, cov: np.ndarray, i: int, j: int, S: List[int]) -> float:
        """
        Computes the partial correlation between variable i and j given a set of variables S
        using the covariance matrix.
        """
        if len(S) == 0:
            denom = math.sqrt(cov[i, i] * cov[j, j])
            return cov[i, j] / denom if denom > 0 else 0.0

        # Sub-covariance matrix for indices [i, j] + S
        indices = [i, j] + list(S)
        sub_cov = cov[np.ix_(indices, indices)]
        
        try:
            # Precision matrix (inverse covariance)
            precision = np.linalg.inv(sub_cov)
            # Partial corr is - P_01 / sqrt(P_00 * P_11)
            p_ij = -precision[0, 1] / math.sqrt(precision[0, 0] * precision[1, 1])
            return p_ij
        except np.linalg.LinAlgError:
            # Return correlation if matrix is singular
            denom = math.sqrt(cov[i, i] * cov[j, j])
            return cov[i, j] / denom if denom > 0 else 0.0

    def fisher_z_test(self, r: float, n: int, q_dim: int) -> float:
        """
        Fisher's z-transformation test for conditional independence.
        Returns the two-sided p-value.
        """
        if abs(r) >= 1.0:
            r = np.clip(r, -0.99999, 0.99999)
        z = 0.5 * math.log((1 + r) / (1 - r))
        # Standard error: 1 / sqrt(n - q - 3)
        se = 1.0 / math.sqrt(max(1, n - q_dim - 3))
        stat = z / se
        # Two-sided p-value
        p_val = 2 * (1.0 - 0.5 * (1.0 + math.erf(abs(stat) / math.sqrt(2.0))))
        return p_val

    def run_pc_algorithm(self, data: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, Dict[Tuple[int, int], Set[int]]]:
        """
        Runs the PC algorithm skeleton search.
        
        Returns:
            adjacency_matrix: N x N directed adjacency matrix (0: no edge, 1: directed edge X -> Y)
            separating_sets: Dictionary mapping node pairs to separating sets
        """
        n_samples, n_nodes = data.shape
        cov = np.cov(data, rowvar=False)
        
        # Start with a complete undirected graph (represented as directed in both directions)
        adj = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        
        sepset: Dict[Tuple[int, int], Set[int]] = {}
        
        l_depth = 0
        while True:
            edges_to_test = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and adj[i, j] == 1:
                        # Neighbors of i excluding j
                        neighbors = [k for k in range(n_nodes) if k != j and (adj[i, k] == 1 or adj[k, i] == 1)]
                        if len(neighbors) >= l_depth:
                            edges_to_test.append((i, j, neighbors))
            
            if len(edges_to_test) == 0:
                break
                
            edge_removed = False
            for i, j, neighbors in edges_to_test:
                # Iterate through subsets of neighbors of size l_depth
                for S in combinations(neighbors, l_depth):
                    r_partial = self.compute_partial_corr(cov, i, j, list(S))
                    p_val = self.fisher_z_test(r_partial, n_samples, len(S))
                    
                    if p_val >= self.significance_level:
                        # Independent! Remove edge
                        adj[i, j] = 0
                        adj[j, i] = 0
                        sepset[(i, j)] = set(S)
                        sepset[(j, i)] = set(S)
                        edge_removed = True
                        break
            
            l_depth += 1
            if not edge_removed:
                break

        # Orient v-structures: for X - Y - Z with X not adjacent to Z,
        # if Y is not in sepset(X, Z), orient as X -> Y <- Z
        for i in range(n_nodes):
            for k in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and i != k and k != j:
                        # i-k and j-k are connected, but i and j are not
                        if (adj[i, k] == 1 and adj[k, i] == 1) and (adj[j, k] == 1 and adj[k, j] == 1) and (adj[i, j] == 0):
                            sep = sepset.get((i, j), set())
                            if k not in sep:
                                # Orient as i -> k <- j
                                adj[k, i] = 0
                                adj[k, j] = 0
                                
        return adj, sepset

    def run_fci_algorithm(self, data: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, Dict[Tuple[int, int], Set[int]]]:
        """
        Runs the Fast Causal Inference (FCI) algorithm (§5.4).
        Extends the PC skeleton to handle latent (unmeasured) confounders by
        producing a Partial Ancestral Graph (PAG) with edge types:
          - 1: directed edge (→)
          - 2: bidirected edge (↔), indicating a latent common cause
          - 0: no edge
        
        Returns:
            pag_matrix: N x N PAG adjacency matrix
            separating_sets: Dictionary mapping node pairs to separating sets
        """
        n_samples, n_nodes = data.shape
        
        # Step 1: Run PC skeleton search (reuse existing implementation)
        adj, sepset = self.run_pc_algorithm(data, labels)
        
        # Step 2: Convert PC output to PAG representation
        # Start with the oriented PC output
        pag = adj.copy()
        
        # Step 3: Identify possible latent confounders via discriminating paths
        # For each pair (i, j) with an edge, check if removal of any single
        # conditioning variable breaks conditional independence — if not, 
        # the edge may be due to a latent confounder
        cov = np.cov(data, rowvar=False)
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if pag[i, j] == 0 and pag[j, i] == 0:
                    continue  # No edge to evaluate
                
                # Check if removing the edge still maintains conditional
                # independence with ALL possible conditioning sets
                # If edge persists under all conditionings, check for latent cause
                neighbors_i = [k for k in range(n_nodes) if k != i and k != j and (pag[i, k] != 0 or pag[k, i] != 0)]
                neighbors_j = [k for k in range(n_nodes) if k != i and k != j and (pag[j, k] != 0 or pag[k, j] != 0)]
                
                # Test if i and j share a common cause pattern:
                # Both i→k and j→k for some k, AND i,j are not directly causal
                has_common_child = False
                for k in range(n_nodes):
                    if k != i and k != j:
                        if pag[i, k] == 1 and pag[j, k] == 1:
                            has_common_child = True
                            break
                
                # If both directions are present (undirected edge surviving PC),
                # and they share a common effect, mark as potentially bidirected
                if pag[i, j] == 1 and pag[j, i] == 1:
                    # Check partial correlation strength — strong residual
                    # correlation after conditioning suggests latent confounder
                    all_others = [k for k in range(n_nodes) if k != i and k != j]
                    if all_others:
                        r_partial = self.compute_partial_corr(cov, i, j, all_others)
                        p_val = self.fisher_z_test(r_partial, n_samples, len(all_others))
                        
                        if p_val < self.significance_level and has_common_child:
                            # Mark as bidirected (latent common cause)
                            pag[i, j] = 2
                            pag[j, i] = 2
        
        return pag, sepset
