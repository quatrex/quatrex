# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
from __future__ import annotations
from typing import Any
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp

class RealSpaceOperator():
    """Data class for real-space 2-index operators.    

    - Sparse matrix form (COO)
    - Functional form (e.g. Coulomb interaction)
    - block sectioning

    """
    def __init__(self, positions:NDArray, cutoff_distance: float, strategy:str='sphere', dtype = float) -> None:
        """Initializes the real-space operator.
        
        Parameters
        ----------
        
        dimension : int
            The dimension of the operator.

        positions : NDArray
            The positions of the sites.
            
        cutoff_distance : float
            The cutoff distance for the interaction.
        """
        self.dimension = 2
        self.cutoff_distance = cutoff_distance
        self.positions = positions     
        self.dtype = dtype   
        self.num_sites = len(positions)
        coo = _compute_sparsity_pattern(positions, cutoff_distance,strategy=strategy)
        self.bandwidth = max(abs(coo.col - coo.row))         
        self.sparsity = coo
        self.matrix = None
        self.functional = None
        self.blocksizes = None
        self.block_offsets = None
        self.block_sparsity_map = None


    def set_matrix(self,matrix:NDArray) -> None:
        """Sets the matrix form of the operator by passing in a data vector.
        
        Parameters
        ----------
        
        matrix : NDArray
            The matrix form of the operator.
        
        """
        if (self.matrix is None) or (len(self.matrix) < len(matrix)):
            self.matrix = xp.zeros(self.sparsity.row.shape, dtype=self.dtype) 
        self.matrix[:len(matrix)] = matrix


    def set_functional(self, functional:function):
        """Sets the functional form of the operator by passing in a function.
        
        Parameters  
        ----------
        functional : function
            The functional form of the operator, takes as input two or more positions of the site.

        """
        self.functional = functional


    def get_dense(self) -> NDArray:
        """Returns the dense form of the operator.

        Returns
        -------
        NDArray
            The dense form of the operator.

        """
        mat = xp.zeros(tuple([self.num_sites]*self.dimension), dtype=self.dtype)
        if self.matrix is not None:
            mat[self.sparsity.row, self.sparsity.col] = self.matrix          
            return mat
        elif self.functional is not None:            
            for i in range(len(self.sparsity.row)):
                mat[self.sparsity.row[i], self.sparsity.col[i]] = self.functional(self.positions[self.sparsity.row[i]], self.positions[self.sparsity.col[i]])             
                    
            return mat
        else:
            raise ValueError("No matrix or functional form set.")
        

    def get_matrix_data(self) -> NDArray: 
        """Returns the data of the operator.

        Returns
        -------
        NDArray
            The data of the operator.

        """
        if self.matrix is not None:
            return self.matrix
        elif self.functional is not None:
            return xp.array([self.functional(self.positions[self.sparsity.row[i]], self.positions[self.sparsity.col[i]]) for i in range(len(self.sparsity.row))], dtype=self.dtype)
        else:
            raise ValueError("No matrix or functional form set.")
        

    def get_blockelement_index(self, block: tuple) -> NDArray:
        """Returns the indices of the block elements.

        Parameters
        ----------
        block : NDArray
            The block index.

        Returns
        -------
        NDArray
            The indices of the block elements.

        """
        if self.block_sparsity_map is None:
            raise ValueError("No block structure imposed.")
        return xp.where((self.block_sparsity_map[:, 0] == block[0]) & (self.block_sparsity_map[:, 1] == block[1]))[0]
    

    def get_dense_block(self, block: tuple) -> NDArray:
        """Returns the dense form of a block.

        Parameters
        ----------
        block : NDArray
            The block index.

        Returns
        -------
        NDArray
            The dense form of the block.

        """
        indices = self.get_blockelement_index(block)
        mat = xp.zeros(tuple([self.blocksizes[i] for i in block]), dtype=self.dtype)
        if self.matrix is not None:
            mat[self.sparsity.row[indices] - self.block_offsets[block[0]], self.sparsity.col[indices] - self.block_offsets[block[1]]] = self.matrix[indices]
        elif self.functional is not None:
            for i in range(len(indices)):
                mat[self.sparsity.row[indices[i]] - self.block_offsets[block[0]], self.sparsity.col[indices[i]] - self.block_offsets[block[1]]] = self.functional(self.positions[self.sparsity.row[indices[i]]], self.positions[self.sparsity.col[indices[i]]])
        else:
            raise ValueError("No matrix or functional form set.")
        return mat
    

    def set_dense_block(self, block: tuple, matrix_block:NDArray) -> None:
        """Returns the dense form of a block.

        Parameters
        ----------
        block : NDArray
            The block index.

        matrix_block : NDArray
            The dense form of the block.        

        """
        if self.matrix is not None:
            indices = self.get_blockelement_index(block)     
            self.matrix[indices] = matrix_block[self.sparsity.row[indices] - self.block_offsets[block[0]], self.sparsity.col[indices] - self.block_offsets[block[1]]]       
        else:
            raise ValueError("No matrix form set.")
        

    def get_coo_block(self, block: tuple) -> sparse.coo_matrix:
        """Returns the COO form of a block.

        Parameters
        ----------
        block : NDArray
            The block index.

        Returns
        -------
        sparse.coo_matrix
            The COO form of the block.

        """
        indices = self.get_blockelement_index(block)        
        rows = self.sparsity.row[indices] - self.block_offsets[block[0]]
        cols = self.sparsity.col[indices] - self.block_offsets[block[1]]
        data = self.matrix[indices] if self.matrix is not None else xp.array([self.functional(self.positions[self.sparsity.row[i]], self.positions[self.sparsity.col[i]]) for i in indices], dtype=self.dtype)        
        return sparse.coo_matrix((data, (rows, cols)), shape=tuple([self.blocksizes[i] for i in block]))
        
    

    def impose_block_structure(self, blocksizes: NDArray) -> None:
        """Imposes a block structure on the operator.

        Parameters
        ----------
        blocksizes : NDArray
            The block sizes.

        """                
        if blocksizes.sum() != self.num_sites:
            raise ValueError("Block sizes do not sum to number of sites.")
        self.blocksizes = xp.asarray(blocksizes, dtype=int)
        self.block_offsets = xp.hstack(([0], xp.cumsum(blocksizes)))
        self.block_sparsity_map = _compute_block_map(self.sparsity, blocksizes)
        

    def update_sparsity(self,cutoff_distance: float):
        """Updates the sparsity pattern of the operator.

        Parameters
        ----------
        cutoff_distance: float
            The new cutoff distance for the interaction.

        """
        new_sparsity = _compute_sparsity_pattern(self.positions, cutoff_distance)
        # if matrix form, then update the sparsity pattern of matrix
        if (self.matrix is not None) and (self.sparsity is not None):
            old_coo = self.get_coo()
            new_matrix_lil = sparse.lil_matrix((self.num_sites, self.num_sites), dtype=self.dtype)
            new_matrix_lil[old_coo.row, old_coo.col] = old_coo.data                        
            self.matrix = new_matrix_lil[new_sparsity.row, new_sparsity.col].toarray().flatten()

        self.sparsity = new_sparsity
        self.bandwidth = max(abs(new_sparsity.col - new_sparsity.row))       
        if self.blocksizes is not None:
            self.block_sparsity_map = _compute_block_map(self.sparsity, self.blocksizes)


    def get_coo(self) -> sparse.coo_matrix:
        """Returns the COO form of the operator.

        Returns
        -------
        sparse.coo_matrix
            The COO form of the operator.

        """
        if self.matrix is not None:
            return sparse.coo_matrix((self.matrix, (self.sparsity.row, self.sparsity.col)), shape=(self.num_sites, self.num_sites))
        elif self.functional is not None:
            return sparse.coo_matrix((self.get_matrix_data(), (self.sparsity.row, self.sparsity.col)), shape=(self.num_sites, self.num_sites))
        else:
            raise ValueError("No matrix or functional form set.")
        

    def get_csr(self) -> sparse.csr_matrix:
        """Returns the CSR form of the operator.

        Returns
        -------
        sparse.csr_matrix
            The CSR form of the operator.

        """
        if self.matrix is not None:
            return sparse.coo_matrix((self.matrix, (self.sparsity.row, self.sparsity.col)), shape=(self.num_sites, self.num_sites)).tocsr()
        elif self.functional is not None:
            return sparse.coo_matrix((self.get_matrix_data(), (self.sparsity.row, self.sparsity.col)), shape=(self.num_sites, self.num_sites)).tocsr()
        else:
            raise ValueError("No matrix or functional form set.")
        

    def apply_operator_on(self,matrix:sparse) -> None:
        """Applies the operator on a matrix.

        Parameters
        ----------

        matrix : sparse
            The matrix to apply the operator on.

        """
        if self.matrix is not None:
            ... # apply matrix
        elif self.functional is not None:
            ... # apply functional
        else:
            raise ValueError("No matrix or functional form set.")

class RealSpacePairOperator():
    """Data class for real-space pair-interaction operators.    

    - Sparse matrix form (COO)
    - Functional form (e.g. Coulomb interaction)
    - block sectioning

    """
    def __init__(self, positions:NDArray, cutoff_distance: float, dtype = float, strategy:str = 'sphere', ordering:str='normal') -> None:
        """Initializes the real-space operator.
        
        Parameters
        ----------
        
        dimension : int
            The dimension of the operator.

        positions : NDArray
            The positions of the sites.
            
        cutoff_distance : float
            The cutoff distance for the interaction.        

        dtype : type    
            The data type of the operator.

        """
        self.dimension = 4
        self.cutoff_distance = cutoff_distance
        self.positions = positions     
        self.dtype = dtype   
        self.num_sites = len(positions)
        coo = _compute_sparsity_pattern(positions, cutoff_distance,strategy=strategy)

        if (ordering == 'arrowhead'):
            # permute the sparsity pattern to allow arrowhead ordering of the pair-interaction matrix            
            perm = xp.lexsort((coo.row, (coo.row == coo.col)))
            row = coo.row[perm]
            col = coo.col[perm]
            coo.row = row
            coo.col = col

        self.sparsity = coo
        self.bandwidth = max(abs(coo.col - coo.row))
        self.ordering = ordering
        self.nni = len(self.sparsity.row)

        coo = _compute_pair_sparsity_pattern(self.sparsity)
        self.pair_sparsity = coo
        self.nnz = len(self.pair_sparsity.row)
        self.pair_bandwidth = max(abs(coo.col - coo.row))
        
        self.matrix = None
        self.functional = None
        self.blocksizes = None
        self.block_offsets = None
        self.block_sparsity_map = None


    def set_matrix(self,matrix:NDArray) -> None:
        """Sets the matrix form of the operator by passing in a data vector.
        
        Parameters
        ----------
        
        matrix : NDArray
            The matrix form of the operator.
        
        """
        if (self.matrix is None) or (len(self.matrix) < len(matrix)):
            self.matrix = xp.zeros(self.pair_sparsity.row.shape, dtype=self.dtype) 
        self.matrix[:len(matrix)] = matrix


    def set_functional(self, functional:function):
        """Sets the functional form of the operator by passing in a function.
        
        Parameters  
        ----------
        functional : function
            The functional form of the operator, takes as input two or more positions of the site.

        """
        self.functional = functional


    def get_dense(self) -> NDArray:
        """Returns the dense form of the pair-operator.

        Returns
        -------
        NDArray
            The dense form of the pair-operator.

        """
        mat = xp.zeros(self.pair_sparsity.shape, dtype=self.dtype)
        if self.matrix is not None:
            mat[self.sparsity.row, self.sparsity.col] = self.matrix          
            return mat
        elif self.functional is not None:       
            for i in range(len(self.pair_sparsity.row)):
                mat[self.pair_sparsity.row[i], self.pair_sparsity.col[i]] = self.functional(self.positions[self.sparsity.row[self.pair_sparsity.row[i]]], 
                                                                                            self.positions[self.sparsity.col[self.pair_sparsity.row[i]]],
                                                                                            self.positions[self.sparsity.row[self.pair_sparsity.col[i]]], 
                                                                                            self.positions[self.sparsity.col[self.pair_sparsity.col[i]]])             
            return mat
            
    def get_matrix_data(self) -> NDArray: 
        """Returns the data of the operator.

        Returns
        -------
        NDArray
            The data of the operator.

        """
        if self.matrix is not None:
            return self.matrix
        elif self.functional is not None:
            return xp.array([self.functional(self.positions[self.sparsity.row[self.pair_sparsity.row[i]]], 
                                                            self.positions[self.sparsity.col[self.pair_sparsity.row[i]]],
                                                            self.positions[self.sparsity.row[self.pair_sparsity.col[i]]], 
                                                            self.positions[self.sparsity.col[self.pair_sparsity.col[i]]])  
                            for i in range(len(self.pair_sparsity.row))], dtype=self.dtype)
        else:
            raise ValueError("No matrix or functional form set.")
        

    
    def get_coo(self) -> sparse.coo_matrix:
        """Returns the COO form of the operator.

        Returns
        -------
        sparse.coo_matrix
            The COO form of the operator.

        """
        if self.matrix is not None:
            return sparse.coo_matrix((self.matrix, (self.pair_sparsity.row, self.pair_sparsity.col)), shape=(self.nni, self.nni))
        elif self.functional is not None:
            return sparse.coo_matrix((self.get_matrix_data(), (self.pair_sparsity.row, self.pair_sparsity.col)), shape=(self.nni, self.nni))
        else:
            raise ValueError("No matrix or functional form set.")
        

    def get_csr(self) -> sparse.csr_matrix:
        """Returns the CSR form of the operator.

        Returns
        -------
        sparse.csr_matrix
            The CSR form of the operator.

        """
        if self.matrix is not None:
            return sparse.coo_matrix((self.matrix, (self.pair_sparsity.row, self.pair_sparsity.col)), shape=(self.nni, self.nni)).tocsr()
        elif self.functional is not None:
            return sparse.coo_matrix((self.get_matrix_data(), (self.pair_sparsity.row, self.pair_sparsity.col)), shape=(self.nni, self.nni)).tocsr()
        else:
            raise ValueError("No matrix or functional form set.")
        




def _compute_block_map(sparsity: sparse.coo_matrix, blocksizes:NDArray) -> NDArray:
    """Computes the block sizes of the sparsity pattern.

    Parameters
    ----------
    sparsity : sparse.coo_matrix
        The sparsity pattern.

    blocksizes : NDArray
        The block sizes.

    Returns
    -------
    NDArray
        The block sizes.

    """
    cumsum = xp.cumsum(blocksizes)
    blockmap = xp.zeros((len(sparsity.row),2), dtype=xp.int32)
    for i in range(len(cumsum) - 1):
        blockmap[:,0] += (sparsity.row >= cumsum[i]) 
        blockmap[:,1] += (sparsity.col >= cumsum[i])
    return blockmap
        
def _compute_pair_sparsity_pattern(
    sparsity: sparse.coo_matrix
) -> NDArray: 
    """Computes the sparsity pattern for a pair-interaction matrix A(a,b,c,d) flattened into a COO matrix by combining first two and last two index.

    Parameters
    ----------
    sparsity : sparse.coo_matrix
        The sparsity pattern of interaction matrix.

    Returns
    -------
    NDArray
        The pair sparsity pattern.

    """    
    lil = sparsity.tolil()
    pair_rows, pair_cols = [], []    
    for i, a in enumerate(sparsity.row):        
        b = sparsity.col[i]        
        cs = xp.where(lil[a,:] != 0)
        ds = xp.where(lil[b,:] != 0)
        
        interactions = xp.where((lil[a, sparsity.row].todense() != 0) & (lil[b, sparsity.col].todense() != 0))
        pair_rows.extend([i]* len(interactions))
        pair_cols.extend(interactions)
        # for j, c in enumerate(sparsity.row):   
        #     d = sparsity.col[j]
        #     if (lil[a, c] != 0) and (lil[b,d] != 0):
        #         pair_cols.append(i)
        #         pair_rows.append(j)                      
    
    rows, cols = xp.array(pair_rows), xp.array(pair_cols)
    return sparse.coo_matrix((xp.ones_like(rows, dtype=xp.float32), (rows, cols)))


def _compute_sparsity_pattern(
    positions: NDArray,
    cutoff_distance: float,
    strategy: str = "box",
) -> sparse.coo_matrix:
    """Computes the sparsity pattern for the interaction matrix.

    Parameters
    ----------
    grid : NDArray
        The grid points.
    interaction_cutoff : float
        The interaction cutoff.
    strategy : str, optional
        The strategy to use, by default "box", where only the distance
        along the transport direction is considered. The other option is
        "sphere", where the usual Euclidean distance between points
        matters.

    Returns
    -------
    sparse.coo_matrix
        The sparsity pattern.

    """
    if strategy == "sphere":

        def distance(x, y):
            """Euclidean distance."""
            return xp.linalg.norm(x - y, axis=-1)

    elif strategy == "box":

        def distance(x, y):
            """Distance along transport direction."""
            return xp.abs(x[..., 0] - y[..., 0])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    rows, cols = [], []
    for i, position in enumerate(positions):
        distances = distance(positions, position)
        interacting = xp.where(distances < cutoff_distance)[0]
        cols.extend(interacting)
        rows.extend([i] * len(interacting))

    rows, cols = xp.array(rows), xp.array(cols)
    return sparse.coo_matrix((xp.ones_like(rows, dtype=xp.float32), (rows, cols)))

