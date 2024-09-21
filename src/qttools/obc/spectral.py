import numpy.linalg as npla

from qttools.datastructures.dsbsparse import _block_view
from qttools.nevp import NEVP
from qttools.obc.obc import OBC
from qttools.utils.gpu_utils import xp


class Spectral(OBC):
    """Spectral open-boundary condition solver.

    This technique of obtaining the surface Green's function is based on
    the solution of a non-linear eigenvalue problem (NEVP), defined via
    the system-matrix blocks in the semi-infinite contacts.

    Those eigenvalues corresponding to reflected modes are filtered out,
    so that only the ones that correspond to modes that propagate into
    the leads or thos that decay away from the system are retained.

    The surface Green's function is then calculated from these filtered
    eigenvalues and eigenvectors.

    Parameters
    ----------
    nevp : NEVP
        The non-linear eigenvalue problem solver to use.
    block_sections : int, optional
        The number of sections to split the periodic matrix layer into.
    min_decay : float, optional
        The decay threshold after which modes are considered to be
        evanescent.
    max_decay : float, optional
        The maximum decay to consider for evanescent modes. If not
        provided, the maximum decay is set to the logarithm of the
        outer radius of the contour annulus if applicable. Otherwise,
        it is set to log(10).
    num_ref_iterations : int, optional
        The number of refinement iterations to perform on the surface
        Green's function.

    """

    def __init__(
        self,
        nevp: NEVP,
        block_sections: int = 1,
        min_decay: float = 1e-6,
        max_decay: float | None = None,
        num_ref_iterations: int = 2,
    ) -> None:
        """Initializes the spectral OBC solver."""
        self.nevp = nevp

        self.min_decay = min_decay
        if max_decay is None:
            max_decay = xp.log(getattr(nevp, "r_o", 10.0))
        self.max_decay = max_decay

        self.num_ref_iterations = num_ref_iterations
        self.block_sections = block_sections

    def _extract_subblocks(
        self,
        a_ji: xp.ndarray,
        a_ii: xp.ndarray,
        a_ij: xp.ndarray,
    ) -> list[xp.ndarray]:
        """Extracts the coefficient blocks from the periodic matrix.

        Parameters
        ----------
        a_ji : xp.ndarray
            The subdiagonal block of the periodic matrix.
        a_ii : xp.ndarray
            The diagonal block of the periodic matrix.
        a_ij : xp.ndarray
            The superdiagonal block of the periodic matrix.

        Returns
        -------
        blocks : list[xp.ndarray]
            The non-zero blocks making up the matrix layer.

        """
        # Construct layer of periodic matrix in semi-infinite lead.
        layer = [a_ji, a_ii, a_ij]
        if self.block_sections == 1:
            return layer

        # Get a nested block view of the layer.
        view = _block_view(xp.hstack(layer), -1, 3 * self.block_sections)
        view = _block_view(view, -2, self.block_sections)

        # Make sure that the reduction leads to periodic sublayers.
        # NOTE: I'm not 100% sure that this is really necessary.
        for i in range(self.block_sections):
            if not xp.allclose(view[0, :], xp.roll(view[i, :], -i, axis=0)):
                raise ValueError("Requested block sectioning is not periodic.")

        # Select relevant blocks and remove empty ones.
        blocks = view[0, : -self.block_sections + 1]
        blocks = [block for block in blocks if xp.any(block)]

        return blocks

    def _get_reflected_modes(
        self,
        ws: xp.ndarray,
        vs: xp.ndarray,
        a_ij: xp.ndarray,
        a_ji: xp.ndarray,
        contact: str,
    ) -> tuple[xp.ndarray, xp.ndarray]:
        """Filters out eigenvalues corresponding to injected modes.

        For the computation of the surface Green's function, only the
        eigenvalues corresponding to modes that propagate or decay into
        the leads are retained. This function filters out the rest.

        Parameters
        ----------
        ws : xp.ndarray
            The eigenvalues of the NEVP.
        vs : xp.ndarray
            The eigenvectors of the NEVP.
        a_ij : xp.ndarray
            The superdiagonal contact block.
        a_ji : xp.ndarray
            The subdiagonal contact block.
        contact : str
            The contact side to filter for.

        Returns
        -------
        ws : xp.ndarray
            The filtered eigenvalues.
        vs : xp.ndarray
            The filtered eigenvectors.

        """
        ks = -1j * xp.log(ws)

        # Calculate the group velocity to select propgagation direction.
        dEk_dk = xp.zeros_like(ws)

        # NOTE: This is actually only correct if we have no overlap.
        # phi.H d/dk (H00 + lambda * H01 + 1/lambda * H10) phi =
        # phi.H d/dk energy (S00 + lambda * S01 + 1/lambda * S10) phi
        for i, w in enumerate(ws):
            a = -(1j * w * a_ij - 1j / w * a_ji)
            phi = vs[:, i]
            dEk_dk[i] = (phi.conj().T @ a @ phi) / (phi.conj().T @ phi)

        # Find eigenvalues that correspond to reflected modes.
        if contact == "left":
            # Find left propagating or left decaying modes.
            inds = xp.argwhere(
                ((dEk_dk.real < 0) & (xp.abs(ks.imag) < self.min_decay))
                | ((ks.imag < -self.min_decay) & (ks.imag > -self.max_decay))
            )[:, 0]
        elif contact == "right":
            # Find right propagating or right decaying modes.
            inds = xp.argwhere(
                ((dEk_dk.real > 0) & (xp.abs(ks.imag) < self.min_decay))
                | ((ks.imag > self.min_decay) & (ks.imag < self.max_decay))
            )[:, 0]
        else:
            raise ValueError(f"Invalid contact: {contact}")

        return ws[inds], vs[:, inds]

    def __call__(
        self,
        a_ii: xp.ndarray,
        a_ij: xp.ndarray,
        a_ji: xp.ndarray,
        contact: str,
        out: None | xp.ndarray = None,
    ) -> xp.ndarray | None:
        """Returns the surface Green's function."""
        blocks = self._extract_subblocks(a_ji, a_ii, a_ij)
        ws, vs = self.nevp(blocks)
        ws, vs = self._get_reflected_modes(ws, vs, a_ij, a_ji, contact)

        # Calculate the surface Green's function.
        vs_inv = xp.linalg.inv(xp.swapaxes(vs, -1, -2) @ vs) @ xp.swapaxes(vs, -1, -2)
        x_ii = xp.linalg.inv(a_ii + a_ji @ vs @ xp.diag(1 / ws) @ vs_inv)

        # Perform a number of refinement iterations.
        for __ in range(self.num_ref_iterations):
            x_ii = npla.inv(a_ii - a_ji @ x_ii @ a_ij)

        # Return the surface Green's function.
        if out is not None:
            out[...] = x_ii
            return

        return x_ii
