import torch

def B_batch(x, grid, k=0, extend=True, device=None, dtype=torch.float32):
    """
    Evaluate x on B-spline bases.
    
    Args:
    -----
        x : 2D torch.tensor
            Inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            Grids, shape (number of splines, number of grid points)
        k : int
            The piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str or torch.device, optional
            Device to perform the computation on.
        dtype : torch.dtype, optional
            Data type of tensors. Default: torch.float32
    
    Returns:
    --------
        spline values : 3D torch.tensor
            Shape (number of splines, number of B-spline bases (coefficients), number of samples). The number of B-spline bases = number of grid points + k - 1.
    
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0, 1, size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1, 1, steps=num_grid_interval + 1))
    >>> B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x, grid = x.to(device, dtype=dtype), grid.to(device, dtype=dtype)

    def extend_grid(grid, k_extend=0):
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        return grid

    if extend:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)
    
    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device, dtype=dtype)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    
    return value

def coef2curve(x_eval, grid, coef, k, device=None, dtype=torch.float32):
    """
    Convert B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor
            Shape (number of splines, number of samples)
        grid : 2D torch.tensor
            Shape (number of splines, number of grid points)
        coef : 2D torch.tensor
            Shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            The piecewise polynomial order of splines.
        device : str or torch.device, optional
            Device to perform the computation on.
        dtype : torch.dtype, optional
            Data type of tensors. Default: torch.float32.
        
    Returns:
    --------
        y_eval : 2D torch.tensor
            Shape (number of splines, number of samples)
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0, 1, size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1, 1, steps=num_grid_interval + 1))
    >>> coef = torch.normal(0, 1, size=(num_spline, num_grid_interval+k))
    >>> coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_eval, grid, coef = x_eval.to(device, dtype=dtype), grid.to(device, dtype=dtype), coef.to(device, dtype=dtype)
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device, dtype=dtype))
    return y_eval

def curve2coef(x_eval, y_eval, grid, k, device=None, dtype=torch.float32):
    """
    Convert B-spline curves to B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            Shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            Shape (number of splines, number of samples)
        grid : 2D torch.tensor
            Shape (number of splines, number of grid points)
        k : int
            The piecewise polynomial order of splines.
        device : str or torch.device, optional
            Device to perform the computation on.
        dtype : torch.dtype, optional
            Data type of tensors. Default: torch.float32.
        
    Returns:
    --------
        coef : 2D torch.tensor
            Shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0, 1, size=(num_spline, num_sample))
    >>> y_eval = torch.normal(0, 1, size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1, 1, steps=num_grid_interval + 1))
    >>> curve2coef(x_eval, y_eval, grids, k=k).shape
    torch.Size([5, 13])
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_eval, y_eval, grid = x_eval.to(device, dtype=dtype), y_eval.to(device, dtype=dtype), grid.to(device, dtype=dtype)
    mat = B_batch(x_eval, grid, k, device=device, dtype=dtype).permute(0, 2, 1)
    coef = torch.linalg.lstsq(mat.to('cpu'), y_eval.unsqueeze(dim=2).to('cpu')).solution[:, :, 0]
    return coef.to(device, dtype=dtype)
