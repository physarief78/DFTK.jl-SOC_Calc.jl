import Interpolations
import Interpolations: interpolate, extrapolate, scale, BSpline, Quadratic, Periodic, OnCell

"""
Interpolate a density expressed in a basis `basis_in` to a basis `basis_out`.
This interpolation uses a very basic real-space algorithm, and makes a DWIM-y attempt
to take into account the fact that `basis_out` can be a supercell of `basis_in`.

[MODIFIED]: Automatically handles spin projection for ALL cases (:none, :collinear, :full).
"""
function interpolate_density(ρ_in::AbstractArray{T, 4},
                             basis_in::PlaneWaveBasis,
                             basis_out::PlaneWaveBasis) where {T}
    # 1. Perform Spatial Interpolation (Grid -> Grid)
    #    This returns a density with the SAME spin components as ρ_in, but on the new grid.
    ρ_spatial = if basis_in.model.lattice == basis_out.model.lattice
        @assert size(ρ_in) == (basis_in.fft_size..., basis_in.model.n_spin_components)
        interpolate_density(ρ_in, basis_out.fft_size)
    else
        interpolate_density(ρ_in, basis_in.fft_size, basis_out.fft_size,
                            basis_in.model.lattice, basis_out.model.lattice)
    end

    # 2. Handle Spin Projection
    n_spin_in  = size(ρ_in, 4)
    n_spin_out = basis_out.model.n_spin_components

    # --- Case A: No Change (1->1, 2->2, 4->4) ---
    if n_spin_in == n_spin_out
        return ρ_spatial
    end

    # Allocate Output Array
    ρ_out = zeros(T, size(ρ_spatial)[1:3]..., n_spin_out)

    # --- Case B: Scalar (1) -> Collinear (2) ---
    # Split non-magnetic density into equal Up/Down
    if n_spin_in == 1 && n_spin_out == 2
        ρ_out[:, :, :, 1] .= ρ_spatial[:, :, :, 1] ./ 2
        ρ_out[:, :, :, 2] .= ρ_spatial[:, :, :, 1] ./ 2
        return ρ_out
    end

    # --- Case C: Scalar (1) -> Full (4) ---
    # Set Total Density, Magnetization is zero
    if n_spin_in == 1 && n_spin_out == 4
        ρ_out[:, :, :, 1] .= ρ_spatial[:, :, :, 1]
        # Mx, My, Mz (indices 2,3,4) remain 0.0
        return ρ_out
    end

    # --- Case D: Collinear (2) -> Full (4) ---
    # Map Up/Down to Total/Mz. (Mx, My = 0)
    if n_spin_in == 2 && n_spin_out == 4
        # Total = Up + Down
        ρ_out[:, :, :, 1] .= ρ_spatial[:, :, :, 1] .+ ρ_spatial[:, :, :, 2]
        # Mz = Up - Down
        ρ_out[:, :, :, 4] .= ρ_spatial[:, :, :, 1] .- ρ_spatial[:, :, :, 2]
        return ρ_out
    end

    # --- Case E: Collinear (2) -> Scalar (1) ---
    # Discard magnetization, keep total charge
    if n_spin_in == 2 && n_spin_out == 1
        ρ_out[:, :, :, 1] .= ρ_spatial[:, :, :, 1] .+ ρ_spatial[:, :, :, 2]
        return ρ_out
    end

    # --- Case F: Full (4) -> Scalar (1) ---
    # Discard magnetization vector, keep total charge
    if n_spin_in == 4 && n_spin_out == 1
        ρ_out[:, :, :, 1] .= ρ_spatial[:, :, :, 1]
        return ρ_out
    end

    # --- Case G: Full (4) -> Collinear (2) ---
    # Project Z-component of magnetization to Up/Down. Discard Mx, My.
    if n_spin_in == 4 && n_spin_out == 2
        # ρ_up   = (Total + Mz) / 2
        ρ_out[:, :, :, 1] .= (ρ_spatial[:, :, :, 1] .+ ρ_spatial[:, :, :, 4]) ./ 2
        # ρ_down = (Total - Mz) / 2
        ρ_out[:, :, :, 2] .= (ρ_spatial[:, :, :, 1] .- ρ_spatial[:, :, :, 4]) ./ 2
        return ρ_out
    end

    # Fallback
    error("Automatic interpolation not implemented for spin change: $n_spin_in -> $n_spin_out")
end

"""
Interpolate a density in real space from one FFT grid to another. Assumes the
lattice is unchanged.
"""
function interpolate_density(ρ_in::AbstractArray{T, 4}, grid_out::NTuple{3}) where {T}
    n_spin = size(ρ_in, 4)
    interpolate_density!(similar(ρ_in, grid_out..., n_spin), ρ_in)
end

"""
Interpolate a density in real space from one FFT grid to another, where
`lattice_in` and `lattice_out` may be supercells of each other.
"""
function interpolate_density(ρ_in::AbstractArray{T, 4},
                             grid_in::NTuple{3}, grid_out::NTuple{3},
                             lattice_in, lattice_out) where {T}
    # The two lattices should have the same dimension.
    @assert iszero.(eachcol(lattice_in)) == iszero.(eachcol(lattice_out))
    @assert size(ρ_in)[1:3] == grid_in

    # Build supercell, array of 3 integers
    supercell = map(eachcol(lattice_in), eachcol(lattice_out)) do col_in, col_out
        iszero(col_in) ? 1 : round(Int, norm(col_out) / norm(col_in))
    end

    # Check if some direction of lattice_in is not too big compared to lattice_out.
    supercell_in = supercell .* lattice_in
    is_suspicious_direction = map(eachcol(supercell_in), eachcol(lattice_out)) do s_in, a_out
        norm(s_in - a_out) > 0.3*norm(a_out)
    end
    for i in findall(is_suspicious_direction)
        @warn "In direction $i, the output lattice is very different from the input lattice"
    end

    # ρ_in represents a periodic function, on a grid 0, 1/N, ... (N-1)/N
    grid_supercell = grid_in .* supercell
    ρ_in_supercell = similar(ρ_in, grid_supercell..., size(ρ_in, 4))
    for i = 1:supercell[1], j = 1:supercell[2], k = 1:supercell[3]
        ρ_in_supercell[1 + (i-1)*grid_in[1] : i*grid_in[1],
                       1 + (j-1)*grid_in[2] : j*grid_in[2],
                       1 + (k-1)*grid_in[3] : k*grid_in[3], :] = ρ_in
    end

    interpolate_density(ρ_in_supercell, grid_out)
end

function interpolate_density!(ρ_out::AbstractArray{T, 3}, ρ_in::AbstractArray{T, 3}) where {T}
    size(ρ_in) == size(ρ_out) && return ρ_out .= ρ_in

    grid_in  = size(ρ_in)
    grid_out = size(ρ_out)
    axes_in = (range(0, 1, length=grid_in[i]+1)[1:end-1] for i=1:3)
    itp = interpolate(ρ_in, BSpline(Quadratic(Interpolations.Periodic(OnCell()))))
    sitp = scale(itp, axes_in...)
    interpolator = extrapolate(sitp, Periodic())
    for i = 1:grid_out[1], j = 1:grid_out[2], k = 1:grid_out[3]
        ρ_out[i, j, k] = interpolator((i-1)/grid_out[1],
                                      (j-1)/grid_out[2],
                                      (k-1)/grid_out[3])
    end
    ρ_out
end
function interpolate_density!(ρ_out::AbstractArray{T, 4}, ρ_in::AbstractArray{T, 4}) where {T}
    @assert size(ρ_in, 4) == size(ρ_out, 4)
    for (ρ_out_slice, ρ_in_slice) in zip(eachslice(ρ_out; dims=4), eachslice(ρ_in; dims=4))
        interpolate_density!(ρ_out_slice, ρ_in_slice)
    end
    ρ_out
end


"""
Interpolate some data from one ``k``-point to another. The interpolation is fast, but not
necessarily exact. Intended only to construct guesses for iterative solvers.
"""
# CHANGED: To apply the SOC and :full spin_polarization calculation
function interpolate_kpoint(data_in::AbstractVecOrMat,
                            basis_in::PlaneWaveBasis,  kpoint_in::Kpoint,
                            basis_out::PlaneWaveBasis, kpoint_out::Kpoint)
    # TODO merge with transfer_blochwave_kpt
    if kpoint_in == kpoint_out
        return copy(data_in)
    end

    # [FIX] Handle SOC (Spinors)
    n_spinors = basis_in.model.spin_polarization == :full ? 2 : 1
    
    n_G_in  = length(G_vectors(basis_in, kpoint_in))
    n_G_out = length(G_vectors(basis_out, kpoint_out))
    
    # [FIX] Assertion must account for spinor size
    @assert size(data_in, 1) == n_G_in * n_spinors

    n_bands = size(data_in, 2)
    
    # [FIX] Allocate output with correct spinor size
    data_out = similar(data_in, n_G_out * n_spinors, n_bands)
    fill!(data_out, 0)

    # Buffer for global G-vector mapping (scalar size is sufficient per component)
    max_nG = max(length(G_vectors(basis_in)), length(G_vectors(basis_out)))
    tmp = similar(data_in, max_nG, n_bands)

    # Iterate over spinor components (1 for Scalar, 2 for SOC)
    for i_spin = 1:n_spinors
        # Define ranges for the current spinor component
        # Component 1: 1:nG, Component 2: nG+1:2nG
        range_in  = (1 + (i_spin-1)*n_G_in) : (i_spin*n_G_in)
        range_out = (1 + (i_spin-1)*n_G_out) : (i_spin*n_G_out)
        
        # 1. Clear buffer
        fill!(tmp, 0)
        
        # 2. Scatter input to global grid (tmp) using input mapping
        # We view the specific component slice of data_in
        tmp[kpoint_in.mapping, :] .= view(data_in, range_in, :)
        
        # 3. Gather from global grid (tmp) to output using output mapping
        data_out[range_out, :] .= view(tmp, kpoint_out.mapping, :)
    end

    ortho_qr(data_out)  # Re-orthogonalize and renormalize
end
