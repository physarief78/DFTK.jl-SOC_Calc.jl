# Densities (and potentials) are represented by arrays
# ρ[ix,iy,iz,iσ] in real space, where iσ ∈ [1:n_spin_components]

"""
    compute_density(basis::PlaneWaveBasis, ψ::AbstractVector, occupation::AbstractVector)

Compute the density for a wave function `ψ` discretized on the plane-wave
grid `basis`, where the individual k-points are occupied according to `occupation`.
`ψ` should be one coefficient matrix per ``k``-point.
It is possible to ask only for occupations higher than a certain level to be computed by
using an optional `occupation_threshold`. By default all occupation numbers are considered.
"""
@views @timing function compute_density(basis::PlaneWaveBasis{T,VT}, ψ, occupation;
                                        occupation_threshold=zero(T)) where {T,VT}
    # Occupation should be on the CPU as we are going to be doing scalar indexing.
    occupation = [to_cpu(oc) for oc in occupation]
    mask_occ = [findall(occnk -> abs(occnk) ≥ occupation_threshold, occk)
                for occk in occupation]

    function allocate_local_storage()
        Tρ = promote_type(T, real(eltype(ψ[1])))
        Tψ = promote_type(VT, real(eltype(ψ[1])))
        
        # Temp buffers for spinors
        tmp_real = zeros(complex(Tψ), basis.fft_size...)
        
        if basis.model.spin_polarization == :full
             tmp_real_2 = zeros(complex(Tψ), basis.fft_size...)
             return (; ρ=zeros(Tρ, basis.fft_size..., basis.model.n_spin_components),
                      ψnk_real=tmp_real, ψnk_real_2=tmp_real_2)
        else
             return (; ρ=zeros(Tρ, basis.fft_size..., basis.model.n_spin_components),
                      ψnk_real=tmp_real)
        end
    end

    range = [(ik, n) for ik = 1:length(basis.kpoints) for n = mask_occ[ik]]

    storages = parallel_loop_over_range(range; allocate_local_storage) do kn, storage
        (ik, n) = kn
        kpt = basis.kpoints[ik]
        
        # === SOC / Non-Collinear Logic ===
        if basis.model.spin_polarization == :full
            n_G = length(G_vectors(basis, kpt))
            
            # 1. Extract and FFT Up component (normalize=false to match weight logic)
            ifft!(storage.ψnk_real, basis, kpt, ψ[ik][1:n_G, n]; normalize=false)
            
            # 2. Extract and FFT Down component
            ifft!(storage.ψnk_real_2, basis, kpt, ψ[ik][n_G+1:end, n]; normalize=false)
            
            # 3. Accumulate Density Components (n, mx, my, mz)
            # Apply normalization: (ifft_norm)^2
            norm_fac = (basis.fft_grid.ifft_normalization)^2
            weight = occupation[ik][n] * basis.kweights[ik] * norm_fac
            
            ψu = storage.ψnk_real
            ψd = storage.ψnk_real_2
            
            @. begin
                storage.ρ[:, :, :, 1] += weight * (abs2(ψu) + abs2(ψd))        # Total Charge n
                storage.ρ[:, :, :, 2] += weight * 2 * real(conj(ψu) * ψd)      # Mx
                storage.ρ[:, :, :, 3] += weight * 2 * imag(conj(ψu) * ψd)      # My
                storage.ρ[:, :, :, 4] += weight * (abs2(ψu) - abs2(ψd))        # Mz
            end
            
        else
            # === Scalar/Collinear Logic ===
            ifft!(storage.ψnk_real, basis, kpt, ψ[ik][:, n]; normalize=false)
            storage.ρ[:, :, :, kpt.spin] .+= (occupation[ik][n] .* basis.kweights[ik]
                                              .* (basis.fft_grid.ifft_normalization)^2
                                              .* abs2.(storage.ψnk_real))
        end
    end
    
    ρ = sum(storage -> storage.ρ, storages)
    mpi_sum!(ρ, basis.comm_kpts)
    ρ = symmetrize_ρ(basis, ρ; do_lowpass=false)

    negtol = max(sqrt(eps(T)), 10occupation_threshold)
    minimum(ρ) < -negtol && @warn("Negative ρ detected", min_ρ=minimum(ρ))

    ρ
end

# Variation in density corresponding to a variation in the orbitals and occupations.
@views @timing function compute_δρ(basis::PlaneWaveBasis{T}, ψ, δψ, occupation,
                                   δoccupation=zero.(occupation);
                                   occupation_threshold=zero(T), q=zero(Vec3{T})) where {T}
    Tψ = promote_type(T, eltype(ψ[1]))
    # δρ is expected to be real when computations are not phonon-related.
    Tδρ = iszero(q) ? real(Tψ) : Tψ
    real_qzero = iszero(q) ? real : identity

    # occupation should be on the CPU as we are going to be doing scalar indexing.
    occupation = [to_cpu(oc) for oc in occupation]
    mask_occ = [findall(occnk -> abs(occnk) ≥ occupation_threshold, occk)
                for occk in occupation]

    function allocate_local_storage()
        (; δρ=zeros_like(G_vectors(basis), Tδρ, basis.fft_size..., basis.model.n_spin_components),
          ψnk_real=zeros_like(G_vectors(basis), Tψ, basis.fft_size...),
         δψnk_real=zeros_like(G_vectors(basis), Tψ, basis.fft_size...))
    end
    range = [(ik, n) for ik = 1:length(basis.kpoints) for n = mask_occ[ik]]

    # The variation of the orbital ψ_k defined in the basis ℬ_k is δψ_{[k+q]} in ℬ_{[k+q]},
    # where [k+q] is equivalent to the basis k+q (see find_equivalent_kpt).
    # The perturbation of the density
    #   |ψ_{n,k}|² is 2 ψ_{n,k} * δψ_{n,k+q}.
    # Hence, we first get the δψ_{[k+q]} as δψ_{k+q}…
    δψ_plus_k = transfer_blochwave_equivalent_to_actual(basis, δψ, q)
    storages = parallel_loop_over_range(range; allocate_local_storage) do kn, storage
        (ik, n) = kn

        kpt = basis.kpoints[ik]
        ifft!(storage.ψnk_real, basis, kpt, ψ[ik][:, n])
        # … and then we compute the real Fourier transform in the adequate basis.
        ifft!(storage.δψnk_real, basis, δψ_plus_k[ik].kpt, δψ_plus_k[ik].ψk[:, n])

        storage.δρ[:, :, :, kpt.spin] .+= real_qzero.(
            2 .* occupation[ik][n]  .* basis.kweights[ik] .* conj.(storage.ψnk_real)
                                                          .* storage.δψnk_real
              .+ δoccupation[ik][n] .* basis.kweights[ik] .* abs2.(storage.ψnk_real))

    end
    δρ = sum(getfield.(storages, :δρ))

    mpi_sum!(δρ, basis.comm_kpts)
    symmetrize_ρ(basis, δρ; do_lowpass=false)
end

@views @timing function compute_kinetic_energy_density(basis::PlaneWaveBasis, ψ, occupation)
    T = promote_type(eltype(basis), real(eltype(ψ[1])))
    τ = similar(ψ[1], T, (basis.fft_size..., basis.model.n_spin_components))
    τ .= 0
    dαψnk_real = zeros_like(G_vectors(basis), complex(eltype(basis)), basis.fft_size...)
    occupation = [to_cpu(oc) for oc in occupation]
    for (ik, kpt) in enumerate(basis.kpoints)
        G_plus_k = [map(p -> p[α], Gplusk_vectors_cart(basis, kpt)) for α = 1:3]
        for n = 1:size(ψ[ik], 2), α = 1:3
            ifft!(dαψnk_real, basis, kpt, im .* G_plus_k[α] .* ψ[ik][:, n])
            @. τ[:, :, :, kpt.spin] += occupation[ik][n] * basis.kweights[ik] / 2 * abs2(dαψnk_real)
        end
    end
    mpi_sum!(τ, basis.comm_kpts)
    symmetrize_ρ(basis, τ; do_lowpass=false)
end

# Modified total_density(ρ) to enable the :full calculation
function total_density(ρ)
    if size(ρ, 4) == 4
        # For :full, the first component is already the total density
        return ρ[:, :, :, 1]
    else
        # For :collinear (2) or :spinless (1), sum is correct
        return dropdims(sum(ρ; dims=4); dims=4)
    end
end
# Modified spin_density(ρ) to match the modified :full calculation support
@views function spin_density(ρ)
    if size(ρ, 4) == 2
        return ρ[:, :, :, 1] - ρ[:, :, :, 2]
    elseif size(ρ, 4) == 4
        # Return magnitude of magnetization vector |M|
        return sqrt.(abs2.(ρ[:, :, :, 2]) .+ abs2.(ρ[:, :, :, 3]) .+ abs2.(ρ[:, :, :, 4]))
    else
        return zero(ρ[:, :, :])
    end
end

# Modified to allow the spin_polarization = :full calculation
function ρ_from_total_and_spin(ρtot, ρspin=nothing)
    if ρspin === nothing
        # Spinless / None
        return cat(ρtot; dims=Val(4))
        
    elseif ρspin isa AbstractArray && size(ρtot) == size(ρspin)
        # Collinear case: Construct (Up, Down)
        # ρspin is treated as Magnetization Magnitude (Mz)
        return cat((ρtot .+ ρspin) ./ 2,
                   (ρtot .- ρspin) ./ 2; dims=Val(4))
                   
    elseif (ρspin isa Tuple || ρspin isa AbstractVector) && length(ρspin) == 3
        # Full / Non-collinear case: Construct (n, Mx, My, Mz)
        # We expect ρspin = (Mx, My, Mz)
        mx, my, mz = ρspin
        return cat(ρtot, mx, my, mz; dims=Val(4))
        
    else
        error("Invalid input for density reconstruction. " * "For :collinear pass a single array (Mz). " * "For :full pass a tuple of 3 arrays (Mx, My, Mz).")
    end
end

# === CRITICAL FUNCTION FOR INITIALIZATION ===
function ρ_from_total(basis, ρtot::AbstractArray{T}) where {T}
    model = basis.model
    if model.spin_polarization in (:none, :spinless)
        ρspin = nothing
    elseif model.spin_polarization == :collinear
        ρspin = zeros(T, basis.fft_size...)
    elseif model.spin_polarization == :full
        z = zeros(T, basis.fft_size...)
        ρspin = (z, z, z)
    else
        error("Unknown spin polarization: $(model.spin_polarization)")
    end
    ρ_from_total_and_spin(ρtot, ρspin)
end

# === HELPER FOR XC FORCES ===
function atomic_density_form_factors(basis::PlaneWaveBasis{T}, atomic_density) where {T}
    n_groups = length(basis.model.atom_groups)
    Gs = G_vectors_cart(basis)
    form_factors = zeros(Complex{T}, length(Gs), n_groups)
    
    for (ig, group) in enumerate(basis.model.atom_groups)
        element = basis.model.atoms[first(group)]
        if element isa ElementPsp
             for (iG, G) in enumerate(Gs)
                 if atomic_density isa CoreDensity
                     form_factors[iG, ig] = eval_psp_density_core_fourier(element.psp, norm(G))
                 else
                     form_factors[iG, ig] = eval_psp_density_valence_fourier(element.psp, norm(G))
                 end
             end
        end
    end
    iG2ifnorm = 1:length(Gs) 
    return form_factors, iG2ifnorm
end

"""
Compute the macroscopic magnetization of the system.
Returns a vector of length 3 (Mx, My, Mz).
"""
function compute_macroscopic_magnetization(basis::PlaneWaveBasis, ρ::AbstractArray)
    # 1. Get volume of one grid point (dVol)
    #    Total Volume / Number of points
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)
    
    n_spin = size(ρ, 4)
    
    if n_spin == 1
        # Non-magnetic
        return [0.0, 0.0, 0.0]
        
    elseif n_spin == 2
        # Collinear: ρ is [Up, Down]
        # Magnetization density m(r) = ρ_up(r) - ρ_down(r)
        # Total Moment M = ∫ (ρ_up - ρ_down) dV
        mag_z = sum(ρ[:, :, :, 1] .- ρ[:, :, :, 2]) * dVol
        return [0.0, 0.0, mag_z]
        
    elseif n_spin == 4
        # Non-Collinear / SOC: ρ is [Total, Mx, My, Mz]
        # Total Moment M = ∫ (Mx, My, Mz) dV
        M_x = sum(ρ[:, :, :, 2]) * dVol
        M_y = sum(ρ[:, :, :, 3]) * dVol
        M_z = sum(ρ[:, :, :, 4]) * dVol
        return [M_x, M_y, M_z]
        
    else
        error("Unknown spin configuration with $n_spin components")
    end
end