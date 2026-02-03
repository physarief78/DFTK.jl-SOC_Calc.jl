using LinearAlgebra
using Interpolations: linear_interpolation
import PseudoPotentialIO: load_psp_file, UpfFile, Psp8File

struct PspUpf{T,I} <: NormConservingPsp
    ## From file
    Zion::Int          # Pseudo-atomic (valence) charge. UPF: `z_valence`
    lmax::Int          # Maximal angular momentum in the non-local part. UPF: `l_max`
    rgrid::Vector{T}   # Radial grid, can be linear or logarithmic. UPF: `PP_MESH/PP_R`
    drgrid::Vector{T}  # Radial grid derivative / integration factors. UPF: `PP_MESH/PP_RAB`
    vloc::Vector{T}    # Local part of the potential on the radial grid. UPF: `PP_LOCAL`
    # r^2 * β where β are Kleinman-Bylander non-local projectors on the radial grid.
    # UPF: `PP_NONLOCAL/PP_BETA.i`
    r2_projs::Vector{Vector{Vector{T}}}
    # Kleinman-Bylander energies. Stored per AM channel `h[l+1][i,j]`.
    # UPF: `PP_DIJ`
    h::Vector{Matrix{T}}
    
    # --- NEW FIELD: Relativistic Energies ---
    # h_so[l+1] contains [h(j=l-0.5), h(j=l+0.5)]
    h_so::Vector{Vector{Vector{T}}}
    # ----------------------------------------

    # Pseudo-wavefunctions on the radial grid. Used as projectors for PDOS
    # and DFT+U(+V), could be used for wavefunction initialization as well.
    # r^2 * χ where χ are pseudo-atomic wavefunctions on the radial grid.
    # UPF: `PP_PSWFC/PP_CHI.i`
    r2_pswfcs::Vector{Vector{Vector{T}}}
    # (UNUSED) Occupations of the pseudo-atomic wavefunctions.
    # UPF: `PP_PSWFC/PP_CHI.i['occupation']`
    pswfc_occs::Vector{Vector{T}}
    # (UNUSED) Energies of the pseudo-atomic wavefunctions.
    # UPF: `PP_PSWFC/PP_CHI.i['pseudo_energy']`
    pswfc_energies::Vector{Vector{T}}
    # Labels of the pseudo-atomic wavefunctions.
    # Used for projector selection in PDOS and DFT+U(+V).
    # UPF: `PP_PSWFC/PP_CHI.i['label']`
    pswfc_labels::Vector{Vector{String}}
    # 4πr^2 ρion where ρion is the pseudo-atomic (valence) charge density on the
    # radial grid. Can be used for charge density initialization.
    # UPF: `PP_RHOATOM`
    r2_ρion::Vector{T}
    # r^2 ρcore where ρcore is the atomic core charge density on the radial grid,
    # used for non-linear core correction.
    # UPF: `PP_NLCC`
    r2_ρcore::Vector{T}

    ## Precomputed for performance
    # (USED IN TESTS) Local potential interpolator, stored for performance.
    vloc_interp::I
    # (USED IN TESTS) Projector interpolators, stored for performance.
    r2_projs_interp::Vector{Vector{I}}
    # (USED IN TESTS) Valence charge density interpolator, stored for performance.
    r2_ρion_interp::I
    # (USED IN TESTS) Core charge density interpolator, stored for performance.
    r2_ρcore_interp::I

    ## Extras
    rcut::T              # Radial cutoff for all quantities except pswfc.
                         # Used to avoid some numerical issues encountered when
                         # integrating over the full radial mesh.
    ircut::Int           # Index of the radial cutoff.
    identifier::String   # String identifying the pseudopotential.
    description::String  # Descriptive string. UPF: `comment`
end

"""
    PspUpf(path[; identifier])

Construct a Unified Pseudopotential Format pseudopotential by reading a file.

Does not support:
- Bare Coulomb / all-electron potentials
- Semilocal potentials
- Ultrasoft potentials
- Projector-augmented wave potentials
- GIPAW reconstruction data
"""
function PspUpf(path::AbstractString; identifier=path, rcut=nothing)
    PspUpf(load_psp_file(path); identifier=identifier, rcut=rcut)
end

"""
    PspUpf(pseudo::Psp8File; identifier)

Construct a Unified Pseudopotential Format pseudopotential from a parsed psp8 file.
Internally, the pseudo is first converted to a `UpfFile` using `PseudoPotentialIO`.
"""
function PspUpf(pseudo::Psp8File; identifier, rcut=nothing)
    PspUpf(UpfFile(pseudo); identifier=identifier, rcut=rcut)
end

"""
    PspUpf(pseudo::UpfFile; identifier)

Construct a Unified Pseudopotential Format pseudopotential from a parsed upf file.
"""
function PspUpf(pseudo::UpfFile; identifier, rcut=nothing)
    unsupported = []
    # REMOVED: pseudo.header.has_so && push!(unsupported, "spin-orbit coupling")
    pseudo.header.pseudo_type == "SL"   && push!(unsupported, "semilocal potential")
    pseudo.header.pseudo_type == "US"   && push!(unsupported, "ultrasoft")
    pseudo.header.pseudo_type == "USPP" && push!(unsupported, "ultrasoft")
    pseudo.header.pseudo_type == "PAW"  && push!(unsupported, "projector-augmented wave")
    pseudo.header.has_gipaw             && push!(unsupported, "gipaw data")
    pseudo.header.pseudo_type == "1/r"  && push!(unsupported, "Coulomb")
    length(unsupported) > 0 && error("Pseudopotential contains the following unsupported" *
                                     " features/quantities: $(join(unsupported, ","))")

    Zion        = Int(pseudo.header.z_valence)
    rgrid       = pseudo.mesh.r
    drgrid      = pseudo.mesh.rab
    lmax        = pseudo.header.l_max
    vloc        = pseudo.local_ ./ 2  # (Ry -> Ha)
    description = something(pseudo.header.comment, "")

    # Ensure rcut is at most the end of the rgrid.
    rcut = isnothing(rcut) ? last(rgrid) : min(rcut, last(rgrid))
    ircut = findfirst(>=(rcut), rgrid)

    r2_projs = map(0:lmax) do l
        betas_l = filter(beta -> beta.angular_momentum == l, pseudo.nonlocal.betas)
        map(betas_l) do beta_li
            r_beta_ha = beta_li.beta[1:beta_li.cutoff_radius_index] ./ 2  # Ry -> Ha
            rgrid[1:length(r_beta_ha)] .* r_beta_ha  # rβ -> r²β
        end
    end
    h = map(0:lmax) do l
        mask_l = findall(beta -> beta.angular_momentum == l, pseudo.nonlocal.betas)
        pseudo.nonlocal.dij[mask_l, mask_l] .* 2  # 1/Ry -> 1/Ha
    end

    # CHANGED --- PARSE SOC DATA ---
    T = eltype(rgrid)
    h_so = [Vector{Vector{T}}() for _ in 0:lmax]
    
    # Check if we can access the j information.
    # If PseudoPotentialIO doesn't expose it, we might need to inspect the raw dictionary
    # or rely on the standard ordering: first set of betas for an l are j-0.5, second set are j+0.5.
    
    if pseudo.header.has_so
        # Initialize slots
        for l in 0:lmax
            push!(h_so[l+1], T[]) # j-
            push!(h_so[l+1], T[]) # j+
        end

        # Group betas by angular momentum l
        for l in 0:lmax
            betas_l = filter(beta -> beta.angular_momentum == l, pseudo.nonlocal.betas)
            
            # For l=0, j must be 0.5 (only one channel)
            if l == 0
                for beta in betas_l
                    d_val = pseudo.nonlocal.dij[beta.index, beta.index] * 2 # Ry -> Ha
                    # l=0 has j=1/2. Conventionally put in the "plus" slot (index 2) or handle as special.
                    # Let's put it in index 1 (minus slot) since |0 - 1/2| = 0.5.
                    push!(h_so[1][1], d_val) 
                end
            else
                # For l > 0, we expect two sets of projectors: j = l-0.5 and j = l+0.5
                # Standard UPF ordering: The first N/2 are j-0.5, the next N/2 are j+0.5
                n_betas = length(betas_l)
                if isodd(n_betas)
                     @warn "Odd number of projectors for l=$l in SOC pseudopotential. Assumption of split j-shells might fail."
                end
                
                # Try to access total_angular_momentum if available (using getfield to avoid error if missing)
                # If not, use the index splitting heuristic.
                
                midpoint = div(n_betas, 2)
                
                for (i, beta) in enumerate(betas_l)
                    d_val = pseudo.nonlocal.dij[beta.index, beta.index] * 2
                    
                    # Try to get j safely
                    j_val = -1.0
                    try
                        j_val = getproperty(beta, :total_angular_momentum)
                    catch
                        # Field missing, fallback to heuristic
                        j_val = (i <= midpoint) ? (l - 0.5) : (l + 0.5)
                    end
                    
                    if abs(j_val - (l - 0.5)) < 1e-4
                        push!(h_so[l+1][1], d_val)
                    elseif abs(j_val - (l + 0.5)) < 1e-4
                        push!(h_so[l+1][2], d_val)
                    else
                        # Heuristic fallback if j_val was garbage
                         if i <= midpoint
                             push!(h_so[l+1][1], d_val)
                         else
                             push!(h_so[l+1][2], d_val)
                         end
                    end
                end
            end
        end
    end
    # ----------------------

    r2_pswfcs = [Vector{Float64}[] for _ = 0:lmax]
    pswfc_occs     = [Float64[]    for _ = 0:lmax]
    pswfc_energies = [Float64[]    for _ = 0:lmax]
    pswfc_labels   = [String[]     for _ = 0:lmax]
    for l = 0:lmax
        pswfcs_l = filter(χ -> χ.l == l, pseudo.pswfc)
        for pswfc_li in pswfcs_l
            push!(r2_pswfcs[l+1], rgrid .* pswfc_li.chi)  # rχ -> r²χ
            push!(pswfc_occs[l+1], pswfc_li.occupation)
            push!(pswfc_energies[l+1], pswfc_li.pseudo_energy)
            push!(pswfc_labels[l+1], pswfc_li.label)
        end
    end

    r2_ρion = pseudo.rhoatom ./ (4π)
    r2_ρcore = rgrid .^ 2 .* (@something pseudo.nlcc zeros(length(rgrid)))

    vloc_interp = linear_interpolation((rgrid,), vloc)
    r2_projs_interp = map(r2_projs) do r2_projs_l
        map(proj -> linear_interpolation((rgrid[1:length(proj)],), proj), r2_projs_l)
    end
    r2_ρion_interp = linear_interpolation((rgrid,), r2_ρion)
    r2_ρcore_interp = linear_interpolation((rgrid,), r2_ρcore)

    PspUpf{eltype(rgrid),typeof(vloc_interp)}(
        Zion, lmax, rgrid, drgrid,
        vloc, r2_projs, h, h_so, # Added h_so
        r2_pswfcs, pswfc_occs, pswfc_energies, pswfc_labels,
        r2_ρion, r2_ρcore,
        vloc_interp, r2_projs_interp, r2_ρion_interp, r2_ρcore_interp,
        rcut, ircut, identifier, description
    )
end

charge_ionic(psp::PspUpf) = psp.Zion
has_valence_density(psp::PspUpf) = !all(iszero, psp.r2_ρion)
has_core_density(psp::PspUpf) = !all(iszero, psp.r2_ρcore)

function eval_psp_projector_real(psp::PspUpf, i, l, r::T)::T where {T<:Real}
    psp.r2_projs_interp[l+1][i](r) / r^2  # TODO if r is below a threshold, return zero
end

function eval_psp_projector_fourier(psp::PspUpf, i, l, p::T)::T where {T<:Real}
    ircut_proj = min(psp.ircut, length(psp.r2_projs[l+1][i]))
    rgrid = @view psp.rgrid[1:ircut_proj]
    r2_proj = @view psp.r2_projs[l+1][i][1:ircut_proj]
    hankel(rgrid, r2_proj, l, p)
end

count_n_pswfc_radial(psp::PspUpf, l) = length(psp.r2_pswfcs[l+1])

pswfc_label(psp::PspUpf, i, l) = psp.pswfc_labels[l+1][i]

function eval_psp_pswfc_real(psp::PspUpf, i, l, r::T)::T where {T<:Real}
    psp.r2_pswfcs_interp[l+1][i](r) / r^2  # TODO if r is below a threshold, return zero
end

function eval_psp_pswfc_fourier(psp::PspUpf, i, l, p::T)::T where {T<:Real}
    return hankel(psp.rgrid, psp.r2_pswfcs[l+1][i], l, p)
end

eval_psp_local_real(psp::PspUpf, r::T) where {T<:Real} = psp.vloc_interp(r)

function eval_psp_local_fourier(psp::PspUpf, p::T)::T where {T<:Real}
    rgrid = @view psp.rgrid[1:psp.ircut]
    vloc  = @view psp.vloc[1:psp.ircut]
    I = simpson(rgrid) do i, r
         r * (r * vloc[i] - -psp.Zion * erf(r)) * sphericalbesselj_fast(0, p * r)
    end
    4T(π) * (I + -psp.Zion / p^2 * exp(-p^2 / T(4)))
end

function eval_psp_density_valence_real(psp::PspUpf, r::T) where {T<:Real}
    psp.r2_ρion_interp(r) / r^2  # TODO if r is below a threshold, return zero
end

function eval_psp_density_valence_fourier(psp::PspUpf, p::T) where {T<:Real}
    rgrid = @view psp.rgrid[1:psp.ircut]
    r2_ρion = @view psp.r2_ρion[1:psp.ircut]
    return hankel(rgrid, r2_ρion, 0, p)
end

function eval_psp_density_core_real(psp::PspUpf, r::T) where {T<:Real}
    psp.r2_ρcore_interp(r) / r^2  # TODO if r is below a threshold, return zero
end

function eval_psp_density_core_fourier(psp::PspUpf, p::T) where {T<:Real}
    rgrid = @view psp.rgrid[1:psp.ircut]
    r2_ρcore = @view psp.r2_ρcore[1:psp.ircut]
    return hankel(rgrid, r2_ρcore, 0, p)
end

function eval_psp_energy_correction(T, psp::PspUpf)
    rgrid = @view psp.rgrid[1:psp.ircut]
    vloc = @view psp.vloc[1:psp.ircut]
    4T(π) * simpson(rgrid) do i, r
        r * (r * vloc[i] - -psp.Zion)
    end
end

# === NEW SOC INTERFACE IMPLEMENTATION ===

has_spin_orbit(psp::PspUpf) = !isempty(psp.h_so) && !all(isempty, psp.h_so)

function get_soc_coupling(psp::PspUpf, l)
    if l > psp.lmax || isempty(psp.h_so)
        return nothing
    end
    # Returns [h_j_minus, h_j_plus]
    return psp.h_so[l+1]
end