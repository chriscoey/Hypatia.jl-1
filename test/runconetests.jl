#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
run barrier tests
=#

using Test
using Printf
import Hypatia.Cones
include(joinpath(@__DIR__, "cone.jl"))

function cone_types(T::Type{<:Real})
    cones_T = [
        Cones.Nonnegative{T},
        Cones.PosSemidefTri{T, T},
        Cones.PosSemidefTri{T, Complex{T}},
        Cones.DoublyNonnegativeTri{T},
        Cones.PosSemidefTriSparse{Cones.PSDSparseDense, T, T},
        Cones.PosSemidefTriSparse{Cones.PSDSparseDense, T, Complex{T}},
        Cones.LinMatrixIneq{T, T},
        Cones.LinMatrixIneq{T, Complex{T}},
        Cones.EpiNormInf{T, T},
        Cones.EpiNormInf{T, Complex{T}},
        Cones.EpiNormEucl{T},
        Cones.EpiPerSquare{T},
        Cones.EpiNormSpectralTri{T, T},
        Cones.EpiNormSpectralTri{T, Complex{T}},
        Cones.EpiNormSpectral{T, T},
        Cones.EpiNormSpectral{T, Complex{T}},
        Cones.MatrixEpiPerSquare{T, T},
        Cones.MatrixEpiPerSquare{T, Complex{T}},
        Cones.GeneralizedPower{T},
        Cones.HypoPowerMean{T},
        Cones.HypoGeoMean{T},
        Cones.HypoRootdetTri{T, T},
        Cones.HypoRootdetTri{T, Complex{T}},
        Cones.HypoPerLog{T},
        Cones.HypoPerLogdetTri{T, T},
        Cones.HypoPerLogdetTri{T, Complex{T}},
        Cones.EpiPerSepSpectral{Cones.VectorCSqr{T}, T},
        Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, T}, T},
        Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, Complex{T}}, T},
        Cones.EpiRelEntropy{T},
        Cones.EpiTrRelEntropyTri{T, T},
        Cones.EpiTrRelEntropyTri{T, Complex{T}},
        Cones.WSOSInterpNonnegative{T, T},
        Cones.WSOSInterpNonnegative{T, Complex{T}},
        Cones.WSOSInterpPosSemidefTri{T},
        Cones.WSOSInterpEpiNormEucl{T},
        Cones.WSOSInterpEpiNormOne{T},
    ]

    if Hypatia.USE_GPL_LIBS && T <: LinearAlgebra.BlasReal
        append!(
            cones_T,
            [
                Cones.PosSemidefTriSparse{Cones.PSDSparseCholmod, T, T},
                Cones.PosSemidefTriSparse{Cones.PSDSparseCholmod, T, Complex{T}},
            ],
        )
    end

    return cones_T
end

sep_spectral_funs = [
    Cones.NegLogSSF(),
    Cones.NegEntropySSF(),
    Cones.NegSqrtSSF(),
    Cones.NegPower01SSF(3 // 10),
    Cones.Power12SSF(1.5),
]

@testset "cone tests" begin
    println("starting oracle tests")
    @testset "oracle tests" begin
        real_types = [
            Float64,
            # Float32,
            # BigFloat,
        ]
        @testset "$cone" for T in real_types, cone in cone_types(T)
            println("$cone")
            test_time = @elapsed test_oracles(cone)
            @printf("%8.2e seconds\n", test_time)
        end
    end

    # eigenvalues that are close relative to their size or only in absolute terms
    @testset "divided differences" begin
        Random.seed!(1)
        rt2 = sqrt(2.0)
        diag_svec(λ) = Cones.smat_to_svec!(zeros(6), diagm(λ), rt2)

        for λ in ([1e-9, 4e-9, 1], [1, 1 + 2e-8, 2]), h in sep_spectral_funs
            point = vcat(Cones.h_val(λ, h) + 1, 1, diag_svec(λ))
            csqr(T) = Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, T}, T}(h, 3)
            test_oracles_vs_bigfloat(csqr, point, randn(8), 1e-12)
        end

        (λ_V, λ_W) = (1e-6 * [1, 4, 2], 1e-6 * [2, 1, 3])
        u = sum(w * log(w / v) for (v, w) in zip(λ_V, λ_W)) + 1e-6
        point = vcat(u, diag_svec(λ_V), diag_svec(λ_W))
        trrelent(T) = Cones.EpiTrRelEntropyTri{T, T}(13)
        test_oracles_vs_bigfloat(trrelent, point, randn(13), 1e-12)
    end

    # println("\nstarting barrier tests")
    # @testset "barrier tests" begin
    # real_types = [
    #     Float64,
    #     # Float32,
    #     # BigFloat,
    #     ]
    # @testset "$cone" for T in real_types, cone in cone_types(T)
    #     println("$cone")
    #     test_time = @elapsed test_barrier(cone)
    #     @printf("%8.2e seconds\n", test_time)
    # end
    # end
    #
    # println("\nstarting time/allocation measurements")
    # @testset "allocation tests" begin
    # real_types = [
    #     Float64,
    #     # Float32,
    #     # BigFloat,
    #     ]
    # @testset "$cone" for T in real_types, cone in cone_types(T)
    #     println("\n$cone")
    #     test_time = @elapsed show_time_alloc(cone)
    #     @printf("%8.2e seconds\n", test_time)
    # end
    # println()
    # end
end;
