#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
run MOI tests
=#

using Test
using LinearAlgebra
import MathOptInterface as MOI
import LowRankOpt as LRO
import Hypatia
include(joinpath(@__DIR__, "moicones.jl"))

@testset "MathOptInterface wrapper tests" begin
    @testset "MOI cone tests" begin
        println("starting MOI wrapper cone tests")
        real_types = [Float64, BigFloat]
        for T in real_types
            @testset "$T" begin
                println(T, " ...")
                test_moi_cones(T)
            end
        end
    end

    @testset "Supports $T" for T in [Float64, BigFloat]
        model = Hypatia.Optimizer{T}()
        for S in [
            Hypatia._PrimalRankOnePSD{T, Vector{T}}
            Hypatia._DualRankOnePSD{T, Vector{T}}
        ]
            @test MOI.supports_constraint(model, MOI.VectorAffineFunction{T}, S)
        end
    end

    @testset "tall $S with multi-term entries" for (S, f) in [
        (MOI.NormSpectralCone, opnorm),
        (MOI.NormNuclearCone, W -> sum(svdvals(W))),
    ]
        # some entries of W have two terms and one has none, so term indices
        # differ from output indices
        x0 = [1.0, 2.0, 3.0]
        W0 = [x0[1]+x0[2] x0[3]; 0.5*x0[1] 2*x0[2]; 4.0 -x0[1]]
        src = MOI.Utilities.Model{Float64}()
        t = MOI.add_variable(src)
        x = MOI.add_variables(src, 3)
        fix_x = MOI.Utilities.vectorize([1.0 * x[i] - x0[i] for i in 1:3])
        MOI.add_constraint(src, fix_x, MOI.Zeros(3))
        W = [
            1.0*x[1]+1.0*x[2] 1.0*x[3]
            0.5*x[1] 2.0*x[2]
            zero(MOI.ScalarAffineFunction{Float64})+4.0 -1.0*x[1]
        ]
        func = MOI.Utilities.vectorize(vcat(1.0 * t, vec(W)))
        MOI.add_constraint(src, func, S(3, 2))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        obj = 1.0 * t
        MOI.set(src, MOI.ObjectiveFunction{typeof(obj)}(), obj)
        model = Hypatia.Optimizer()
        MOI.set(model, MOI.Silent(), true)
        MOI.copy_to(model, src)
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ f(W0) rtol = 1e-6
    end

    # real types, tolerances, and tests to include for MOI.Test tests
    test_T = [
        (Float64, 2 * sqrt(sqrt(eps())), 4, String[], String[]),
        (
            BigFloat,
            2 * eps(BigFloat)^0.15,
            1,
            String["test_linear", "test_conic", "test_modification"],
            String["test_linear_INFEASIBLE_2",],
        ),
    ]

    @testset "MOI.Test tests: $T" for (T, tol_test, tol_relax, includes, excludes) in test_T
        println("\nstarting MOI.Test tests: $T")
        model = MOI.Bridges.full_bridge_optimizer(
            MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{T}()),
                Hypatia.Optimizer{T}(; default_tol_relax = tol_relax),
            ),
            T,
        )
        MOI.set(model, MOI.Silent(), true)
        config = MOI.Test.Config(
            T,
            atol = tol_test,
            rtol = tol_test,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ObjectiveBound,
                MOI.SolverVersion,
            ],
        )
        MOI.Test.runtests(
            model,
            config,
            include = includes,
            exclude = vcat(
                excludes,
                String[
                    # TODO(odow): unexpected failure, probably in the bridge layer
                    "test_model_UpperBoundAlreadySet",
                    "test_model_LowerBoundAlreadySet",
                ],
            ),
        )
        MOI.Test.runtests(model, config, test_module = LRO.Test)
    end
end;
