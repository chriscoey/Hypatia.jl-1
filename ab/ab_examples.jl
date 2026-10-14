# solve example instances and write one CSV row per instance
# usage: julia --project=HYP/examples ab_examples.jl HYP OUT.csv FILTER
# FILTER: "matspec" (instances using MatrixCSqr or EpiTrRelEntropyTri cones) or "all"
const HYP = abspath(ARGS[1])
const OUT = abspath(ARGS[2])
const FILTER = ARGS[3]
include(joinpath(HYP, "examples", "Examples.jl"))
using Printf

const DIRECT = ("relentrentanglement", "entanglementassisted", "nearestcorrelation")
# formulations that use EpiPerSepSpectral with MatrixCSqr (not the extended or logdet ones)
is_matspec(x) =
    x isa Union{Examples.MatSepSpecPrim, Examples.MatSepSpecDual} && !(x isa Examples.MatLogdetCone)
affected(ex, inst_data) =
    FILTER == "all" || ex in DIRECT || any(is_matspec, inst_data)

const OPTS = (verbose = false, default_tol_relax = 10, iter_limit = 250, time_limit = 600.0)
const FIELDS = (:status, :iters, :solve_time, :primal_obj, :dual_obj, :x_viol, :y_viol, :z_viol, :compl, :rel_obj_diff)

open(io -> println(io, join(("example", "set", "inst", FIELDS..., "error"), ",")), OUT, "w")

function run_one(ex, set, ex_type, args)
    inst_str = replace(string(args), "," => ";", "\"" => "")
    vals = Any[missing for _ in FIELDS]
    err = ""
    try
        r = Examples.run_instance(ex_type{Float64}, args...; default_options = OPTS, test = false, verbose = false)
        vals = Any[get(r, f, missing) for f in FIELDS]
    catch e
        err = replace(first(sprint(showerror, e), 120), "," => ";", "\n" => " ")
    end
    open(io -> println(io, join((ex, set, inst_str, vals..., err), ",")), OUT, "a")
    @printf("%-22s %-8s %-50s %s it=%s t=%s %s\n", ex, set, first(inst_str, 50), vals[1], vals[2], vals[3], err)
    flush(stdout)
end

jobs = []
for (ex, (ex_type, insts)) in Examples.get_test_instances()["JuMP"]
    for set in ("minimal", "fast", "various")
        haskey(insts, set) || continue
        for inst in insts[set]
            affected(ex, inst[1]) && push!(jobs, (ex, set, ex_type, inst))
        end
    end
end
for ex in ("experimentdesign", "covarianceest", "centralpolymat")
    ex_type = Examples.get_test_instances("JuMP", ex)[1]
    for (set, (ext, inst_lists)) in Examples.get_benchmark_instances("JuMP", ex)[2]
        for insts in inst_lists
            sel = [i for i in insts if affected(ex, i)]
            for i in first(sel, 5)
                push!(jobs, (ex, "bench_" * set, ex_type, (i, ext)))
            end
        end
    end
end
println(length(jobs), " instances")
get(ENV, "AB_DRY", "") == "1" && (foreach(j -> println(j[1], " ", j[2], " ", j[4]), jobs); exit())
haskey(ENV, "AB_LIMIT") && (jobs = jobs[1:parse(Int, ENV["AB_LIMIT"])])

# compile the solve path before anything is timed
run_one(jobs[1][1], "warmup", jobs[1][3], jobs[1][4])
for (ex, set, ex_type, args) in jobs
    run_one(ex, set, ex_type, args)
end
