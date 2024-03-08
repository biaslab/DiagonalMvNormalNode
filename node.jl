using RxInfer
using ReactiveMP
import ReactiveMP: NodeInterface, IndexedNodeInterface, Marginalisation, FactorNodeCreationOptions, connectvariable!, messageout, getlastindex, getpipeline, collect_meta, collect_pipeline, collect_factorisation, factorisation, metadata, name, IncludeAll, FactorNodeLocalMarginal, AbstractNodeFunctionalDependenciesPipeline
import ReactiveMP: setmessagein!, functionalform, interfaces, factorisation, localmarginals, localmarginalnames, metadata, getpipeline, setmarginal!, getmarginal!, functional_dependencies, get_messages_observable, get_marginals_observable, score, as_node_symbol, as_node_functional_form, ValidNodeFunctionalForm, Stochastic, combineLatest, map_to, ManyOf, PushNew, map, combineLatestUpdates, of, IncludeAll, getmarginal, connectedvar, Val
# Normal Diagonal precision Functional Form
struct MvNormalDiagonalPrecision{N} end

ReactiveMP.as_node_symbol(::Type{<:MvNormalDiagonalPrecision}) = :MvNormalDiagonalPrecision

# Special node
# Generic FactorNode implementation does not work with dynamic number of inputs
# We need to reimplement the following set of functions
# functionalform(factornode::FactorNode)          
# sdtype(factornode::FactorNode)                 
# interfaces(factornode::FactorNode)              
# factorisation(factornode::FactorNode)           
# localmarginals(factornode::FactorNode)          
# localmarginalnames(factornode::FactorNode)      
# metadata(factornode::FactorNode)                
# get_pipeline_stages(factornode::FactorNode)       
#
# setmarginal!(factornode::FactorNode, cname::Symbol, marginal)
# getmarginal!(factornode::FactorNode, localmarginal::FactorNodeLocalMarginal)
#
# functional_dependencies(factornode::FactorNode, iindex::Int)
# get_messages_observable(factornode, message_dependencies)
# get_marginals_observable(factornode, marginal_dependencies)
#
# score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::AbstractFactorNode, scheduler) where T
#
# Base.show

const MvNormalDiagonalPrecisionNodeFactorisationSupport = Union{MeanField, FullFactorisation}

struct MvNormalDiagonalPrecisionNode{N, F <: MvNormalDiagonalPrecisionNodeFactorisationSupport, M, P} <: ReactiveMP.AbstractFactorNode
    factorisation::F

    # Interfaces
    out    :: NodeInterface
    mean   :: NodeInterface
    precs  :: NTuple{N, IndexedNodeInterface}

    meta     :: M
    pipeline :: P
end

functionalform(factornode::MvNormalDiagonalPrecisionNode{N}) where {N} = MvNormalDiagonalPrecision{N}
sdtype(factornode::MvNormalDiagonalPrecisionNode)                      = Stochastic()
ReactiveMP.interfaces(factornode::MvNormalDiagonalPrecisionNode)                  = (factornode.out, factornode.mean, factornode.precs...)
factorisation(factornode::MvNormalDiagonalPrecisionNode)               = factornode.factorisation
localmarginals(factornode::MvNormalDiagonalPrecisionNode)              = error("localmarginals() function is not implemented for NormalMixtureNode")
localmarginalnames(factornode::MvNormalDiagonalPrecisionNode)          = error("localmarginalnames() function is not implemented for NormalMixtureNode")
metadata(factornode::MvNormalDiagonalPrecisionNode)                    = factornode.meta
getpipeline(factornode::MvNormalDiagonalPrecisionNode)                 = factornode.pipeline

setmarginal!(factornode::MvNormalDiagonalPrecisionNode, cname::Symbol, marginal)                = error("setmarginal() function is not implemented for NormalMixtureNode")
getmarginal!(factornode::MvNormalDiagonalPrecisionNode, localmarginal::FactorNodeLocalMarginal) = error("getmarginal() function is not implemented for NormalMixtureNode")

function interfaceindex(factornode::MvNormalDiagonalPrecisionNode, iname::Symbol)
    if iname === :out
        return 1
    elseif iname === :m
        return 2
    elseif iname === :p
        return 3
    else
        error("Unknown interface ':$(iname)' for the [ $(functionalform(factornode)) ] node")
    end
end

struct MvNormalDiagonalPrecisionNodeFunctionalDependencies <: AbstractNodeFunctionalDependenciesPipeline end

ReactiveMP.default_functional_dependencies_pipeline(::Type{<:MvNormalDiagonalPrecision}) = MvNormalDiagonalPrecisionNodeFunctionalDependencies()

function ReactiveMP.functional_dependencies(::MvNormalDiagonalPrecisionNodeFunctionalDependencies, factornode::MvNormalDiagonalPrecisionNode{N, F}, iindex::Int) where {N, F}
    message_dependencies = ()

    marginal_dependencies = if iindex === 1
        (factornode.mean, factornode.precs)
    elseif iindex === 2
        (factornode.out, factornode.precs)
    elseif 2 < iindex <= N + 2
        (factornode.out, factornode.mean)
    else
        error("Bad index in functional_dependencies for NormalMixtureNode")
    end

    return message_dependencies, marginal_dependencies
end

function ReactiveMP.get_messages_observable(factornode::MvNormalDiagonalPrecisionNode{N, F}, message_dependencies::Tuple{}) where {N, F}
    return nothing, of(nothing)
end

function ReactiveMP.get_marginals_observable(
    factornode::MvNormalDiagonalPrecisionNode{N, F}, marginal_dependencies::Tuple{NodeInterface, NTuple{N, IndexedNodeInterface}}
) where {N, F}
    meaninterface   = marginal_dependencies[1]
    precsinterfaces = marginal_dependencies[2]

    marginal_names = Val{(name(meaninterface), name(precsinterfaces[1]))}()
    marginals_observable =
        combineLatest(
            (
                getmarginal(connectedvar(meaninterface), IncludeAll()),
                combineLatest(map((prec) -> getmarginal(connectedvar(prec), IncludeAll()), reverse(precsinterfaces)), PushNew())
            ),
            PushNew()
        ) |> map_to((
            getmarginal(connectedvar(meaninterface), IncludeAll()),
            ManyOf(map((prec) -> getmarginal(connectedvar(prec), IncludeAll()), precsinterfaces))
        ))

    return marginal_names, marginals_observable
end

function ReactiveMP.get_marginals_observable(factornode::MvNormalDiagonalPrecisionNode{N, F}, marginal_dependencies::Tuple{NodeInterface, NodeInterface}) where {N, F}
    outinterface    = marginal_dependencies[1]
    varinterface    = marginal_dependencies[2]

    marginal_names       = Val{(name(outinterface), name(varinterface))}()
    marginals_observable = combineLatestUpdates((getmarginal(connectedvar(outinterface), IncludeAll()), getmarginal(connectedvar(varinterface), IncludeAll())), PushNew())

    return marginal_names, marginals_observable
end

as_node_functional_form(::Type{<:MvNormalDiagonalPrecision}) = ValidNodeFunctionalForm()

# Node creation related functions

sdtype(::Type{<:MvNormalDiagonalPrecision}) = Stochastic()

collect_factorisation(::Type{<:MvNormalDiagonalPrecision}, ::Nothing)                = MeanField()
collect_factorisation(::Type{<:MvNormalDiagonalPrecision}, factorisation::MeanField) = factorisation
collect_factorisation(::Type{<:MvNormalDiagonalPrecision}, factorisation::FullFactorisation) = factorisation
collect_factorisation(::Type{<:MvNormalDiagonalPrecision}, factorisation::Any)       = __normal_mixture_incompatible_factorisation_error()

function collect_factorisation(::Type{<:MvNormalDiagonalPrecision{N}}, factorisation::NTuple{R, Tuple{<:Integer}}) where {N, R}
    # 2 * (m, w) + s + out, equivalent to MeanField 
    return (R === N + 2) ? MeanField() : __normal_mixture_incompatible_factorisation_error()
end

__normal_mixture_incompatible_factorisation_error() =
    error("`MvNormalDiagonalPrecision` supports only following global factorisations: [ $(MvNormalDiagonalPrecisionNodeFactorisationSupport) ] or manually set to equivalent via constraints")

function ReactiveMP.make_node(::Type{<:MvNormalDiagonalPrecision{N}}, options::FactorNodeCreationOptions) where {N}
    @assert N >= 2 "`MvNormalDiagonalPrecision` requires at least two precisions on input"
    out    = NodeInterface(:out, Marginalisation())
    mean   = NodeInterface(:m, Marginalisation())
    precs  = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:p, Marginalisation())), N)

    _factorisation = collect_factorisation(MvNormalDiagonalPrecision{N}, factorisation(options))
    _meta = collect_meta(MvNormalDiagonalPrecision{N}, metadata(options))
    _pipeline = collect_pipeline(MvNormalDiagonalPrecision{N}, getpipeline(options))

    @assert typeof(_factorisation) <: MvNormalDiagonalPrecisionNodeFactorisationSupport "`MvNormalDiagonalPrecision` supports only following factorisations: [ $(MvNormalDiagonalPrecisionNodeFactorisationSupport) ]"

    F = typeof(_factorisation)
    M = typeof(_meta)
    P = typeof(_pipeline)

    return MvNormalDiagonalPrecisionNode{N, F, M, P}(_factorisation, out, mean, precs, _meta, _pipeline)
end

function ReactiveMP.make_node(
    ::Type{<:MvNormalDiagonalPrecision},
    options::FactorNodeCreationOptions,
    out::AbstractVariable,
    mean::AbstractVariable,
    precs::NTuple{N, AbstractVariable}
) where {N}
    node = make_node(MvNormalDiagonalPrecision{N}, options)

    # out
    out_index = getlastindex(out)
    connectvariable!(node.out, out, out_index)
    setmessagein!(out, out_index, messageout(node.out))

    # mean
    mean_index = getlastindex(mean)
    connectvariable!(node.mean, mean, mean_index)
    setmessagein!(mean, mean_index, messageout(node.mean))

    # precs
    foreach(zip(node.precs, precs)) do (pinterface, pvar)
        prec_index = getlastindex(pvar)
        connectvariable!(pinterface, pvar, prec_index)
        setmessagein!(pvar, prec_index, messageout(pinterface))
    end

    return node
end
