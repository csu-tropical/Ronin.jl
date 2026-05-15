push!(LOAD_PATH,"../src/")
using Documenter, Ronin

makedocs(sitename="Ronin.jl",
        modules= [Ronin],
        pages = [
            "Home"                      => "index.md",
            "Concepts"                  => "concepts.md",
            "Workflow Guide"            => "workflow.md",
            "Convolution Feature Mode"  => "convolution.md",
            "Legacy Hand-Tuned Mode"    => "legacy.md",
            "Choosing a QC Entry Point" => "entrypoints.md",
            "API Reference"             => "api.md",
        ],
        format = Documenter.HTML(prettyurls = false),
        checkdocs=:exports,
        checkdocs_ignored_modules=[Ronin.DecisionTree])

deploydocs(;
    repo="github.com/csu-tropical/Ronin.jl.git",
    devbranch="main"
)
