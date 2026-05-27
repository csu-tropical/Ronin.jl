# Choosing a QC Entry Point

Three functions can apply a trained model to radar data. They differ in **what
you hand them** and **what workflow they fit**. All three honor
`config.task_mode` (convolution or legacy) and the multi-pass cascade. All
three produce identical QC results for the same model and data — the choice is
about control flow and what you get back, not accuracy.

## `QC_scan(config::ModelConfig)` — the default entry point

Config-and-directory driven. It reads `config.input_path` (a directory or a
single CfRadial) and internally calls
`composite_prediction(config; QC_mode=true)`, writing QC'd fields
(`<var><QC_SUFFIX>` for each `config.VARS_TO_QC`) back into the files. Use this
for the normal "I trained a model, now clean this dataset" case. This is what
`workflow/06_qc.jl` uses. **Start here** unless you have a specific reason not
to.

## `composite_prediction(config; …)` — the analysis / evaluation engine

Same cascade inference as `QC_scan`, but it **returns**
`(predictions, verification, indexers, pass_probs)` and only writes QC'd fields
if you pass `QC_mode=true`. Reach for this when you want the predictions and
probabilities in memory — building metrics, plotting met-prob distributions,
comparing thresholds, or any programmatic analysis where you don't (yet) want
to mutate the CfRadials. It also backs [`run_evaluation`](@ref). Pass
`skip_existing_met_probs=true` to reuse previously written `met_prob_pass_<i>`
fields instead of recomputing features (much faster for re-scoring).

## `composite_QC(config, files::Vector{String}[, models])` — the streaming / operational entry point

Driven by an **explicit vector of file paths**, not `config.input_path`. It
runs each cascade pass file-by-file, writing `met_prob_pass_<i>` and
`mask_pass_<i+1>` between passes, then writes QC'd fields. The 2-arg form loads
the random forests internally from `config.model_output_paths`; the 3-arg form
takes preloaded `models` (load once, QC many files). This is the right tool for
realtime / aircraft-side pipelines and any driver that already has a file list
and a `ModelConfig` reloaded via
`load_config("model_config_<EXPERIMENT_NAME>.jld2")`. See
`REALTIME/Ronin_realtime.jl` for a worked operational example.

## Decision shortcut

- Cleaning a directory you've pointed `config.input_path` at →
  `QC_scan(config)`.
- Need predictions/probabilities in memory, or computing metrics →
  `composite_prediction(config; …)` (or `run_evaluation`).
- Operational/streaming loop over an explicit file list, models loaded once →
  `composite_QC(config, files, models)`.
