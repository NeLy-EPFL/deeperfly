<p align="center">
  <img src="assets/logo.svg" alt="deeperfly logo" width="180">
</p>

# deeperfly

Markerless 3D pose estimation of tethered *Drosophila* from a multi-camera rig.
`deeperfly` estimates camera parameters and 2D/3D keypoint locations from
behavioral recordings through one linear pipeline: **2D pose → bundle adjustment
→ triangulation → smoothing → corrections → joint angles → visualization**.

The detector is **dense**: it predicts all 38 points of the `fly38` skeleton — six
5-point legs, two antennae, the neck and a 5-point dorsal-midline abdomen chain — in
**every** view, so a contralateral joint arrives as a prediction to correct rather than a
gap to fill. The packaged config describes an 8-camera rig, and a run
[narrows itself](getting-started.md#4-run-the-pipeline) to the footage it actually finds.

It is both a command-line tool and a Python library, and a modern rewrite of
[DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D),
[DeepFly2D](https://github.com/NeLy-EPFL/DeepFly2D) and
[PyBundleAdjustment](https://github.com/semihgunel/PyBundleAdjustment).

## Install

```bash
uv tool install git+https://github.com/NeLy-EPFL/deeperfly --python 3.13 --torch-backend=auto
```

## Run

Nothing downloads at run time: the detector is trained per project, so a checkpoint has
to be on this machine first and `$DEEPERFLY_MODELS` has to say where
([getting started, step 2](getting-started.md#2-get-the-detector-weights)).

```bash
export DEEPERFLY_MODELS=/path/to/models                      # where the checkpoints live
deeperfly doctor                                             # check the install, incl. weights
deeperfly init config.toml                                   # write a config (edit if needed)
deeperfly run examples/data/ -c config.toml                  # 2D -> 3D -> video
deeperfly inspect examples/data/deeperfly_outputs/results.h5 # summarize the result
```

Every stage runs by default except `pictorial_structures`; the inverse-kinematics stage
needs the optional `ik` extra and skips with the reason logged when it is absent.

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting started](getting-started.md)** — run the
  bundled example end to end, from install to a rendered 3D video.
- :material-console: **[CLI usage](guides/cli.md)** — every command and flag.
- :material-file-cog: **[Writing configs](guides/configuration.md)** — the
  `config.toml`, section by section.
- :material-camera-iris: **[The dense-38 detectors](explanation/detectors.md)** — the
  two detector classes, `mvt` and `hrnet`, and how a config selects one.
- :material-language-python: **[Library API](guides/library.md)** — use the
  pipeline, bundle adjustment and geometry from Python.
- :material-sitemap: **[How it works](explanation/pipeline.md)** — the pipeline
  stage by stage, plus the [conventions](explanation/conventions.md) the whole
  package shares.
- :material-book-open-variant: **Reference** — the complete
  [configuration](reference/configuration.md),
  [output format](reference/output-format.md) and
  [library API](reference/api.md).

</div>

## License

GPL-3.0-only. See [LICENSE](https://github.com/NeLy-EPFL/deeperfly/blob/main/LICENSE).
