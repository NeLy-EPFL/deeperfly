<p align="center">
  <img src="assets/logo.svg" alt="deeperfly logo" width="180">
</p>

# deeperfly

Markerless 3D pose estimation of tethered *Drosophila* from a multi-camera rig.
`deeperfly` estimates camera parameters and 2D/3D keypoint locations from
behavioral recordings through one linear pipeline: **2D pose → bundle adjustment
→ triangulation → visualization**.

It is both a command-line tool and a Python library, and a modern rewrite of
[DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D),
[DeepFly2D](https://github.com/NeLy-EPFL/DeepFly2D) and
[PyBundleAdjustment](https://github.com/semihgunel/PyBundleAdjustment).

## Install

```bash
uv tool install git+https://github.com/NeLy-EPFL/deeperfly --python 3.13 --torch-backend=auto
```

## Run

```bash
deeperfly doctor                                             # check the install
deeperfly init config.toml                                   # write a config (edit if needed)
deeperfly run examples/data/ -c config.toml                  # 2D -> 3D -> video
deeperfly inspect examples/data/deeperfly_outputs/results.h5 # summarize the result
```

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting started](getting-started.md)** — run the
  bundled example end to end, from install to a rendered 3D video.
- :material-console: **[CLI usage](guides/cli.md)** — every command and flag.
- :material-file-cog: **[Writing configs](guides/configuration.md)** — the
  `config.toml`, section by section.
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
