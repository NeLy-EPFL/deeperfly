# Library API reference

The complete public API, generated from the source docstrings. Everything here is
importable from the top level (`from deeperfly import ...`). For task-oriented
examples see the [library guide](../guides/library.md); for the array and
coordinate conventions these functions share, see
[Conventions & glossary](../explanation/conventions.md).

## Configuration

::: deeperfly.config.Config

## Cameras

::: deeperfly.cameras.Camera

::: deeperfly.cameras.CameraGroup

## Skeleton

::: deeperfly.skeleton.Skeleton

::: deeperfly.skeleton.infer_symmetries_by_name

## Chirality (left/right swap detection)

::: deeperfly.chirality

## Results

::: deeperfly.results.PoseResult

## Recordings

::: deeperfly.recordings.Recording

::: deeperfly.recordings.resolve_recordings

## Bundle adjustment

::: deeperfly.bundle_adjustment.bundle_adjust

::: deeperfly.bundle_adjustment.bundle_adjust_from_config

## Pipeline

::: deeperfly.pipeline.run_from_points2d

::: deeperfly.pipeline.run_recording

## 2D detection

::: deeperfly.pose2d.stream.load_models

::: deeperfly.pose2d.stream.detect_2d

::: deeperfly.pose2d.autocrop

::: deeperfly.pose2d.autocrop.ensure_resolved

## Geometry primitives

::: deeperfly.geometry

## Triangulation helpers

::: deeperfly.triangulation

## Ensemble Kalman smoother

::: deeperfly.eks

::: deeperfly.eks.smooth

::: deeperfly.eks.EksResult

## Acquisition (active learning)

::: deeperfly.acquisition

## Pictorial structures

::: deeperfly.pictorial

## Frame I/O

::: deeperfly.io
