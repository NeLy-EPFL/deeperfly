# Library API reference

The public API, generated from the source docstrings. Most of it is re-exported at the
top level (`from deeperfly import Config, run_recording, ...` — `deeperfly.__all__` is
the list); the rest is imported from the module named in its heading below. For
task-oriented examples see the [library guide](../guides/library.md); for the array and
coordinate conventions these functions share, see
[Conventions & glossary](../explanation/conventions.md).

## Configuration

::: deeperfly.config.Config

## Cameras

::: deeperfly.cameras.Camera

::: deeperfly.cameras.CameraGroup

## Camera calibration files

The portable form of a solved rig — see
[`calibration.toml`](output-format.md#calibrationtoml) for the file itself.

::: deeperfly.calibration.Calibration

## Skeleton

::: deeperfly.skeleton.Skeleton

::: deeperfly.skeleton.resolve_points


## Chirality (left/right swap detection)

::: deeperfly.chirality

## Results

::: deeperfly.results.PoseResult

::: deeperfly.results.StageStore

::: deeperfly.results.repack

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

::: deeperfly.pose2d.pathways.DetectionPlan

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

## Pose corrections

The `postprocess` stage's op chain — one pure function per correction, registered in
`OPS`; see [`[postprocess]`](configuration.md#postprocess) for the config side.

::: deeperfly.postprocess

## Inverse kinematics

Fitting NeuroMechFly's joint angles to a 3D pose. Needs the optional `deeperfly[ik]`
extra (QuickIK); without it `solve_inverse_kinematics` raises `MissingQuickIK`, which
is what the pipeline stage catches to skip rather than fail the run.

::: deeperfly.inverse_kinematics.solve_inverse_kinematics

::: deeperfly.inverse_kinematics.IKResult

::: deeperfly.inverse_kinematics.KinematicTemplate

## Acquisition (active learning)

::: deeperfly.acquisition

## Pictorial structures

::: deeperfly.pictorial

## Frame I/O

::: deeperfly.io
