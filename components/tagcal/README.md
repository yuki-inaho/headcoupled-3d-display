# tagcal

Camera intrinsic calibration from an AprilTag grid, shown on a monitor at true
physical size or printed. Record a video, get intrinsics.

Requires `uv`, Python 3.11-3.13, and `opencv-contrib-python` (AprilTag
dictionaries). PySide6 for the GUI.

## Install

```bash
uv sync --all-extras
```

## Use

```bash
uv run tagcal panel
```

`V` shows the board on a monitor, `C` opens the camera view, `R` starts and stops
recording. Stopping runs selection and calibration.

Same thing from the CLI:

```bash
uv run tagcal screen list
uv run tagcal screen show artifacts/screen --monitor DP-2 --tag-size-mm 40
uv run tagcal devices
uv run tagcal record capture.mp4 --camera 0 --width 1920 --height 1080
uv run tagcal process capture.mp4 artifacts/screen/pattern.json artifacts/session
```

Printed board instead of a monitor — measure the 100 mm reference bar and pass it
back:

```bash
uv run tagcal pattern generate artifacts/pattern --columns 6 --rows 4 --tag-size-mm 35
uv run tagcal pattern confirm-scale artifacts/pattern/pattern.json --measured-reference-mm 98.7
```

Check the result against a tape measure:

```bash
uv run tagcal verify artifacts/screen/pattern.json \
  --calibration artifacts/session/calibration.json --camera 0
```

## Output

```
artifacts/session/
  calibration.json          intrinsics, per-view errors, standard deviations
  calibration_opencv.yaml   OpenCV FileStorage
  camera_info.yaml          ROS CameraInfo
  report.html
  keyframes/
```

## Notes

- Tag size is the outer edge of the black border, not the white quiet zone —
  confusing the two scales every distance by 1.25x. On-screen boards take their
  true size from the monitor's pixel pitch and write it into `pattern.json`, which
  is what the solver reads. See [docs/screen_display.md](docs/screen_display.md).
- MJPG is requested from the camera by default. V4L2 otherwise serves uncompressed
  YUYV, which USB bandwidth caps at 5 fps at 1080p. Any difference between the
  requested and granted mode is reported.
- Read the standard deviations next to `fx`/`fy`/`cx`/`cy`, not only the RMS.
  Fewer views always fit better while estimating worse.

## Development

```bash
just test
just lint
just typecheck
```

MIT
