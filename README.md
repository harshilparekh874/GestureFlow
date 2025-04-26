# GestureFlow

A real-time, hands-free YouTube Gesture Controller built in Python. GestureFlow leverages computer vision and MediaPipe Hands to recognize hand poses and translate them into YouTube player controls, with an extensible architecture for calibration, Chrome extension packaging, and integration into collaboration tools.

## Features

- **Play / Pause**: Open palm to play or resume, fist to pause.
- **Volume Control**: Index + middle fingers up to increase volume, ring + pinky fingers up to decrease volume.
- **Time Skip**: Pinky-only to skip forward 10 seconds, all fingers except pinky to skip backward 10 seconds.
- **Two-Hand Navigation**: Pointing "gun" shape with left or right hand to go to next or previous video via browser history shortcuts.
- **On-Screen Feedback**: Real-time overlay of the detected gesture name.
- **Cooldown Logic**: Built-in timing control to prevent accidental repeated triggers.
- **Gesture Calibration**: Interactive calibration mode to teach custom gestures.
- **Chrome Extension (Planned)**: Packaging as a browser extension for seamless in-browser control.

## Tech Stack

- **Python**: Core language for scripting and control logic.
- **OpenCV**: Captures webcam frames, flips image, converts color spaces, and displays video overlays.
- **MediaPipe Hands**: Provides efficient, accurate hand landmark detection and tracking in real time.
- **PyAutoGUI**: Simulates keyboard events to control YouTube and other applications via native hotkeys.
- **NumPy**: Numeric operations and vector calculations for gesture template matching in calibration mode.
- **PyInstaller** (optional): Bundles the Python app into standalone executables for distribution.

## Repository Structure

```
gestureflow/
├── controller.py       # Main application logic (built-in and calibrated modes)
├── gestures.json       # Saved templates from calibration mode
├── requirements.txt    # Python dependencies
├── setup.py            # Packaging configuration for PyPI
├── README.md           # Project documentation (this file)
└── LICENSE             # License file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/gestureflow.git
   cd gestureflow
   ```
2. Create and activate a Python environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### Built-In Gestures

Run the default, rule-based detector:
```bash
python controller.py
```

### Calibration Mode

Record your own custom gestures:
```bash
python controller.py --calibrate
```
Follow the on-screen prompts and press `c` to capture samples.

### Calibrated Mode

Run using your saved templates:
```bash
python controller.py --use_calibrated
```

## Packaging as a Chrome Extension

A JavaScript / manifest-based version is in development, leveraging MediaPipe’s JS API to embed gesture detection into the browser.

## Integration with Collaboration Tools

GestureFlow can map gestures to application hotkeys:

- **Zoom**: Alt+A to mute/unmute, Alt+Y to raise hand.
- **Microsoft Teams**: Ctrl+Shift+M to mute/unmute, Ctrl+Shift+E to share screen.
- **Slack**: Use PyAutoGUI or Slack SDK to send messages or reactions.
- **Presentations**: Simulate arrow keys or page controls to advance slides.

Customize the `ACTIONS` mapping in `controller.py` or your `config.yml` to suit your workflows.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to:

- Add new gesture mappings
- Improve detection accuracy
- Extend integration examples
- Package executables for non-Python users

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

