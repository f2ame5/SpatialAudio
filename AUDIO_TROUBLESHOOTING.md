# Audio Troubleshooting Guide

## Common Issues and Solutions

### Audio Files Not Loading

If you see errors like "Unable to decode audio data", try these solutions:

1. **Use Generated Test Sounds**:
   - Select "Test Tone (440Hz)" or "White Noise" from the Audio File dropdown
   - These are generated programmatically and don't require external files

2. **Check Audio File Format**:
   - WAV files should be:
     - PCM format (not compressed)
     - 16 or 24-bit
     - 44.1kHz or 48kHz sample rate
   - If your WAV files don't work, try converting them with a tool like Audacity

3. **File Location**:
   - Audio files should be in the `public/soundfile/` directory
   - They are accessed via URLs like `/soundfile/filename.wav`

4. **Browser Console**:
   - Check the browser's developer console for specific error messages
   - Some browsers have stricter audio format requirements

### Converting Audio Files

If your WAV files don't work, convert them using ffmpeg:

```bash
# Convert to standard PCM WAV
ffmpeg -i input.wav -acodec pcm_s16le -ar 44100 output.wav
```

Or use Audacity:
1. Open the file in Audacity
2. File → Export → Export Audio
3. Choose WAV format
4. Set encoding to "Signed 16-bit PCM"

### Test Without Audio Files

The system includes built-in test sounds:
- **Test Tone (440Hz)**: A simple sine wave
- **White Noise**: Random noise for testing

These work without any external files and are good for testing the spatial audio system.

## Browser Requirements

- Chrome/Edge 113+ with WebGPU enabled
- Web Audio API support (all modern browsers)
- User interaction required to start audio (click a button first)
