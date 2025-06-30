
# Gemini Project Context

This file provides context for the Gemini AI assistant to understand the project.

## Project Overview

This project is a web-based application that simulates and visualizes spatial audio in a virtual 3D environment. It utilizes WebGPU for rendering and the Web Audio API for audio processing. The core of the project is the generation of a binaural impulse response (IR) that captures the acoustic properties of a virtual room. This IR is then used to spatialize a sound source, creating a realistic 3D audio experience for the listener.

## Core Technologies

*   **Frontend:**
    *   **TypeScript:** The primary programming language.
    *   **Vite:** The build tool and development server.
    *   **HTML5/CSS3:** For the user interface.
*   **Graphics and Rendering:**
    *   **WebGPU:** For high-performance 3D rendering of the room, sound source, and listener.
    *   **gl-matrix:** For vector and matrix operations in WebGL and WebGPU.
*   **Audio Processing:**
    *   **Web Audio API:** For all audio-related tasks, including generating, processing, and playing back audio.
*   **UI and Debugging:**
    *   **dat.gui:** For creating a simple UI to control simulation parameters.

## Key Concepts and Algorithms

The project is built around several key concepts and algorithms from the fields of acoustics, digital signal processing, and computer graphics.

### 1. Ray Tracing for Acoustics

*   **Image Source Method:** Used to calculate early reflections by creating virtual sound sources mirrored across room surfaces.
*   **Stochastic Ray Tracing:** Used to model late reverberation by tracing rays in random directions and simulating their interactions with the environment.

### 2. Impulse Response (IR) Generation

*   **Early Reflections:** The first part of the IR, corresponding to direct sound and distinct early echoes.
*   **Late Reverberation:** The dense, diffuse part of the IR that follows the early reflections.
*   **Binaural Impulse Response:** A stereo IR that includes directional cues for realistic 3D audio playback over headphones.

### 3. Late Reverberation Modeling

*   **Feedback Delay Network (FDN):** An algorithm that creates a dense and natural-sounding reverberation tail using a network of delay lines and a feedback matrix.
*   **Velvet Noise:** An efficient algorithm for generating diffuse reverberation that is perceptually smooth.
*   **Diffuse Field Model:** A statistical model of the sound field after early reflections have passed.

### 4. Binaural Spatialization

*   **Head-Related Transfer Functions (HRTF):** Filters that simulate how the head, torso, and outer ears affect incoming sound, providing the primary cues for sound localization.
*   **Spherical Head Model:** A simplified model used to calculate HRTF-based gains for left and right ears.

### 5. Diffraction Modeling

*   The project includes research on implementing diffraction models, which is a crucial aspect of realistic sound propagation, especially around obstacles.
*   **Uniform Theory of Diffraction (UTD):** A candidate model for handling edge diffraction.

## Project Structure

*   `src/`: Contains the main source code.
*   `src/main.ts`: The main entry point of the application.
*   `src/raytracer/`: Contains the ray tracing and acoustic simulation logic.
*   `src/sound/`: Contains the audio processing and impulse response generation logic.
*   `src/room/`: Contains the room geometry and material properties.
*   `public/`: Contains static assets, including HRTF data.
*   `index.html`: The main HTML file.
*   `package.json`: Defines project dependencies and scripts.
