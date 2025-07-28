/**
 * RoomPresets - Configuration for different room types with realistic acoustic properties
 * 
 * This file defines presets for various room types including cathedrals, studios,
 * concert halls, and other acoustic environments. Each preset includes both
 * geometric dimensions and material properties that affect sound propagation.
 */

import { RoomConfig } from './room';
import { WallMaterial } from './room-materials';

// Define interface for room presets
export interface RoomPreset {
  name: string;
  description: string;
  // Geometric properties
  dimensions: {
    width: number;
    height: number;
    depth: number;
  };
  // Material properties for different surfaces
  materials: {
    walls: Partial<WallMaterial>;
    ceiling: Partial<WallMaterial>;
    floor: Partial<WallMaterial>;
  };
  // Additional acoustic characteristics
  characteristics: {
    reverbTime: number; // RT60 in seconds
    clarity: number; // Clarity index (C80)
    warmth: number; // Warmth index (C50)
    intimacy: boolean; // Whether the room feels intimate
    diffusion: number; // Diffusion coefficient
  };
}

// Helper to create 8-band properties from 3-band
const createBands = (low: number, mid: number, high: number) => ({
    absorption63: low,
    absorption125: low,
    absorption250: (low + mid) / 2,
    absorption500: mid,
    absorption1k: mid,
    absorption2k: (mid + high) / 2,
    absorption4k: high,
    absorption8k: high,
    
    scattering63: low,
    scattering125: low,
    scattering250: (low + mid) / 2,
    scattering500: mid,
    scattering1k: mid,
    scattering2k: (mid + high) / 2,
    scattering4k: high,
    scattering8k: high
});

// Export all room presets
export const ROOM_PRESETS: { [key: string]: RoomPreset } = {
  /**
   * Cathedral - Large reverberant space with stone surfaces
   * Characterized by long reverb times and strong early reflections
   */
  CATHEDRAL: {
    name: "Cathedral",
    description: "Large stone cathedral with high ceilings and long reverberation time",
    dimensions: {
      width: 30,
      height: 40,
      depth: 50
    },
    materials: {
      walls: {
        ...createBands(0.05, 0.04, 0.03),
        roughness: 0.3,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      ceiling: {
        ...createBands(0.04, 0.03, 0.02),
        roughness: 0.35,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      floor: {
        ...createBands(0.1, 0.08, 0.06),
        roughness: 0.25,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      }
    },
    characteristics: {
      reverbTime: 8.0,
      clarity: -4.0,
      warmth: -2.0,
      intimacy: false,
      diffusion: 0.7
    }
  },

  /**
   * Recording Studio - Small controlled environment
   * Designed for minimal reflections and balanced frequency response
   */
  RECORDING_STUDIO: {
    name: "Recording Studio",
    description: "Professional recording studio with acoustic treatment for neutral sound",
    dimensions: {
      width: 6,
      height: 3,
      depth: 5
    },
    materials: {
      walls: {
        absorption63: 0.3, absorption125: 0.4, absorption250: 0.5, absorption500: 0.6,
        absorption1k: 0.7, absorption2k: 0.8, absorption4k: 0.8, absorption8k: 0.8,
        scattering63: 0.4, scattering125: 0.5, scattering250: 0.6, scattering500: 0.6,
        scattering1k: 0.7, scattering2k: 0.8, scattering4k: 0.8, scattering8k: 0.8,
        roughness: 0.6,
        phaseShift: 0.0,
        phaseRandomization: 0.2
      },
      ceiling: {
        absorption63: 0.25, absorption125: 0.35, absorption250: 0.45, absorption500: 0.5,
        absorption1k: 0.6, absorption2k: 0.7, absorption4k: 0.7, absorption8k: 0.7,
        scattering63: 0.3, scattering125: 0.4, scattering250: 0.5, scattering500: 0.5,
        scattering1k: 0.6, scattering2k: 0.7, scattering4k: 0.7, scattering8k: 0.7,
        roughness: 0.5,
        phaseShift: 0.0,
        phaseRandomization: 0.2
      },
      floor: {
        absorption63: 0.15, absorption125: 0.18, absorption250: 0.2, absorption500: 0.25,
        absorption1k: 0.3, absorption2k: 0.3, absorption4k: 0.3, absorption8k: 0.3,
        scattering63: 0.2, scattering125: 0.25, scattering250: 0.3, scattering500: 0.3,
        scattering1k: 0.35, scattering2k: 0.4, scattering4k: 0.4, scattering8k: 0.4,
        roughness: 0.3,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      }
    },
    characteristics: {
      reverbTime: 0.3,
      clarity: 2.0,
      warmth: 1.0,
      intimacy: true,
      diffusion: 0.9
    }
  },

  /**
   * Concert Hall - Medium-large performance space
   * Balanced acoustics for musical performances
   */
  CONCERT_HALL: {
    name: "Concert Hall",
    description: "Symphony concert hall with balanced acoustics for orchestral music",
    dimensions: {
      width: 25,
      height: 15,
      depth: 40
    },
    materials: {
      walls: {
        ...createBands(0.15, 0.12, 0.1),
        roughness: 0.45,
        phaseShift: 0.0,
        phaseRandomization: 0.15
      },
      ceiling: {
        ...createBands(0.12, 0.1, 0.08),
        roughness: 0.4,
        phaseShift: 0.0,
        phaseRandomization: 0.15
      },
      floor: {
        ...createBands(0.2, 0.18, 0.15),
        roughness: 0.4,
        phaseShift: 0.0,
        phaseRandomization: 0.15
      }
    },
    characteristics: {
      reverbTime: 2.0,
      clarity: 1.5,
      warmth: 2.5,
      intimacy: true,
      diffusion: 0.8
    }
  },

  /**
   * Home Theater - Medium-sized dedicated viewing room
   * Designed for clear dialogue and immersive surround sound
   */
  HOME_THEATER: {
    name: "Home Theater",
    description: "Dedicated home theater room with acoustic treatment for movie watching",
    dimensions: {
      width: 8,
      height: 3,
      depth: 10
    },
    materials: {
      walls: {
        ...createBands(0.25, 0.4, 0.6),
        roughness: 0.5,
        phaseShift: 0.0,
        phaseRandomization: 0.15
      },
      ceiling: {
        ...createBands(0.2, 0.35, 0.5),
        roughness: 0.45,
        phaseShift: 0.0,
        phaseRandomization: 0.15
      },
      floor: {
        ...createBands(0.1, 0.15, 0.2),
        roughness: 0.3,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      }
    },
    characteristics: {
      reverbTime: 0.4,
      clarity: 3.0,
      warmth: 1.5,
      intimacy: true,
      diffusion: 0.85
    }
  },

  /**
   * Classroom - Medium-sized educational space
   * Optimized for speech intelligibility
   */
  CLASSROOM: {
    name: "Classroom",
    description: "Standard classroom with focus on speech intelligibility",
    dimensions: {
      width: 10,
      height: 3,
      depth: 8
    },
    materials: {
      walls: {
        ...createBands(0.15, 0.3, 0.4),
        roughness: 0.4,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      ceiling: {
        ...createBands(0.2, 0.4, 0.6),
        roughness: 0.5,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      floor: {
        ...createBands(0.1, 0.15, 0.2),
        roughness: 0.3,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      }
    },
    characteristics: {
      reverbTime: 0.6,
      clarity: 2.5,
      warmth: 1.0,
      intimacy: true,
      diffusion: 0.75
    }
  },

  /**
   * Bathroom - Small reflective tiled space
   * Characterized by short, bright reverb and strong reflections
   */
  BATHROOM: {
    name: "Bathroom",
    description: "Small bathroom with tiled walls and bright, reflective acoustics",
    dimensions: {
      width: 2.5,
      height: 2.5,
      depth: 2
    },
    materials: {
      walls: {
        ...createBands(0.02, 0.02, 0.03),
        roughness: 0.25,
        phaseShift: 0.0,
        phaseRandomization: 0.05
      },
      ceiling: {
        ...createBands(0.05, 0.05, 0.06),
        roughness: 0.3,
        phaseShift: 0.0,
        phaseRandomization: 0.05
      },
      floor: {
        ...createBands(0.03, 0.03, 0.04),
        roughness: 0.28,
        phaseShift: 0.0,
        phaseRandomization: 0.05
      }
    },
    characteristics: {
      reverbTime: 1.2,
      clarity: -1.0,
      warmth: -3.0,
      intimacy: true,
      diffusion: 0.6
    }
  },

  /**
   * Wood Panel Room - Medium-sized room with wood paneling
   * Warm, resonant acoustics with natural wood reflections
   */
  WOOD_PANEL_ROOM: {
    name: "Wood Panel Room",
    description: "Room with wood paneling on walls and ceiling, creating warm acoustics",
    dimensions: {
      width: 7,
      height: 3,
      depth: 6
    },
    materials: {
      walls: {
        ...createBands(0.1, 0.08, 0.06),
        roughness: 0.4,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      ceiling: {
        ...createBands(0.08, 0.06, 0.05),
        roughness: 0.35,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      floor: {
        ...createBands(0.15, 0.12, 0.1),
        roughness: 0.45,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      }
    },
    characteristics: {
      reverbTime: 0.8,
      clarity: 1.0,
      warmth: 3.5,
      intimacy: true,
      diffusion: 0.7
    }
  },

  /**
   * Anechoic Chamber - Specialized acoustic space
   * Designed to absorb all sound reflections
   */
  ANECHOIC_CHAMBER: {
    name: "Anechoic Chamber",
    description: "Specialized chamber with wedge absorbers to eliminate all reflections",
    dimensions: {
      width: 4,
      height: 3,
      depth: 4
    },
    materials: {
      walls: {
        ...createBands(0.95, 0.98, 0.99),
        roughness: 0.9,
        phaseShift: 0.0,
        phaseRandomization: 0.3
      },
      ceiling: {
        ...createBands(0.95, 0.98, 0.99),
        roughness: 0.9,
        phaseShift: 0.0,
        phaseRandomization: 0.3
      },
      floor: {
        ...createBands(0.95, 0.98, 0.99),
        roughness: 0.9,
        phaseShift: 0.0,
        phaseRandomization: 0.3
      }
    },
    characteristics: {
      reverbTime: 0.05,
      clarity: 10.0,
      warmth: -5.0,
      intimacy: false,
      diffusion: 0.95
    }
  }
};

// Default room preset
export const DEFAULT_ROOM_PRESET = ROOM_PRESETS.RECORDING_STUDIO;