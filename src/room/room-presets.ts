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
        absorptionLow: 0.05,
        absorptionMid: 0.04,
        absorptionHigh: 0.03,
        scatteringLow: 0.1,
        scatteringMid: 0.15,
        scatteringHigh: 0.2,
        roughness: 0.3,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      ceiling: {
        absorptionLow: 0.04,
        absorptionMid: 0.03,
        absorptionHigh: 0.02,
        scatteringLow: 0.1,
        scatteringMid: 0.15,
        scatteringHigh: 0.2,
        roughness: 0.35,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      floor: {
        absorptionLow: 0.1,
        absorptionMid: 0.08,
        absorptionHigh: 0.06,
        scatteringLow: 0.2,
        scatteringMid: 0.25,
        scatteringHigh: 0.3,
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
        absorptionLow: 0.3,
        absorptionMid: 0.6,
        absorptionHigh: 0.8,
        scatteringLow: 0.4,
        scatteringMid: 0.6,
        scatteringHigh: 0.8,
        roughness: 0.6,
        phaseShift: 0.0,
        phaseRandomization: 0.2
      },
      ceiling: {
        absorptionLow: 0.25,
        absorptionMid: 0.5,
        absorptionHigh: 0.7,
        scatteringLow: 0.3,
        scatteringMid: 0.5,
        scatteringHigh: 0.7,
        roughness: 0.5,
        phaseShift: 0.0,
        phaseRandomization: 0.2
      },
      floor: {
        absorptionLow: 0.15,
        absorptionMid: 0.2,
        absorptionHigh: 0.3,
        scatteringLow: 0.2,
        scatteringMid: 0.3,
        scatteringHigh: 0.4,
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
        absorptionLow: 0.15,
        absorptionMid: 0.12,
        absorptionHigh: 0.1,
        scatteringLow: 0.3,
        scatteringMid: 0.4,
        scatteringHigh: 0.5,
        roughness: 0.45,
        phaseShift: 0.0,
        phaseRandomization: 0.15
      },
      ceiling: {
        absorptionLow: 0.12,
        absorptionMid: 0.1,
        absorptionHigh: 0.08,
        scatteringLow: 0.25,
        scatteringMid: 0.35,
        scatteringHigh: 0.45,
        roughness: 0.4,
        phaseShift: 0.0,
        phaseRandomization: 0.15
      },
      floor: {
        absorptionLow: 0.2,
        absorptionMid: 0.18,
        absorptionHigh: 0.15,
        scatteringLow: 0.35,
        scatteringMid: 0.45,
        scatteringHigh: 0.55,
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
        absorptionLow: 0.25,
        absorptionMid: 0.4,
        absorptionHigh: 0.6,
        scatteringLow: 0.35,
        scatteringMid: 0.5,
        scatteringHigh: 0.7,
        roughness: 0.5,
        phaseShift: 0.0,
        phaseRandomization: 0.15
      },
      ceiling: {
        absorptionLow: 0.2,
        absorptionMid: 0.35,
        absorptionHigh: 0.5,
        scatteringLow: 0.3,
        scatteringMid: 0.45,
        scatteringHigh: 0.6,
        roughness: 0.45,
        phaseShift: 0.0,
        phaseRandomization: 0.15
      },
      floor: {
        absorptionLow: 0.1,
        absorptionMid: 0.15,
        absorptionHigh: 0.2,
        scatteringLow: 0.2,
        scatteringMid: 0.25,
        scatteringHigh: 0.3,
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
        absorptionLow: 0.15,
        absorptionMid: 0.3,
        absorptionHigh: 0.4,
        scatteringLow: 0.3,
        scatteringMid: 0.4,
        scatteringHigh: 0.5,
        roughness: 0.4,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      ceiling: {
        absorptionLow: 0.2,
        absorptionMid: 0.4,
        absorptionHigh: 0.6,
        scatteringLow: 0.35,
        scatteringMid: 0.5,
        scatteringHigh: 0.65,
        roughness: 0.5,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      floor: {
        absorptionLow: 0.1,
        absorptionMid: 0.15,
        absorptionHigh: 0.2,
        scatteringLow: 0.2,
        scatteringMid: 0.25,
        scatteringHigh: 0.3,
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
        absorptionLow: 0.02,
        absorptionMid: 0.02,
        absorptionHigh: 0.03,
        scatteringLow: 0.1,
        scatteringMid: 0.15,
        scatteringHigh: 0.2,
        roughness: 0.25,
        phaseShift: 0.0,
        phaseRandomization: 0.05
      },
      ceiling: {
        absorptionLow: 0.05,
        absorptionMid: 0.05,
        absorptionHigh: 0.06,
        scatteringLow: 0.15,
        scatteringMid: 0.2,
        scatteringHigh: 0.25,
        roughness: 0.3,
        phaseShift: 0.0,
        phaseRandomization: 0.05
      },
      floor: {
        absorptionLow: 0.03,
        absorptionMid: 0.03,
        absorptionHigh: 0.04,
        scatteringLow: 0.12,
        scatteringMid: 0.18,
        scatteringHigh: 0.22,
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
        absorptionLow: 0.1,
        absorptionMid: 0.08,
        absorptionHigh: 0.06,
        scatteringLow: 0.25,
        scatteringMid: 0.35,
        scatteringHigh: 0.45,
        roughness: 0.4,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      ceiling: {
        absorptionLow: 0.08,
        absorptionMid: 0.06,
        absorptionHigh: 0.05,
        scatteringLow: 0.2,
        scatteringMid: 0.3,
        scatteringHigh: 0.4,
        roughness: 0.35,
        phaseShift: 0.0,
        phaseRandomization: 0.1
      },
      floor: {
        absorptionLow: 0.15,
        absorptionMid: 0.12,
        absorptionHigh: 0.1,
        scatteringLow: 0.3,
        scatteringMid: 0.4,
        scatteringHigh: 0.5,
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
        absorptionLow: 0.95,
        absorptionMid: 0.98,
        absorptionHigh: 0.99,
        scatteringLow: 0.9,
        scatteringMid: 0.95,
        scatteringHigh: 0.98,
        roughness: 0.9,
        phaseShift: 0.0,
        phaseRandomization: 0.3
      },
      ceiling: {
        absorptionLow: 0.95,
        absorptionMid: 0.98,
        absorptionHigh: 0.99,
        scatteringLow: 0.9,
        scatteringMid: 0.95,
        scatteringHigh: 0.98,
        roughness: 0.9,
        phaseShift: 0.0,
        phaseRandomization: 0.3
      },
      floor: {
        absorptionLow: 0.95,
        absorptionMid: 0.98,
        absorptionHigh: 0.99,
        scatteringLow: 0.9,
        scatteringMid: 0.95,
        scatteringHigh: 0.98,
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