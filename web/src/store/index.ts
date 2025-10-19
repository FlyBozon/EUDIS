import { create } from 'zustand';
import { Detection, EspNode, DeploymentLine, Theme } from '@/types/index';
import type { Language } from '@/i18n/translations';

interface AppStore {
  
  theme: Theme;
  setTheme: (theme: Theme) => void;

  
  language: Language;
  setLanguage: (language: Language) => void;

  
  deploymentLines: DeploymentLine[];
  addDeploymentLine: (line: DeploymentLine) => void;
  clearDeploymentLines: () => void;

  
  espNodes: EspNode[];
  addEspNode: (node: EspNode) => void;
  addEspNodes: (nodes: EspNode[]) => void;
  clearEspNodes: () => void;

  
  detections: Detection[];
  addDetection: (detection: Detection) => void;
  clearDetections: () => void;

  
  dronePositions: Array<[number, number]>;
  setDronePositions: (positions: Array<[number, number]>) => void;

  
  selectedLineId: string | null;
  setSelectedLineId: (id: string | null) => void;

  
  missionStartPoint: [number, number] | null;
  missionEndPoint: [number, number] | null;
  setMissionStartPoint: (point: [number, number] | null) => void;
  setMissionEndPoint: (point: [number, number] | null) => void;
  clearMissionPoints: () => void;
}

export const useAppStore = create<AppStore>((set) => ({
  
  theme: 'auto',
  setTheme: (theme) => set({ theme }),

  
  language: 'pl',
  setLanguage: (language) => set({ language }),

  
  deploymentLines: [],
  addDeploymentLine: (line) =>
    set((state) => ({
      deploymentLines: [...state.deploymentLines, line],
    })),
  clearDeploymentLines: () => set({ deploymentLines: [] }),

  
  espNodes: [],
  addEspNode: (node) =>
    set((state) => ({
      espNodes: [...state.espNodes, node],
    })),
  addEspNodes: (nodes) =>
    set((state) => ({
      espNodes: [...state.espNodes, ...nodes],
    })),
  clearEspNodes: () => set({ espNodes: [] }),

  
  detections: [],
  addDetection: (detection) =>
    set((state) => ({
      detections: [...state.detections, detection],
    })),
  clearDetections: () => set({ detections: [] }),

  
  dronePositions: [],
  setDronePositions: (positions) => set({ dronePositions: positions }),

  
  selectedLineId: null,
  setSelectedLineId: (id) => set({ selectedLineId: id }),

  
  missionStartPoint: null,
  missionEndPoint: null,
  setMissionStartPoint: (point) => set({ missionStartPoint: point }),
  setMissionEndPoint: (point) => set({ missionEndPoint: point }),
  clearMissionPoints: () => set({ missionStartPoint: null, missionEndPoint: null }),
}));