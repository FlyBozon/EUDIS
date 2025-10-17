import { create } from 'zustand';
import { Detection, EspNode, DeploymentLine, Theme } from '@/types/index';

interface AppStore {
  // Theme management
  theme: Theme;
  setTheme: (theme: Theme) => void;

  // Deployment data
  deploymentLines: DeploymentLine[];
  addDeploymentLine: (line: DeploymentLine) => void;
  clearDeploymentLines: () => void;

  // ESP nodes
  espNodes: EspNode[];
  addEspNode: (node: EspNode) => void;
  addEspNodes: (nodes: EspNode[]) => void;
  clearEspNodes: () => void;

  // Detections
  detections: Detection[];
  addDetection: (detection: Detection) => void;
  clearDetections: () => void;

  // UI states
  selectedLineId: string | null;
  setSelectedLineId: (id: string | null) => void;

  // Mission planning
  missionStartPoint: [number, number] | null;
  missionEndPoint: [number, number] | null;
  setMissionStartPoint: (point: [number, number] | null) => void;
  setMissionEndPoint: (point: [number, number] | null) => void;
  clearMissionPoints: () => void;
}

export const useAppStore = create<AppStore>((set) => ({
  // Theme
  theme: 'auto',
  setTheme: (theme) => set({ theme }),

  // Deployment lines
  deploymentLines: [],
  addDeploymentLine: (line) =>
    set((state) => ({
      deploymentLines: [...state.deploymentLines, line],
    })),
  clearDeploymentLines: () => set({ deploymentLines: [] }),

  // ESP nodes
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

  // Detections
  detections: [],
  addDetection: (detection) =>
    set((state) => ({
      detections: [...state.detections, detection],
    })),
  clearDetections: () => set({ detections: [] }),

  // UI
  selectedLineId: null,
  setSelectedLineId: (id) => set({ selectedLineId: id }),

  // Mission planning
  missionStartPoint: null,
  missionEndPoint: null,
  setMissionStartPoint: (point) => set({ missionStartPoint: point }),
  setMissionEndPoint: (point) => set({ missionEndPoint: point }),
  clearMissionPoints: () => set({ missionStartPoint: null, missionEndPoint: null }),
}));
