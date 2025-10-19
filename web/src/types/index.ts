export interface Detection {
  id: string;
  lineId: string;
  timestamp: number;
  intensity: number; 
  frequency: number; 
}

export interface EspNode {
  id: string;
  latitude: number;
  longitude: number;
  lineId: string;
  deploymentTime?: number;
}

export interface DeploymentLine {
  id: string;
  startLat: number;
  startLng: number;
  endLat: number;
  endLng: number;
  spacing: 100 | 200 | 300; 
  createdAt: number;
  costMin?: number; 
  costMax?: number; 
  weightMin?: number; 
  weightMax?: number; 
  distanceKm?: number; 
}

export interface DroneRoute {
  id: string;
  waypoints: Array<{
    lat: number;
    lng: number;
  }>;
  status: 'planning' | 'flying' | 'completed' | 'cancelled';
  createdAt: number;
}

export type Theme = 'light' | 'dark' | 'auto';