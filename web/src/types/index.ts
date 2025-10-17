export interface Detection {
  id: string;
  lineId: string;
  timestamp: number;
  intensity: number; // 0-100
  frequency: number; // dB
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
  spacing: 300 | 500 | 1000; // meters
  createdAt: number;
  costMin?: number; // PLN
  costMax?: number; // PLN
  weightMin?: number; // kg
  weightMax?: number; // kg
  distanceKm?: number; // km
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
