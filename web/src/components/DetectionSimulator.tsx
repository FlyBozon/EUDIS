import { useEffect, useRef } from 'react';
import { useAppStore } from '@/store/index';
import { Detection } from '@/types/index';

// Funkcja do obliczania odległości między dwoma punktami GPS (w metrach)
const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
  const R = 6371000; // Promień Ziemi w metrach
  const φ1 = (lat1 * Math.PI) / 180;
  const φ2 = (lat2 * Math.PI) / 180;
  const Δφ = ((lat2 - lat1) * Math.PI) / 180;
  const Δλ = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
    Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
};

interface DroneState {
  position: { lat: number; lng: number };
  target: { lat: number; lng: number };
}

export const DetectionSimulator = () => {
  const espNodes = useAppStore((state) => state.espNodes);
  const addDetection = useAppStore((state) => state.addDetection);
  const setDronePositions = useAppStore((state) => state.setDronePositions);
  const dronesRef = useRef<DroneState[]>([]);
  const lastDetectionTimeRef = useRef<{ [droneNodeKey: string]: number }>({});

  useEffect(() => {
    if (espNodes.length === 0) return;

    // Znajdź środek i zakres obszaru z czujników
    const avgLat = espNodes.reduce((sum, node) => sum + node.latitude, 0) / espNodes.length;
    const avgLng = espNodes.reduce((sum, node) => sum + node.longitude, 0) / espNodes.length;
    const latitudes = espNodes.map((n) => n.latitude);
    const longitudes = espNodes.map((n) => n.longitude);
    const latRange = Math.max(...latitudes) - Math.min(...latitudes);
    const lngRange = Math.max(...longitudes) - Math.min(...longitudes);

    // Inicjalizacja wielu dronów dla lepszej wizualizacji systemu
    const NUM_DRONES = 8; // 8 dronów jednocześnie
    const initializeDrone = (): DroneState => {
      return {
        position: {
          lat: avgLat + (Math.random() - 0.5) * latRange * 1.3,
          lng: avgLng + (Math.random() - 0.5) * lngRange * 1.3,
        },
        target: {
          lat: avgLat + (Math.random() - 0.5) * latRange * 1.3,
          lng: avgLng + (Math.random() - 0.5) * lngRange * 1.3,
        },
      };
    };

    // Dostosuj liczbę dronów do liczby czujników
    const numDrones = Math.max(1, Math.min(NUM_DRONES, Math.ceil(espNodes.length / 3)));
    dronesRef.current = Array.from({ length: numDrones }, initializeDrone);

    const DETECTION_RANGE_METERS = 200; // Zasięg wykrywania drona
    const DRONE_SPEED_M_PER_S = 20; // Szybsze drony (20 m/s ≈ 72 km/h)
    const UPDATE_INTERVAL_MS = 150; // Częstsze aktualizacje (150ms)

    const interval = setInterval(() => {
      const currentTime = Date.now();

      // Aktualizuj każdy dron
      dronesRef.current.forEach((drone, droneIdx) => {
        const { lat: droneLat, lng: droneLng } = drone.position;
        const { lat: targetLat, lng: targetLng } = drone.target;

        // Sprawdź czy dron dotarł do celu
        const distanceToTarget = calculateDistance(droneLat, droneLng, targetLat, targetLng);

        if (distanceToTarget < 50) {
          // Jeśli dron dotarł do celu, wybierz nowy cel
          drone.target = {
            lat: avgLat + (Math.random() - 0.5) * latRange * 1.3,
            lng: avgLng + (Math.random() - 0.5) * lngRange * 1.3,
          };
        } else {
          // Przesuń drona w kierunku celu
          const distanceToMove = (DRONE_SPEED_M_PER_S * UPDATE_INTERVAL_MS) / 1000; // metry

          // Oblicz kierunek ruchu
          const latDiff = targetLat - droneLat;
          const lngDiff = targetLng - droneLng;
          const angle = Math.atan2(latDiff, lngDiff);

          // Przesuń drona (przybliżone przeliczenie metrów na stopnie)
          const latMove = (distanceToMove * Math.sin(angle)) / 111000;
          const lngMove = (distanceToMove * Math.cos(angle)) / (111000 * Math.cos((droneLat * Math.PI) / 180));

          drone.position = {
            lat: droneLat + latMove,
            lng: droneLng + lngMove,
          };
        }

        // Sprawdź które czujniki są w zasięgu tego drona i wygeneruj detekcje
        espNodes.forEach((node) => {
          const distance = calculateDistance(
            drone.position.lat,
            drone.position.lng,
            node.latitude,
            node.longitude
          );

          if (distance <= DETECTION_RANGE_METERS) {
            // Unikalny klucz dla kombinacji dron-węzeł
            const droneNodeKey = `${droneIdx}-${node.id}`;
            const lastDetectionTime = lastDetectionTimeRef.current[droneNodeKey] || 0;
            
            // Krótsza przerwa między detekcjami
            if (currentTime - lastDetectionTime >= 500) {
              // Usuń poprzednie wykrycie z tej pary dron-czujnik (jeśli istnieje)
              useAppStore.setState((state) => ({
                detections: state.detections.filter((d) => {
                  // Sprawdź czy to wykrycie pochodzi z tej samej pary dron-czujnik
                  const detectionKey = d.id.split('-')[2] + '-' + d.lineId; // droneIdx-nodeId
                  const currentKey = droneIdx + '-' + node.id;
                  return detectionKey !== currentKey;
                }),
              }));

              // Intensywność zależy od odległości (im bliżej, tym silniejszy sygnał)
              const intensityBase = 100 - (distance / DETECTION_RANGE_METERS) * 40; // 60-100%
              const intensity = Math.max(60, Math.min(100, intensityBase + (Math.random() - 0.5) * 10));

              // Częstotliwość z lekką losowością
              const frequency = 50 + Math.random() * 50; // 50-100 dB

              const detection: Detection = {
                id: `detection-${currentTime}-${droneIdx}-${node.id}-${Math.random()}`,
                lineId: node.id,
                timestamp: currentTime,
                intensity,
                frequency,
              };

              addDetection(detection);
              lastDetectionTimeRef.current[droneNodeKey] = currentTime;

              // Usuń detekcje po dłuższym czasie (3s) - ale i tak będzie tylko jedno per para
              setTimeout(() => {
                useAppStore.setState((state) => ({
                  detections: state.detections.filter((d) => d.id !== detection.id),
                }));
              }, 3000);
            }
          }
        });
      });

      // Wyświetl wszystkie drony na mapie
      const allPositions: Array<[number, number]> = dronesRef.current.map((drone) => [
        drone.position.lat,
        drone.position.lng,
      ]);
      setDronePositions(allPositions);
    }, UPDATE_INTERVAL_MS);

    return () => {
      clearInterval(interval);
      setDronePositions([]); // Usuń drony z mapy po zakończeniu
    };
  }, [espNodes, addDetection, setDronePositions]);

  return null;
};
