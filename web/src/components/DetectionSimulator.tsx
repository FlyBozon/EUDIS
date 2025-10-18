import { useEffect } from 'react';
import { useAppStore } from '@/store/index';
import { Detection } from '@/types/index';

export const DetectionSimulator = () => {
  const espNodes = useAppStore((state) => state.espNodes);
  const addDetection = useAppStore((state) => state.addDetection);

  useEffect(() => {
    if (espNodes.length === 0) return;

    const interval = setInterval(() => {
      // Random chance of detection (15% chance every second)
      if (Math.random() > 0.85) {
        const randomNode = espNodes[Math.floor(Math.random() * espNodes.length)];
        
        const detection: Detection = {
          id: `detection-${Date.now()}-${Math.random()}`,
          lineId: randomNode.id,
          timestamp: Date.now(),
          intensity: 60 + Math.random() * 40,
          frequency: 20 + Math.random() * 80,
        };

        addDetection(detection);

        // Auto-remove detections after 5 seconds
        const timeoutId = setTimeout(() => {
          useAppStore.setState((state) => ({
            detections: state.detections.filter((d) => d.id !== detection.id),
          }));
        }, 5000);

        // Cleanup timeout on unmount
        return () => clearTimeout(timeoutId);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [espNodes, addDetection]);

  return null;
};
