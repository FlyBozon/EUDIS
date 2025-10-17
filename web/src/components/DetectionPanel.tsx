import { useAppStore } from '@/store/index';
import { Activity, AlertCircle } from 'lucide-react';

export const DetectionPanel = () => {
  const detections = useAppStore((state) => state.detections);
  const espNodes = useAppStore((state) => state.espNodes);

  const recentDetections = detections.slice(-10).reverse();

  return (
    <div className="bg-card border border-border rounded-lg p-4 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <AlertCircle className="w-5 h-5 text-accent" />
        <h2 className="font-semibold">Wykrycia</h2>
        <span className="ml-auto text-sm text-muted-foreground">
          {detections.length}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-2">
        {recentDetections.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <Activity className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-sm">Brak wykryć</p>
          </div>
        ) : (
          recentDetections.map((detection) => {
            const node = espNodes.find((n) => n.id === detection.lineId);
            return (
              <div
                key={detection.id}
                className="bg-background border border-border rounded p-2 text-sm animate-detection"
              >
                <div className="flex justify-between items-start">
                  <div className="font-medium text-accent">
                    {node?.id || 'Unknown'}
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {new Date(detection.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Intensywność: {Math.round(detection.intensity)}% | Częstotliwość:{' '}
                  {Math.round(detection.frequency)} dB
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
