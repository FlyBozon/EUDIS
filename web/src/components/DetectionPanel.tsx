import { useAppStore } from '@/store/index';

export const DetectionPanel = () => {
  const detections = useAppStore((state) => state.detections);
  const espNodes = useAppStore((state) => state.espNodes);

  const recentDetections = detections.slice(-10).reverse();

  return (
    <div className="bg-card border border-border rounded-lg p-4 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <h2 className="font-semibold">Wykrycia</h2>
        <span className="ml-auto text-sm text-muted-foreground">{detections.length}</span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-2">
        {recentDetections.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground">
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
                  Pewność: {Math.round(detection.intensity)}% | Poziom sygnału: {Math.round(detection.frequency)} dB
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
