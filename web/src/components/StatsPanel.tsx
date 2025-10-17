import { useAppStore } from '@/store/index';
import { MapPin, Zap, AlertCircle } from 'lucide-react';

export const StatsPanel = () => {
  const deploymentLines = useAppStore((state) => state.deploymentLines);
  const espNodes = useAppStore((state) => state.espNodes);
  const detections = useAppStore((state) => state.detections);

  return (
    <div className="bg-card border border-border rounded-lg p-4 space-y-4">
      <div className="flex items-center gap-3 p-3 bg-background rounded-lg">
        <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
          <Zap className="w-5 h-5 text-blue-500" />
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Linie wdrożeń</p>
          <p className="text-lg font-semibold">{deploymentLines.length}</p>
        </div>
      </div>

      <div className="flex items-center gap-3 p-3 bg-background rounded-lg">
        <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
          <MapPin className="w-5 h-5 text-blue-500" />
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Węzły ESP</p>
          <p className="text-lg font-semibold">{espNodes.length}</p>
        </div>
      </div>

      <div className="flex items-center gap-3 p-3 bg-background rounded-lg">
        <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center">
          <AlertCircle className="w-5 h-5 text-amber-500" />
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Aktywne wykrycia</p>
          <p className="text-lg font-semibold">{detections.length}</p>
        </div>
      </div>
    </div>
  );
};
