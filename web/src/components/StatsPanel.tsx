import { useAppStore } from '@/store/index';

export const StatsPanel = () => {
  const deploymentLines = useAppStore((state) => state.deploymentLines);
  const espNodes = useAppStore((state) => state.espNodes);
  const detections = useAppStore((state) => state.detections);

  return (
    <div className="bg-card border border-border rounded-lg p-4 space-y-2">
      <div className="p-3 bg-background rounded-lg">
        <p className="text-xs text-muted-foreground">Linie wdrożeń</p>
        <p className="text-lg font-semibold">{deploymentLines.length}</p>
      </div>

      <div className="p-3 bg-background rounded-lg">
        <p className="text-xs text-muted-foreground">Węzły ESP</p>
        <p className="text-lg font-semibold">{espNodes.length}</p>
      </div>

      <div className="p-3 bg-background rounded-lg">
        <p className="text-xs text-muted-foreground">Aktywne wykrycia</p>
        <p className="text-lg font-semibold">{detections.length}</p>
      </div>
    </div>
  );
};
