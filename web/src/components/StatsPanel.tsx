import { useAppStore } from '@/store/index';
import { useTranslation } from '@/i18n/useTranslation';

export const StatsPanel = () => {
  const deploymentLines = useAppStore((state) => state.deploymentLines);
  const espNodes = useAppStore((state) => state.espNodes);
  const detections = useAppStore((state) => state.detections);
  const t = useTranslation();

  return (
    <div className="bg-card border border-border rounded-lg p-4 space-y-2">
      <div className="p-3 bg-background rounded-lg">
        <p className="text-xs text-muted-foreground">{t.stats.routes}</p>
        <p className="text-lg font-semibold">{deploymentLines.length}</p>
      </div>

      <div className="p-3 bg-background rounded-lg">
        <p className="text-xs text-muted-foreground">{t.stats.sensors}</p>
        <p className="text-lg font-semibold">{espNodes.length}</p>
      </div>

      <div className="p-3 bg-background rounded-lg">
        <p className="text-xs text-muted-foreground">{t.stats.detections}</p>
        <p className="text-lg font-semibold">{detections.length}</p>
      </div>
    </div>
  );
};