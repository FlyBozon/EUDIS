import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { useAppStore } from '@/store/index';
import { DeploymentLine } from '@/types/index';
import { useTranslation } from '@/i18n/useTranslation';

export const DeploymentDialog = () => {
  const [spacing, setSpacing] = useState<100 | 200 | 300>(200);
  const [isOpen, setIsOpen] = useState(false);

  const missionStartPoint = useAppStore((state) => state.missionStartPoint);
  const missionEndPoint = useAppStore((state) => state.missionEndPoint);
  const addDeploymentLine = useAppStore((state) => state.addDeploymentLine);
  const addEspNodes = useAppStore((state) => state.addEspNodes);
  const clearMissionPoints = useAppStore((state) => state.clearMissionPoints);
  const t = useTranslation();

  
  const costAndWeightData: Record<100 | 200 | 300, { costMin: number; costMax: number; weightMin: number; weightMax: number }> = {
    100: { costMin: 800, costMax: 950, weightMin: 4.5, weightMax: 5.5 },
    200: { costMin: 600, costMax: 750, weightMin: 3.5, weightMax: 4.5 },
    300: { costMin: 530, costMax: 600, weightMin: 3.0, weightMax: 4.0 },
  };

  
  const calculateDistance = (): number => {
    if (!missionStartPoint || !missionEndPoint) return 0;
    const distanceInMeters = Math.sqrt(
      Math.pow(missionEndPoint[0] - missionStartPoint[0], 2) +
      Math.pow(missionEndPoint[1] - missionStartPoint[1], 2)
    ) * 111000; 
    return distanceInMeters / 1000;
  };

  const distanceKm = calculateDistance();
  
  
  const calculateSensorCount = (): number => {
    if (!missionStartPoint || !missionEndPoint) return 0;
    const distanceInMeters = distanceKm * 1000;
    return Math.floor(distanceInMeters / spacing) + 1; 
  };
  
  const sensorCount = calculateSensorCount();

  const handleDeploy = () => {
    if (!missionStartPoint || !missionEndPoint) {
      alert(t.deploymentDialog.clickMap);
      return;
    }

    const [startLat, startLng] = missionStartPoint;
    const [endLat, endLng] = missionEndPoint;

    
    const distanceInMeters = Math.sqrt(
      Math.pow(endLat - startLat, 2) +
      Math.pow(endLng - startLng, 2)
    ) * 111000; 
    const distanceKm = distanceInMeters / 1000;

    
    const costWeightData = costAndWeightData[spacing];
    
    
    const totalCostMin = Math.round(costWeightData.costMin * distanceKm);
    const totalCostMax = Math.round(costWeightData.costMax * distanceKm);
    const totalWeightMin = Math.round((costWeightData.weightMin * distanceKm) * 10) / 10; 
    const totalWeightMax = Math.round((costWeightData.weightMax * distanceKm) * 10) / 10;

    const lineId = `line-${Date.now()}`;
    const line: DeploymentLine = {
      id: lineId,
      startLat,
      startLng,
      endLat,
      endLng,
      spacing,
      createdAt: Date.now(),
      distanceKm: Math.round(distanceKm * 100) / 100,
      costMin: totalCostMin,
      costMax: totalCostMax,
      weightMin: totalWeightMin,
      weightMax: totalWeightMax,
    };

    
    const nodes = [];
    const steps = Math.floor(distanceInMeters / spacing);
    const latStep = (line.endLat - line.startLat) / steps;
    const lngStep = (line.endLng - line.startLng) / steps;

    for (let i = 0; i <= steps; i++) {
      nodes.push({
        id: `esp-${lineId}-${i}`,
        latitude: line.startLat + latStep * i,
        longitude: line.startLng + lngStep * i,
        lineId,
        deploymentTime: Date.now() + i * 100,
      });
    }

    addDeploymentLine(line);
    addEspNodes(nodes);
    clearMissionPoints();
    setIsOpen(false);
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button
          size="lg"
          className={`gap-2 font-semibold w-full ${!missionStartPoint || !missionEndPoint ? 'bg-gray-300 text-gray-600 cursor-not-allowed' : 'bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white'}`}
          disabled={!missionStartPoint || !missionEndPoint}
          title={!missionStartPoint || !missionEndPoint ? t.deploymentDialog.clickMap : undefined}
        >
          {missionStartPoint && missionEndPoint ? t.deploymentDialog.deploy : t.deploymentDialog.clickMap}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-xl">
            {t.deploymentDialog.title}
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-5">
          { }
          <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded-lg p-4 space-y-3">
            <h4 className="font-semibold text-sm text-blue-900 dark:text-blue-200">
              {t.deploymentDialog.subtitle}
            </h4>
            
            <div className="space-y-2 ml-6 text-sm">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 inline-block rounded-full" style={{ background: missionStartPoint ? '#16a34a' : '#cbd5e1' }} />
                <span className="text-muted-foreground">{t.deploymentDialog.startPoint}</span>
                {missionStartPoint && (
                  <span className="text-green-600 dark:text-green-400 text-xs ml-auto">
                    {missionStartPoint[0].toFixed(3)}°, {missionStartPoint[1].toFixed(3)}°
                  </span>
                )}
              </div>
              
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 inline-block rounded-full" style={{ background: missionEndPoint ? '#dc2626' : '#cbd5e1' }} />
                <span className="text-muted-foreground">{t.deploymentDialog.endPoint}</span>
                {missionEndPoint && (
                  <span className="text-red-600 dark:text-red-400 text-xs ml-auto">
                    {missionEndPoint[0].toFixed(3)}°, {missionEndPoint[1].toFixed(3)}°
                  </span>
                )}
              </div>
            </div>

            {!missionStartPoint || !missionEndPoint ? (
              <p className="text-xs text-blue-700 dark:text-blue-300 ml-6">
                {t.deploymentDialog.clickMap}
              </p>
            ) : (
              <div className="ml-6 pt-2 border-t border-blue-200 dark:border-blue-700">
                <div className="text-xs font-semibold text-blue-900 dark:text-blue-200">
                  {t.deploymentDialog.distance}: <span className="text-blue-600 dark:text-blue-300">
                    {Math.round(Math.sqrt(
                      Math.pow(missionEndPoint[0] - missionStartPoint[0], 2) +
                      Math.pow(missionEndPoint[1] - missionStartPoint[1], 2)
                    ) * 111000)} m
                  </span>
                </div>
              </div>
            )}
          </div>

          { }
          <div className="space-y-3">
            <label className="text-sm font-semibold block">{t.deploymentDialog.sensorSpacing}</label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { value: 100, label: '100m', desc: t.deploymentDialog.spacingOptions.dense },
                { value: 200, label: '200m', desc: t.deploymentDialog.spacingOptions.standard },
                { value: 300, label: '300m', desc: t.deploymentDialog.spacingOptions.wide },
              ].map((option) => (
                <button
                  key={option.value}
                  onClick={() => setSpacing(option.value as 100 | 200 | 300)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    spacing === option.value
                      ? 'bg-primary text-primary-foreground border-primary shadow-lg'
                      : 'border-border hover:border-accent hover:bg-accent/50'
                  }`}
                >
                  <div className="font-semibold">{option.label}</div>
                  <div className="text-xs opacity-75">{option.desc}</div>
                </button>
              ))}
            </div>
            {missionStartPoint && missionEndPoint ? (
              <p className="text-xs font-semibold text-foreground">
                {t.deploymentDialog.sensorsCount}: <span className="text-primary">{sensorCount}</span>
              </p>
            ) : (
              <p className="text-xs text-muted-foreground">
                {t.deploymentDialog.clickMap}
              </p>
            )}
          </div>

          { }
          {missionStartPoint && missionEndPoint && (
            <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-lg p-4 space-y-3">
              <h4 className="font-semibold text-sm text-amber-900 dark:text-amber-200">
                {t.deploymentDialog.costRange}
              </h4>
              
              <div className="space-y-2 ml-6 text-sm">
                <div className="flex items-center justify-between text-muted-foreground">
                  <span>{t.deploymentDialog.distance}:</span>
                  <span className="font-semibold text-amber-600 dark:text-amber-400">
                    {Math.round(distanceKm * 100) / 100} km
                  </span>
                </div>
                
                <div className="flex items-center justify-between text-muted-foreground">
                  <span>{t.deploymentDialog.sensorSpacing}:</span>
                  <span className="font-semibold text-amber-600 dark:text-amber-400">{spacing} m</span>
                </div>
                
                <div className="flex items-center justify-between text-muted-foreground">
                  <span>{t.deploymentDialog.sensorsCount}:</span>
                  <span className="font-semibold text-amber-600 dark:text-amber-400">{sensorCount}</span>
                </div>

                <div className="border-t border-amber-200 dark:border-amber-700 pt-2 mt-2">
                  <div className="flex items-center justify-between text-muted-foreground mb-2">
                    <span>{t.deploymentDialog.costRange} (PLN):</span>
                    <span className="font-bold text-amber-700 dark:text-amber-300">
                      {Math.round(costAndWeightData[spacing].costMin * distanceKm)} – {Math.round(costAndWeightData[spacing].costMax * distanceKm)}
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-muted-foreground">
                    <span>{t.deploymentDialog.weightRange} (kg):</span>
                    <span className="font-bold text-amber-700 dark:text-amber-300">
                      {Math.round(costAndWeightData[spacing].weightMin * distanceKm * 10) / 10} – {Math.round(costAndWeightData[spacing].weightMax * distanceKm * 10) / 10}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          { }
          <Button 
            onClick={handleDeploy} 
            className="w-full h-11 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white font-semibold gap-2"
            disabled={!missionStartPoint || !missionEndPoint}
          >
              {missionStartPoint && missionEndPoint ? t.deploymentDialog.deploy : t.deploymentDialog.clickMap}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};