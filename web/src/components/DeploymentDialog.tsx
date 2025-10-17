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
import { Plane } from 'lucide-react';

export const DeploymentDialog = () => {
  const [spacing, setSpacing] = useState<300 | 500 | 1000>(500);
  const [isOpen, setIsOpen] = useState(false);

  const missionStartPoint = useAppStore((state) => state.missionStartPoint);
  const missionEndPoint = useAppStore((state) => state.missionEndPoint);
  const addDeploymentLine = useAppStore((state) => state.addDeploymentLine);
  const addEspNodes = useAppStore((state) => state.addEspNodes);
  const clearMissionPoints = useAppStore((state) => state.clearMissionPoints);

  // Dane kosztów i wagi na podstawie rozstawu
  const costAndWeightData: Record<300 | 500 | 1000, { costMin: number; costMax: number; weightMin: number; weightMax: number }> = {
    300: { costMin: 3470, costMax: 7867, weightMin: 3.0, weightMax: 4.5 },
    500: { costMin: 3080, costMax: 7320, weightMin: 2.6, weightMax: 4.0 },
    1000: { costMin: 2790, costMax: 6910, weightMin: 2.2, weightMax: 3.7 },
  };

  // Funkcja pomocnicza do obliczenia dystansu w km
  const calculateDistance = (): number => {
    if (!missionStartPoint || !missionEndPoint) return 0;
    const distanceInMeters = Math.sqrt(
      Math.pow(missionEndPoint[0] - missionStartPoint[0], 2) +
      Math.pow(missionEndPoint[1] - missionStartPoint[1], 2)
    ) * 111000; // approximately meters per degree
    return distanceInMeters / 1000;
  };

  const distanceKm = calculateDistance();

  const handleDeploy = () => {
    if (!missionStartPoint || !missionEndPoint) {
      alert('Proszę wybrać zarówno punkt startu jak i punkt końca na mapie');
      return;
    }

    const [startLat, startLng] = missionStartPoint;
    const [endLat, endLng] = missionEndPoint;

    // Obliczenie dystansu w km
    const distanceInMeters = Math.sqrt(
      Math.pow(endLat - startLat, 2) +
      Math.pow(endLng - startLng, 2)
    ) * 111000; // approximately meters per degree
    const distanceKm = distanceInMeters / 1000;

    // Pobierz dane kosztów i wagi dla wybranego rozstawu
    const costWeightData = costAndWeightData[spacing];
    
    // Oblicz całkowity koszt i wagę na podstawie dystansu
    const totalCostMin = Math.round(costWeightData.costMin * distanceKm);
    const totalCostMax = Math.round(costWeightData.costMax * distanceKm);
    const totalWeightMin = Math.round((costWeightData.weightMin * distanceKm) * 10) / 10; // 1 decimal place
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

    // Calculate ESP node positions along the line
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
          className="gap-2 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white font-semibold"
        >
          <Plane className="w-5 h-5" />
          Wyślij drona
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl">
            <span className="text-2xl">✈️</span> Konfiguruj wdrożenie
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-5">
          {/* Checklist - Route Planning */}
          <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded-lg p-4 space-y-3">
            <h4 className="font-semibold text-sm text-blue-900 dark:text-blue-200 flex items-center gap-2">
              <span>📍</span> Zaplanowana trasa
            </h4>
            
            <div className="space-y-2 ml-6 text-sm">
              <div className="flex items-center gap-2">
                <span className={missionStartPoint ? '✅' : '⭕'}>
                </span>
                <span className="text-muted-foreground">Punkt startowy</span>
                {missionStartPoint && (
                  <span className="text-green-600 dark:text-green-400 text-xs ml-auto">
                    {missionStartPoint[0].toFixed(3)}°, {missionStartPoint[1].toFixed(3)}°
                  </span>
                )}
              </div>
              
              <div className="flex items-center gap-2">
                <span className={missionEndPoint ? '✅' : '⭕'}>
                </span>
                <span className="text-muted-foreground">Punkt końcowy</span>
                {missionEndPoint && (
                  <span className="text-red-600 dark:text-red-400 text-xs ml-auto">
                    {missionEndPoint[0].toFixed(3)}°, {missionEndPoint[1].toFixed(3)}°
                  </span>
                )}
              </div>
            </div>

            {!missionStartPoint || !missionEndPoint ? (
              <p className="text-xs text-blue-700 dark:text-blue-300 ml-6">
                💡 Zatwierdź trasę na mapie przed konfiguracją wdrożenia
              </p>
            ) : (
              <div className="ml-6 pt-2 border-t border-blue-200 dark:border-blue-700">
                <div className="text-xs font-semibold text-blue-900 dark:text-blue-200">
                  Dystans trasy: <span className="text-blue-600 dark:text-blue-300">
                    {Math.round(Math.sqrt(
                      Math.pow(missionEndPoint[0] - missionStartPoint[0], 2) +
                      Math.pow(missionEndPoint[1] - missionStartPoint[1], 2)
                    ) * 111000)} m
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Spacing Configuration */}
          <div className="space-y-3">
            <label className="text-sm font-semibold block">🎯 Rozstaw sensorów ESP</label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { value: 300, label: '300m', desc: 'Denser' },
                { value: 500, label: '500m', desc: 'Standard' },
                { value: 1000, label: '1000m', desc: 'Wide' },
              ].map((option) => (
                <button
                  key={option.value}
                  onClick={() => setSpacing(option.value as 300 | 500 | 1000)}
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
            <p className="text-xs text-muted-foreground">
              {spacing === 300 && '🔍 Sensorów: ~20-30'}
              {spacing === 500 && '📊 Sensorów: ~10-20'}
              {spacing === 1000 && '📡 Sensorów: ~5-10'}
            </p>
          </div>

          {/* Cost and Weight Summary */}
          {missionStartPoint && missionEndPoint && (
            <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-lg p-4 space-y-3">
              <h4 className="font-semibold text-sm text-amber-900 dark:text-amber-200 flex items-center gap-2">
                <span>💰</span> Szacunkowe koszty i waga
              </h4>
              
              <div className="space-y-2 ml-6 text-sm">
                <div className="flex items-center justify-between text-muted-foreground">
                  <span>Dystans:</span>
                  <span className="font-semibold text-amber-600 dark:text-amber-400">
                    {Math.round(distanceKm * 100) / 100} km
                  </span>
                </div>
                
                <div className="flex items-center justify-between text-muted-foreground">
                  <span>Rozstaw ESP:</span>
                  <span className="font-semibold text-amber-600 dark:text-amber-400">{spacing} m</span>
                </div>

                <div className="border-t border-amber-200 dark:border-amber-700 pt-2 mt-2">
                  <div className="flex items-center justify-between text-muted-foreground mb-2">
                    <span>💵 Koszt całkowity (PLN):</span>
                    <span className="font-bold text-amber-700 dark:text-amber-300">
                      {Math.round(costAndWeightData[spacing].costMin * distanceKm)} – {Math.round(costAndWeightData[spacing].costMax * distanceKm)}
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-muted-foreground">
                    <span>⚖️ Waga całkowita (kg):</span>
                    <span className="font-bold text-amber-700 dark:text-amber-300">
                      {Math.round(costAndWeightData[spacing].weightMin * distanceKm * 10) / 10} – {Math.round(costAndWeightData[spacing].weightMax * distanceKm * 10) / 10}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Deploy Button */}
          <Button 
            onClick={handleDeploy} 
            className="w-full h-11 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white font-semibold gap-2"
            disabled={!missionStartPoint || !missionEndPoint}
          >
            <Plane className="w-5 h-5" />
            {missionStartPoint && missionEndPoint ? 'Wdróż drona' : 'Wybierz trasę na mapie'}
          </Button>

          {/* Info */}
          <p className="text-xs text-muted-foreground text-center leading-relaxed">
            Po kliknięciu "Wdróż drona", system automatycznie rozmieści sensory na trasie i rozpocznie monitorowanie
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
};
