import { MapComponent } from '@/components/MapComponent';
import { DeploymentDialog } from '@/components/DeploymentDialog';
import { DetectionPanel } from '@/components/DetectionPanel';
import { StatsPanel } from '@/components/StatsPanel';
import { ThemeSwitcher } from '@/components/ThemeSwitcher';
import { DetectionSimulator } from '@/components/DetectionSimulator';
import { useAppStore } from '@/store/index';

function App() {
  const theme = useAppStore((state) => state.theme);

  return (
    <div className={theme === 'dark' || (theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : ''}>
      <div className="min-h-screen bg-background text-foreground">
        {/* Header */}
        <header className="border-b border-border sticky top-0 z-40 bg-card/95 backdrop-blur">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">EUDIS</h1>
              <p className="text-xs text-muted-foreground">
                System monitorowania dronów z wykorzystaniem sensorów audio
              </p>
            </div>
            <ThemeSwitcher />
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto p-4">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 h-[calc(100vh-120px)]">
            {/* Map */}
            <div className="lg:col-span-3">
              <div className="bg-card border border-border rounded-lg overflow-hidden h-full">
                <MapComponent />
              </div>
            </div>

            {/* Sidebar */}
            <div className="flex flex-col gap-4">
              {/* Deploy Button */}
              <DeploymentDialog />

              {/* Stats */}
              <StatsPanel />

              {/* Detections */}
              <div className="flex-1 min-h-0">
                <DetectionPanel />
              </div>
            </div>
          </div>
        </main>

        {/* Detection Simulator */}
        <DetectionSimulator />
      </div>
    </div>
  );
}

export default App;
