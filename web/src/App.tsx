import { MapComponent } from '@/components/MapComponent';
import { DeploymentDialog } from '@/components/DeploymentDialog';
import { DetectionPanel } from '@/components/DetectionPanel';
import { StatsPanel } from '@/components/StatsPanel';
import { ThemeSwitcher } from '@/components/ThemeSwitcher';
import { LanguageSwitch } from '@/components/LanguageSwitch';
import { DetectionSimulator } from '@/components/DetectionSimulator';
import { useAppStore } from '@/store/index';
import { useTranslation } from '@/i18n/useTranslation';

function App() {
  const theme = useAppStore((state) => state.theme);
  const t = useTranslation();

  return (
    <div className={theme === 'dark' || (theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : ''}>
      <div className="min-h-screen bg-background text-foreground">
        { }
        <header className="border-b border-border sticky top-0 z-40 bg-card/95 backdrop-blur">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">{t.app.title}</h1>
              <p className="text-xs text-muted-foreground">
                {t.app.subtitle}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <LanguageSwitch />
              <ThemeSwitcher />
            </div>
          </div>
        </header>

        { }
        <main className="max-w-7xl mx-auto p-4">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 h-[calc(100vh-120px)]">
            { }
            <div className="lg:col-span-3">
              <div className="bg-card border border-border rounded-lg overflow-hidden h-full">
                <MapComponent />
              </div>
            </div>

            { }
            <div className="flex flex-col gap-4">
              { }
              <DeploymentDialog />

              { }
              <StatsPanel />

              { }
              <div className="flex-1 min-h-0">
                <DetectionPanel />
              </div>
            </div>
          </div>
        </main>

        { }
        <DetectionSimulator />
      </div>
    </div>
  );
}

export default App;