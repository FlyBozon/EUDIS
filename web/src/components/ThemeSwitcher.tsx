import { useEffect, useState } from 'react';
import { useAppStore } from '@/store/index';
import { Button } from '@/components/ui/button';

export const ThemeSwitcher = () => {
  const theme = useAppStore((state) => state.theme);
  const setTheme = useAppStore((state) => state.setTheme);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    
    // Apply theme to document
    const applyTheme = () => {
      const root = document.documentElement;
      let effectiveTheme = theme;
      
      if (theme === 'auto') {
        effectiveTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
          ? 'dark'
          : 'light';
      }

      if (effectiveTheme === 'dark') {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    };

    applyTheme();

    // Listen for system theme changes
    if (theme === 'auto') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      mediaQuery.addEventListener('change', applyTheme);
      return () => mediaQuery.removeEventListener('change', applyTheme);
    }
  }, [theme]);

  if (!mounted) return null;

  return (
    <div className="flex items-center gap-2" aria-label="Wybór motywu">
      <span className="text-sm text-muted-foreground">Motyw:</span>
      <div role="group" aria-label="Motyw" className="inline-flex rounded-md border border-border overflow-hidden">
        <Button
          variant={theme === 'light' ? 'default' : 'ghost'}
          onClick={() => setTheme('light')}
          aria-pressed={theme === 'light'}
          className={`px-3 py-1.5 rounded-none ${theme === 'light' ? '' : 'hover:bg-accent'}`}
          title="Motyw jasny"
        >
          Jasny
        </Button>
        <Button
          variant={theme === 'dark' ? 'default' : 'ghost'}
          onClick={() => setTheme('dark')}
          aria-pressed={theme === 'dark'}
          className={`px-3 py-1.5 rounded-none border-l border-border ${theme === 'dark' ? '' : 'hover:bg-accent'}`}
          title="Motyw ciemny"
        >
          Ciemny
        </Button>
        <Button
          variant={theme === 'auto' ? 'default' : 'ghost'}
          onClick={() => setTheme('auto')}
          aria-pressed={theme === 'auto'}
          className={`px-3 py-1.5 rounded-none border-l border-border ${theme === 'auto' ? '' : 'hover:bg-accent'}`}
          title="Motyw systemowy"
        >
          Systemowy
        </Button>
      </div>
    </div>
  );
};
