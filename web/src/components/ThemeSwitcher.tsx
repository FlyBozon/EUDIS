import { useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';
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
    <div className="flex items-center gap-2">
      <Button
        variant={theme === 'light' ? 'default' : 'outline'}
        size="icon"
        onClick={() => setTheme('light')}
        title="Light theme"
      >
        <Sun className="w-4 h-4" />
      </Button>
      <Button
        variant={theme === 'dark' ? 'default' : 'outline'}
        size="icon"
        onClick={() => setTheme('dark')}
        title="Dark theme"
      >
        <Moon className="w-4 h-4" />
      </Button>
      <Button
        variant={theme === 'auto' ? 'default' : 'outline'}
        size="icon"
        onClick={() => setTheme('auto')}
        title="Auto theme"
      >
        <div className="w-4 h-4 text-xs flex items-center justify-center">A</div>
      </Button>
    </div>
  );
};
