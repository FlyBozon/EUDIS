export type Language = 'pl' | 'en';

export interface Translations {
  
  stats: {
    title: string;
    routes: string;
    sensors: string;
    detections: string;
  };
  
  
  detectionPanel: {
    title: string;
    noDetections: string;
    sensor: string;
    route: string;
    intensity: string;
    frequency: string;
  };
  
  
  deploymentDialog: {
    title: string;
    subtitle: string;
    startPoint: string;
    endPoint: string;
    clickMap: string;
    distance: string;
    sensorSpacing: string;
    spacingOptions: {
      dense: string;
      standard: string;
      wide: string;
    };
    costRange: string;
    weightRange: string;
    sensorsCount: string;
    deploy: string;
    cancel: string;
  };
  
  
  map: {
    fullscreen: string;
    exitFullscreen: string;
    searchPlaceholder: string;
    searching: string;
    noResults: string;
    sensor: string;
    route: string;
    selectStart: string;
    selectEnd: string;
    drone: string;
    detection: string;
    planRoute: string;
    close: string;
    reset: string;
    done: string;
    plannedRoute: string;
    start: string;
    end: string;
    distance: string;
    editRoute: string;
    removeRoute: string;
  };
  
  
  language: {
    polish: string;
    english: string;
  };
  
  
  theme: {
    label: string;
    light: string;
    dark: string;
    auto: string;
    lightTitle: string;
    darkTitle: string;
    autoTitle: string;
  };
  
  
  app: {
    title: string;
    subtitle: string;
  };
}

export const translations: Record<Language, Translations> = {
  pl: {
    stats: {
      title: 'Statystyki',
      routes: 'Trasy monitorowania',
      sensors: 'Czujniki',
      detections: 'Aktywne wykrycia',
    },
    detectionPanel: {
      title: 'Wykrycia dronów',
      noDetections: 'Brak aktywnych wykryć',
      sensor: 'Czujnik',
      route: 'Trasa',
      intensity: 'Intensywność',
      frequency: 'Częstotliwość',
    },
    deploymentDialog: {
      title: 'Nowe wdrożenie czujników',
      subtitle: 'Wybierz punkty początkowy i końcowy na mapie',
      startPoint: 'Punkt początkowy',
      endPoint: 'Punkt końcowy',
      clickMap: 'Kliknij na mapie aby wybrać',
      distance: 'Dystans',
      sensorSpacing: 'Rozstaw czujników',
      spacingOptions: {
        dense: 'Gęsty',
        standard: 'Standard',
        wide: 'Szeroki',
      },
      costRange: 'Szacowany koszt',
      weightRange: 'Waga sprzętu',
      sensorsCount: 'Liczba czujników',
      deploy: 'Wdróż',
      cancel: 'Anuluj',
    },
    map: {
      fullscreen: 'Pełny ekran',
      exitFullscreen: 'Wyjdź z pełnego ekranu',
      searchPlaceholder: 'Szukaj miejsca...',
      searching: 'Wyszukiwanie...',
      noResults: 'Brak wyników',
      sensor: 'Czujnik',
      route: 'Trasa',
      selectStart: 'Wybierz punkt początkowy',
      selectEnd: 'Wybierz punkt końcowy',
      drone: 'Dron',
      detection: 'Wykrycie',
      planRoute: 'Zaplanuj trasę',
      close: 'Zamknij',
      reset: 'Resetuj',
      done: 'Gotowe',
      plannedRoute: 'Zaplanowana trasa',
      start: 'Start',
      end: 'Koniec',
      distance: 'Dystans',
      editRoute: 'Edytuj trasę',
      removeRoute: 'Usuń trasę',
    },
    language: {
      polish: 'Polski',
      english: 'Angielski',
    },
    theme: {
      label: 'Motyw:',
      light: 'Jasny',
      dark: 'Ciemny',
      auto: 'Systemowy',
      lightTitle: 'Motyw jasny',
      darkTitle: 'Motyw ciemny',
      autoTitle: 'Motyw systemowy',
    },
    app: {
      title: 'EUDIS',
      subtitle: 'System monitorowania dronów z wykorzystaniem sensorów audio',
    },
  },
  en: {
    stats: {
      title: 'Statistics',
      routes: 'Monitoring Routes',
      sensors: 'Sensors',
      detections: 'Active Detections',
    },
    detectionPanel: {
      title: 'Drone Detections',
      noDetections: 'No active detections',
      sensor: 'Sensor',
      route: 'Route',
      intensity: 'Intensity',
      frequency: 'Frequency',
    },
    deploymentDialog: {
      title: 'New Sensor Deployment',
      subtitle: 'Select start and end points on the map',
      startPoint: 'Start Point',
      endPoint: 'End Point',
      clickMap: 'Click on map to select',
      distance: 'Distance',
      sensorSpacing: 'Sensor Spacing',
      spacingOptions: {
        dense: 'Dense',
        standard: 'Standard',
        wide: 'Wide',
      },
      costRange: 'Estimated Cost',
      weightRange: 'Equipment Weight',
      sensorsCount: 'Number of Sensors',
      deploy: 'Deploy',
      cancel: 'Cancel',
    },
    map: {
      fullscreen: 'Fullscreen',
      exitFullscreen: 'Exit Fullscreen',
      searchPlaceholder: 'Search for a place...',
      searching: 'Searching...',
      noResults: 'No results',
      sensor: 'Sensor',
      route: 'Route',
      selectStart: 'Select start point',
      selectEnd: 'Select end point',
      drone: 'Drone',
      detection: 'Detection',
      planRoute: 'Plan Route',
      close: 'Close',
      reset: 'Reset',
      done: 'Done',
      plannedRoute: 'Planned Route',
      start: 'Start',
      end: 'End',
      distance: 'Distance',
      editRoute: 'Edit Route',
      removeRoute: 'Remove Route',
    },
    language: {
      polish: 'Polish',
      english: 'English',
    },
    theme: {
      label: 'Theme:',
      light: 'Light',
      dark: 'Dark',
      auto: 'System',
      lightTitle: 'Light theme',
      darkTitle: 'Dark theme',
      autoTitle: 'System theme',
    },
    app: {
      title: 'EUDIS',
      subtitle: 'Drone monitoring system using audio sensors',
    },
  },
};