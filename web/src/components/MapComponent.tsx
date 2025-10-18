import { useEffect, useRef, useState } from 'react';
import * as L from 'leaflet';
import { useAppStore } from '@/store/index';
import './styles/map.css';

export const MapComponent = () => {
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isSelectingMode, setIsSelectingMode] = useState(false);
  const [selectionStep, setSelectionStep] = useState<'start' | 'end' | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Array<{ name: string; lat: number; lon: number }>>([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedResultIndex, setSelectedResultIndex] = useState(0);
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchAbortRef = useRef<AbortController | null>(null);
  
  const deploymentLines = useAppStore((state) => state.deploymentLines);
  const espNodes = useAppStore((state) => state.espNodes);
  const detections = useAppStore((state) => state.detections);
  const dronePositions = useAppStore((state) => state.dronePositions);
  const missionStartPoint = useAppStore((state) => state.missionStartPoint);
  const missionEndPoint = useAppStore((state) => state.missionEndPoint);
  const setMissionStartPoint = useAppStore((state) => state.setMissionStartPoint);
  const setMissionEndPoint = useAppStore((state) => state.setMissionEndPoint);
  
  const layerRefsRef = useRef<{ [key: string]: L.Polyline | L.CircleMarker | L.Marker | L.Circle | L.DivIcon }>({});

  // Funkcja do wyszukiwania lokalizacji
  const handleSearch = async (query: string) => {
    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current);
    }
    if (query.length < 2) {
      setSearchResults([]);
      setShowSearchResults(false);
      setSelectedResultIndex(0);
      return;
    }

    searchDebounceRef.current = setTimeout(async () => {
      try {
        // cancel previous request if any
        if (searchAbortRef.current) {
          searchAbortRef.current.abort();
        }
        const controller = new AbortController();
        searchAbortRef.current = controller;

        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&countrycodes=pl&limit=10`,
          { signal: controller.signal }
        );
        const data = await response.json();
        
        // Usuń duplikaty (te same koordynaty) - często są 2 źródła w OSM
        const uniqueResults = data.filter((item: any, index: number, self: any[]) => 
          index === self.findIndex((t) => 
            Math.abs(parseFloat(t.lat) - parseFloat(item.lat)) < 0.0001 && 
            Math.abs(parseFloat(t.lon) - parseFloat(item.lon)) < 0.0001
          )
        ).slice(0, 5); // Ogranicz do 5 po usunięciu duplikatów
        
        setSearchResults(uniqueResults);
        setShowSearchResults(true);
        setSelectedResultIndex(0); // Automatycznie zaznacz pierwszy wynik
      } catch (error) {
        if ((error as any)?.name !== 'AbortError') {
          console.error('Search failed:', error);
          setSearchResults([]);
        }
      }
    }, 500);
  };

  // Cleanup pending debounce/requests on unmount
  useEffect(() => {
    return () => {
      if (searchDebounceRef.current) {
        clearTimeout(searchDebounceRef.current);
      }
      if (searchAbortRef.current) {
        searchAbortRef.current.abort();
      }
    };
  }, []);

  // Funkcja do zbliżenia do wybranej lokalizacji
  const handleLocationSelect = (lat: number, lon: number) => {
    if (mapRef.current) {
      mapRef.current.setView([lat, lon], 13);
      setSearchQuery('');
      setShowSearchResults(false);
      setSearchResults([]);
      setSelectedResultIndex(0);
    }
  };

  // Obsługa klawiatury dla wyników wyszukiwania
  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showSearchResults || searchResults.length === 0) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedResultIndex((prev) => (prev + 1) % searchResults.length);
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedResultIndex((prev) => (prev - 1 + searchResults.length) % searchResults.length);
        break;
      case 'Enter':
        e.preventDefault();
        const selected = searchResults[selectedResultIndex];
        if (selected) {
          handleLocationSelect(Number(selected.lat), Number(selected.lon));
        }
        break;
      case 'Escape':
        e.preventDefault();
        setShowSearchResults(false);
        setSelectedResultIndex(0);
        break;
    }
  };

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize map centered on Poland
    if (!mapRef.current) {
      mapRef.current = L.map(containerRef.current, {
        zoomControl: false, // Wyłączam domyślny zoom
      }).setView([52.0, 19.0], 6);

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19,
      }).addTo(mapRef.current);

      // Dodaję zoom kontrolkę w lewym dolnym rogu
      L.control.zoom({
        position: 'bottomleft',
      }).addTo(mapRef.current);
    }

    // Clear previous layers
    Object.values(layerRefsRef.current).forEach((layer) => {
      mapRef.current?.removeLayer(layer);
    });
    layerRefsRef.current = {};

    // Draw deployment lines
    deploymentLines.forEach((line) => {
      const polyline = L.polyline(
        [
          [line.startLat, line.startLng],
          [line.endLat, line.endLng],
        ],
        {
          color: '#ef4444',
          weight: 3,
          opacity: 0.8,
          dashArray: '5, 5',
        }
      ).addTo(mapRef.current!);

      layerRefsRef.current[`line-${line.id}`] = polyline;
    });

    // Draw ESP nodes
    espNodes.forEach((node) => {
      const circle = L.circleMarker([node.latitude, node.longitude], {
        radius: 6,
        fillColor: '#3b82f6',
        color: '#1e40af',
        weight: 2,
        opacity: 0.8,
        fillOpacity: 0.7,
      }).addTo(mapRef.current!);

      circle.bindPopup(`ESP Node: ${node.id}<br/>Line: ${node.lineId}`);
      layerRefsRef.current[`node-${node.id}`] = circle;
    });

    // Draw detections with animation
    detections.forEach((detection) => {
      const node = espNodes.find((n) => n.id === detection.lineId);
      if (node) {
        const circle = L.circleMarker([node.latitude, node.longitude], {
          radius: 10 + (detection.intensity / 10),
          fillColor: '#f59e0b',
          color: '#d97706',
          weight: 2,
          opacity: 0.8,
          fillOpacity: 0.6,
        }).addTo(mapRef.current!);

        circle.bindPopup(
          `Detection<br/>Intensity: ${detection.intensity}%<br/>Frequency: ${detection.frequency} dB`
        );
        layerRefsRef.current[`detection-${detection.id}`] = circle;
      }
    });

    // Draw mission planning points
    if (missionStartPoint) {
      const startMarker = L.marker(missionStartPoint, {
        icon: L.icon({
          iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
          shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
          iconSize: [25, 41],
          iconAnchor: [12, 41],
          popupAnchor: [1, -34],
          shadowSize: [41, 41],
        }),
      }).addTo(mapRef.current!);
      startMarker.bindPopup('Punkt startu');
      layerRefsRef.current[`mission-start`] = startMarker;
    }

    if (missionEndPoint) {
      const endMarker = L.marker(missionEndPoint, {
        icon: L.icon({
          iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
          shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
          iconSize: [25, 41],
          iconAnchor: [12, 41],
          popupAnchor: [1, -34],
          shadowSize: [41, 41],
        }),
      }).addTo(mapRef.current!);
      endMarker.bindPopup('Punkt końca');
      layerRefsRef.current[`mission-end`] = endMarker;
    }

    // Draw line between start and end points if both exist
    if (missionStartPoint && missionEndPoint) {
      const line = L.polyline([missionStartPoint, missionEndPoint], {
        color: '#10b981',
        weight: 2,
        opacity: 0.6,
        dashArray: '10, 5',
      }).addTo(mapRef.current!);
      layerRefsRef.current[`mission-line`] = line;
    }

    // Draw all drone positions
    dronePositions.forEach((position, index) => {
      // Zasięg wykrywania (400m)
      const detectionRangeCircle = L.circle(position, {
        radius: 400,
        fillColor: '#8b5cf6',
        color: '#7c3aed',
        weight: 1,
        opacity: 0.2,
        fillOpacity: 0.05,
      }).addTo(mapRef.current!);
      layerRefsRef.current[`drone-range-${index}`] = detectionRangeCircle;

      // Ikona drona
      const droneMarker = L.marker(position, {
        icon: L.divIcon({
          className: 'drone-marker',
          html: `<div style="
            width: 16px;
            height: 16px;
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
            border: 2px solid white;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(139, 92, 246, 0.6);
            position: relative;
            animation: pulse-drone-${index} 2s ease-in-out infinite;
          "></div>
          <style>
            @keyframes pulse-drone-${index} {
              0%, 100% { transform: scale(1); opacity: 1; }
              50% { transform: scale(1.2); opacity: 0.8; }
            }
          </style>`,
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        }),
        zIndexOffset: 1000,
      }).addTo(mapRef.current!);
      
      droneMarker.bindPopup(`🚁 Dron #${index + 1}<br/>Zasięg: 400m`);
      layerRefsRef.current[`drone-marker-${index}`] = droneMarker;
    });

    // Update cursor based on selection mode
    if (mapRef.current) {
      if (isSelectingMode) {
        mapRef.current.getContainer().style.cursor = selectionStep === 'start' 
          ? 'crosshair' 
          : 'crosshair';
      } else {
        mapRef.current.getContainer().style.cursor = 'grab';
      }
    }
  }, [deploymentLines, espNodes, detections, dronePositions, missionStartPoint, missionEndPoint, isSelectingMode, selectionStep]);

  // Handle map clicks for mission planning
  useEffect(() => {
    if (!mapRef.current || !isSelectingMode || !selectionStep) return;

    const handleMapClick = (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng;
      
      if (selectionStep === 'start') {
        setMissionStartPoint([lat, lng]);
        setSelectionStep('end');
      } else if (selectionStep === 'end') {
        setMissionEndPoint([lat, lng]);
        setIsSelectingMode(false);
        setSelectionStep(null);
      }
    };

    mapRef.current.on('click', handleMapClick);

    return () => {
      mapRef.current?.off('click', handleMapClick);
    };
  }, [isSelectingMode, selectionStep]);

  // Fullscreen handler
  const toggleFullscreen = () => {
    const mapContainer = containerRef.current?.parentElement;
    if (!mapContainer) return;

    if (!document.fullscreenElement) {
      mapContainer.requestFullscreen().then(() => {
        setIsFullscreen(true);
        // Wymuszenie ponownego renderowania mapy po zmianie rozmiaru
        setTimeout(() => {
          mapRef.current?.invalidateSize();
        }, 100);
      }).catch((err) => {
        console.error('Nie udało się włączyć fullscreen:', err);
      });
    } else {
      document.exitFullscreen().then(() => {
        setIsFullscreen(false);
        // Wymuszenie ponownego renderowania mapy po zmianie rozmiaru
        setTimeout(() => {
          mapRef.current?.invalidateSize();
        }, 100);
      });
    }
  };

  // Nasłuchiwanie na zmiany fullscreen (np. gdy użytkownik wciśnie ESC)
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
      // Wymuszenie ponownego renderowania mapy
      setTimeout(() => {
        mapRef.current?.invalidateSize();
      }, 100);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  return (
    <div className="relative w-full h-full">
      <div
        ref={containerRef}
        className="w-full h-full rounded-lg border border-border overflow-hidden"
      />
      
      {/* Search Bar - Top Left */}
      <div className="absolute top-4 left-4 z-40 w-72">
        <div className="relative">
          <div className="flex items-center bg-card border border-border rounded-lg shadow-lg px-3 py-2">
            <input
              type="text"
              placeholder="Szukaj lokalizacji..."
              value={searchQuery}
              onChange={(e) => {
                const value = e.target.value;
                setSearchQuery(value);
                handleSearch(value);
              }}
              onKeyDown={handleSearchKeyDown}
              onFocus={() => searchResults.length > 0 && setShowSearchResults(true)}
              className="flex-1 bg-transparent outline-none text-sm placeholder:text-muted-foreground"
            />
            {searchQuery && (
              <button
                onClick={() => {
                  setSearchQuery('');
                  setSearchResults([]);
                  setShowSearchResults(false);
                  setSelectedResultIndex(0);
                }}
                className="text-muted-foreground hover:text-foreground"
                aria-label="Wyczyść"
              >
                ×
              </button>
            )}
          </div>
          
          {/* Fullscreen Button */}
          <button
            onClick={toggleFullscreen}
            className="absolute -right-14 top-0 p-2 bg-card border border-border rounded-lg shadow-lg hover:bg-accent transition-colors"
            aria-label={isFullscreen ? 'Wyjdź z pełnego ekranu' : 'Pełny ekran'}
            title={isFullscreen ? 'Wyjdź z pełnego ekranu (ESC)' : 'Pełny ekran'}
          >
            {isFullscreen ? (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
              </svg>
            )}
          </button>

          {/* Search Results Dropdown */}
          {showSearchResults && searchResults.length > 0 && (
            <div className="absolute top-full mt-2 w-full bg-card border border-border rounded-lg shadow-lg overflow-hidden z-50">
              {searchResults.map((result, index) => (
                <button
                  key={index}
                  onClick={() => handleLocationSelect(Number(result.lat), Number(result.lon))}
                  className={`w-full px-4 py-2 text-left text-sm transition-colors border-b border-border last:border-b-0 ${
                    index === selectedResultIndex 
                      ? 'bg-accent font-medium' 
                      : 'hover:bg-accent/50'
                  }`}
                >
                  <div className="truncate">{result.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {Number(result.lat).toFixed(4)}, {Number(result.lon).toFixed(4)}
                  </div>
                </button>
              ))}
              <div className="px-4 py-1 text-xs text-muted-foreground bg-muted/30 border-t border-border">
                ↑↓ Nawiguj | Enter Wybierz | Esc Zamknij
              </div>
            </div>
          )}

          {showSearchResults && searchResults.length === 0 && searchQuery.length >= 2 && (
            <div className="absolute top-full mt-2 w-full bg-card border border-border rounded-lg shadow-lg p-3 z-50">
              <p className="text-xs text-muted-foreground text-center">Brak wyników</p>
            </div>
          )}
        </div>
      </div>

      {/* Main Control Button - Top Right */}
      {!isSelectingMode && !missionStartPoint && !missionEndPoint && (
        <div className="absolute top-4 right-4 z-40">
          <button
            onClick={() => {
              setIsSelectingMode(true);
              setSelectionStep('start');
            }}
            className="px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg shadow-lg hover:shadow-xl hover:from-blue-600 hover:to-blue-700 transition-all font-semibold"
          >
            Zaplanuj trasę
          </button>
        </div>
      )}

      {/* Stepper - Selection Mode */}
      {isSelectingMode && (
        <div className="absolute top-4 right-4 z-40">
          <div className="bg-card border border-border rounded-lg shadow-lg p-4 w-72">
            {/* Header */}
              <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-base">Zaplanuj trasę</h3>
              <button
                onClick={() => {
                  setIsSelectingMode(false);
                  setSelectionStep(null);
                }}
                className="text-muted-foreground hover:text-foreground text-xl leading-none"
                aria-label="Zamknij"
              >
                ×
              </button>
            </div>

            {/* Stepper Steps */}
            <div className="space-y-4">
              {/* Step 1 - Start Point */}
              <div className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center font-semibold transition-all ${
                    selectionStep === 'start' 
                      ? 'bg-green-500 text-white scale-110' 
                      : missionStartPoint
                      ? 'bg-green-500 text-white'
                      : 'bg-muted text-muted-foreground border border-border'
                  }`}>
                    {'1'}
                  </div>
                  {selectionStep === 'start' && <div className="w-0.5 h-12 bg-blue-400 mt-2" />}
                </div>
                <div className="flex-1 pt-1">
                  <div className="font-medium text-sm">Punkt startowy</div>
                  {missionStartPoint ? (
                    <p className="text-xs text-green-600 dark:text-green-400">
                      Wybrano: {missionStartPoint[0].toFixed(4)}°, {missionStartPoint[1].toFixed(4)}°
                    </p>
                  ) : selectionStep === 'start' ? (
                    <p className="text-xs text-blue-600 dark:text-blue-400 animate-pulse">
                      Kliknij na mapie aby wybrać punkt startowy
                    </p>
                  ) : (
                    <p className="text-xs text-muted-foreground">Wybierz punkt startu drona</p>
                  )}
                </div>
              </div>

              {/* Step 2 - End Point */}
              <div className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center font-semibold transition-all ${
                    selectionStep === 'end' 
                      ? 'bg-red-500 text-white scale-110' 
                      : missionEndPoint
                      ? 'bg-red-500 text-white'
                      : missionStartPoint
                      ? 'bg-muted text-muted-foreground border border-border'
                      : 'bg-gray-300 text-gray-500'
                  }`}>
                    {'2'}
                  </div>
                </div>
                <div className="flex-1 pt-1">
                  <div className="font-medium text-sm">Punkt końcowy</div>
                  {missionEndPoint ? (
                    <p className="text-xs text-red-600 dark:text-red-400">
                      Wybrano: {missionEndPoint[0].toFixed(4)}°, {missionEndPoint[1].toFixed(4)}°
                    </p>
                  ) : selectionStep === 'end' ? (
                    <p className="text-xs text-blue-600 dark:text-blue-400 animate-pulse">
                      Kliknij na mapie aby wybrać punkt końcowy
                    </p>
                  ) : missionStartPoint ? (
                    <p className="text-xs text-muted-foreground">Wybierz punkt końca drona</p>
                  ) : (
                    <p className="text-xs text-gray-400">Najpierw wybierz punkt startowy</p>
                  )}
                </div>
              </div>
            </div>

            {/* Distance Display */}
            {missionStartPoint && missionEndPoint && (
              <div className="mt-4 pt-4 border-t border-border">
                <div className="text-xs text-muted-foreground">Dystans:</div>
                <div className="text-lg font-semibold text-accent">
                  {Math.round(Math.sqrt(
                    Math.pow(missionEndPoint[0] - missionStartPoint[0], 2) +
                    Math.pow(missionEndPoint[1] - missionStartPoint[1], 2)
                  ) * 111000)} m
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="mt-4 flex gap-2">
              <button
                onClick={() => {
                  setMissionStartPoint(null);
                  setMissionEndPoint(null);
                  setSelectionStep('start');
                }}
                disabled={!missionStartPoint && !missionEndPoint}
                className="flex-1 px-3 py-2 text-sm rounded border border-input hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Resetuj
              </button>
              <button
                onClick={() => {
                  setIsSelectingMode(false);
                  setSelectionStep(null);
                }}
                className="flex-1 px-3 py-2 text-sm rounded bg-primary text-primary-foreground hover:bg-primary/90 transition-colors font-medium"
              >
                Gotowe
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Mission Points Summary - Bottom Left (when not in selection mode) */}
      {(missionStartPoint || missionEndPoint) && !isSelectingMode && (
        <div className="absolute bottom-4 left-4 z-40">
          <div className="bg-card border border-border rounded-lg shadow-lg p-3 space-y-2 w-72">
            <div className="flex items-center justify-between">
              <h4 className="font-semibold text-sm">Zaplanowana trasa</h4>
              <button
                onClick={() => {
                  setMissionStartPoint(null);
                  setMissionEndPoint(null);
                }}
                className="text-muted-foreground hover:text-foreground text-sm"
                aria-label="Usuń trasę"
              >
                ×
              </button>
            </div>

            {missionStartPoint && (
              <div className="flex items-center gap-2 text-xs">
                <span className="w-2.5 h-2.5 rounded-full bg-green-500 inline-block" aria-hidden />
                <div>
                  <div className="font-medium">Start</div>
                  <div className="text-muted-foreground">
                    {missionStartPoint[0].toFixed(4)}°, {missionStartPoint[1].toFixed(4)}°
                  </div>
                </div>
              </div>
            )}

            {missionEndPoint && (
              <div className="flex items-center gap-2 text-xs">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" aria-hidden />
                <div>
                  <div className="font-medium">Koniec</div>
                  <div className="text-muted-foreground">
                    {missionEndPoint[0].toFixed(4)}°, {missionEndPoint[1].toFixed(4)}°
                  </div>
                </div>
              </div>
            )}

            {missionStartPoint && missionEndPoint && (
              <div className="pt-2 border-t border-border text-xs">
                <div className="text-muted-foreground">Dystans:</div>
                <div className="font-semibold text-base text-accent">
                  {Math.round(Math.sqrt(
                    Math.pow(missionEndPoint[0] - missionStartPoint[0], 2) +
                    Math.pow(missionEndPoint[1] - missionStartPoint[1], 2)
                  ) * 111000)} m
                </div>
              </div>
            )}

            <button
              onClick={() => {
                setIsSelectingMode(true);
                setSelectionStep('start');
              }}
              className="w-full px-3 py-2 text-xs rounded border border-input hover:bg-accent transition-colors"
            >
              Edytuj trasę
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
