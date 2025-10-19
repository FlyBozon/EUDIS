# EUDIS - Drone Monitoring System

System monitorowania dronów z wykorzystaniem sieci sensorów audio (ESP32/ESP8266) rozproszczonych w linii, które wykrywają i lokalizują drony na podstawie dźwięku.

## Funkcjonalności

### 🗺️ Mapa interaktywna
- Wizualizacja linii wdrożeń sensorów
- Wyświetlanie pozycji węzłów ESP
- Wizualizacja wykryć w real-time
- **🔍 Wyszukiwarka lokalizacji** - szybkie przejście do wybranych miast i miejsc w Polsce
- **🎮 Kontrolki zoom** - umieszczone w lewym dolnym rogu

### 🚁 Planowanie misji
- Wysyłanie drona w wybrane miejsce
- Definiowanie linii zrzutu sensorów
- Konfiguracja rozstawu ESP (100m, 200m, 300m)
- **💰 Kalkulator kosztów** - szacunkowy koszt całkowity wdrożenia (PLN)
- **⚖️ Kalkulator wagi** - szacunkowa waga infrastruktury (kg)

### 📊 Monitorowanie
- Panel statystyk (liczba linii, węzłów, aktywnych wykryć)
- Widok wykryć w real-time z intensywnością i częstotliwością
- Mock data dla demonstracji

### 🎨 Motywy
- Jasny (light)
- Ciemny (dark)
- Automatyczny (zgodnie z ustawieniami systemu)

## Instalacja

```bash
# Zainstaluj zależności
npm install

# Uruchom serwer deweloperski
npm run dev

# Zbuduj dla produkcji
npm run build
```

## Stack technologiczny

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Shadcn/ui** - UI components
- **Leaflet** - Mapa
- **Zustand** - State management
- **Lucide React** - Ikony
- **Recharts** - Wykresy (przygotowane do rozszerzenia)

## Struktura projektu

```
src/
├── components/        # Komponenty React
│   ├── ui/           # Komponenty Shadcn
│   ├── MapComponent.tsx
│   ├── DeploymentDialog.tsx
│   ├── DetectionPanel.tsx
│   ├── StatsPanel.tsx
│   ├── ThemeSwitcher.tsx
│   └── DetectionSimulator.tsx
├── store/            # Zustand store
├── types/            # TypeScript types
├── lib/              # Utility functions
├── App.tsx
└── main.tsx
```

## Mock Data

Aktualnie system generuje losowe dane detektywne dla demonstracji:
- Każdy aktywny węzeł ESP ma szansę na generowanie detektywnego raz na sekundę
- Detektywne znikają po 5 sekundach
- Intensywność i częstotliwość są generowane losowo

## Dane kosztów i wagi dla rostawów

Kalkulator automatycznie oblicza koszty i wagę na podstawie:

| Rozstaw | Koszt (PLN/km) | Waga (kg/km) |
|---------|---|---|
| **ESP co 100m** | 800-950 | 4.5–5.5 |
| **ESP co 200m** | 600-750 | 3.5–4.5 |
| **ESP co 300m** | 530-600 | 3.0–4.0 |

Wartości są multipliko wane przez dystans trasy do uzyskania całkowitego szacunku.

## Implementowane funkcjonalności

- [x] Mapa interaktywna z Leaflet
- [x] Planowanie misji (wybór punktu startu i końca)
- [x] Wizualizacja linii wdrożeń i węzłów ESP
- [x] Konfiguracja rozstawu sensorów (100m, 200m, 300m)
- [x] **Wyszukiwarka lokalizacji** - OpenStreetMap Nominatim API
- [x] **Kalkulator kosztów i wagi** - szacunki dla różnych rostawów
- [x] Motywy jasny/ciemny
- [x] Panel statystyk
- [x] Panel wykryć
- [x] Symulator detektywnych

## Dalsze rozwoje

- [ ] Integracja z rzeczywistym źródłem danych
- [ ] Historia detektywnych
- [ ] Analityka i raporty
- [ ] Eksport danych
- [ ] Websockety do komunikacji real-time
- [ ] Kalibracja sensorów
- [ ] Triangulacja pozycji drona
- [ ] Pomiar rzeczywistego dystansu trasy (nie tylko aproksymacja)
- [ ] Eksport konfiguracji wdrożenia
- [ ] Planowanie tras optymalizujących pokrycie
