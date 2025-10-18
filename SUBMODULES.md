# Submodules Guide - EUDIS

## Czym są Git Submodules?

Git Submodules pozwalają na osadzenie innego repozytorium Git jako podfoldera w twoim projekcie. W naszym przypadku:

```
EUDIS/
└── Microcontroller/lib/
    └── EloquentTinyML/  ← oddzielne repozytorium Git
```

**Korzyści:**
- ✅ Nie przechowujemy obcego kodu w naszym repozytorium
- ✅ Łatwa aktualizacja biblioteki do nowych wersji
- ✅ Pozostajemy zsynchronizowani z oficjalnym repozytorium
- ✅ Czyste i małe repo (zamiast 1.9MB, tylko metadata)

## Jak Pracować z Submodułami

### 1. Pierwsze Klonowanie (Nowy Użytkownik)

```bash
# PRAWIDŁOWO - z submodułami
git clone --recurse-submodules https://github.com/FlyBozon/EUDIS.git

# LUB alternatywa (jeśli już sklonowałeś bez submodułów)
git clone https://github.com/FlyBozon/EUDIS.git
cd EUDIS
git submodule update --init --recursive
```

### 2. Wciągnięcie Zmian (git pull)

```bash
# Normalnie
git pull

# JEŚLI podmoduł ma nowe commity
git pull --recurse-submodules
```

Lub odrazu skonfiguruj by zawsze pobierał submodules:
```bash
git config submodule.recurse true
```

### 3. Aktualizacja Submodułu do Najnowszej Wersji

```bash
# Wejdź do folderu submodułu
cd Microcontroller/lib/EloquentTinyML

# Pobierz najnowszą wersję
git fetch
git merge origin/master  # lub origin/main w zależności od brancha

# Wróć do głównego repo
cd ../../..

# Zacommituj zaktualizowaną wersję
git add Microcontroller/lib/EloquentTinyML
git commit -m "Update EloquentTinyML to latest version"
git push
```

### 4. Zmiana Wersji Submodułu na Starszą/Nowszą

```bash
cd Microcontroller/lib/EloquentTinyML

# Pokaż dostępne tagi
git tag

# Przejdź na konkretny tag
git checkout v2.1.0

# Wróć i zacommituj
cd ../../..
git add Microcontroller/lib/EloquentTinyML
git commit -m "Pin EloquentTinyML to v2.1.0"
git push
```

## ❌ Częste Problemy i Rozwiązania

### Problem: "fatal: No submodule mapping found in .gitmodules"

```bash
# Rozwiązanie - inicjuj podmodule
git submodule update --init --recursive
```

### Problem: Folder EloquentTinyML jest pusty

```bash
# Rozwiązanie
git submodule update --init --recursive
# lub
cd Microcontroller/lib/EloquentTinyML
git fetch
cd ../../..
```

### Problem: "dirty submodule"

Submoduł pokazuje zmiany w `git status`:

```bash
cd Microcontroller/lib/EloquentTinyML
git status  # Sprawdź co się zmieniło
git checkout .  # Zresetuj na ostatni commit
cd ../../..
git add Microcontroller/lib/EloquentTinyML
git commit -m "Revert submodule changes"
```

### Problem: Chcę dodać do repo nikomu innego submodułu

```bash
# Dodaj nowy submoduł
git submodule add https://github.com/autor/biblioteka.git Microcontroller/lib/biblioteka

# Zacommituj
git add .gitmodules Microcontroller/lib/biblioteka
git commit -m "Add biblioteka as submodule"
git push
```

## 📊 Struktura po Poprawnym Sklonowaniu

```bash
cd EUDIS
ls -la  # Powinno być:

README.md                          # ← glavni README
.gitmodules                         # ← konfiguracja submodułów
Microcontroller/
├── src/main.cpp
├── lib/
│   └── EloquentTinyML/             # ← submoduł (pełny folder z .git)
│       ├── README.md
│       ├── src/
│       └── examples/
├── platformio.ini
└── SETUP.md
```

## 🔍 Sprawdzenie Status Submodułów

```bash
# Pokaż status wszystkich submodułów
git submodule status

# Szczegółowy raport
git submodule foreach git status
```

Output powinien wyglądać tak:
```
5d3636f8ab5eddb23691c79e2dd14dd5858f8be3 Microcontroller/lib/EloquentTinyML (heads/master)
```

## ✅ Checklist dla Deweloperów

- [ ] Klonuję z `--recurse-submodules`
- [ ] Po `git pull` wykonuję `git submodule update --recursive`
- [ ] Nie edytuję plików w `lib/EloquentTinyML` bezpośrednio
- [ ] Jeśli chcę zmienić wersję, robi to świadomie i commituje
- [ ] Sprawdzam `.gitmodules` przed commitowaniem zmian

## 📚 Dodatkowe Zasoby

- [Git Submodules Official Docs](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Atlassian Git Submodules Tutorial](https://www.atlassian.com/git/tutorials/git-submodule)
- [EloquentTinyML Repository](https://github.com/eloquentarduino/EloquentTinyML)

---

**Ostatnia aktualizacja**: Październik 2025
