# 📊 StockFetcher – Historical Options Data + Stock Data Fetcher

A cross-platform desktop application to fetch and store **historical stock and options data** (with combined CSV output).  
Built with **Python, Tkinter, Pandas**, and **AlphaVantage API**.

## 🪟 Build Executable for Windows (.exe)

### Step 1: Install PyInstaller
```bash
pip install pyinstaller
```

### Step 2: Build Executable
Run the following command **from your project root**:
```bash
pyinstaller gui_app/main.py --name "StockFetcher" --onefile --noconsole --icon=icon.ico
```

💡 Note:
Sometimes, PyInstaller automatically includes unnecessary libraries even if they are not used in your app. so we need to exclude it to make it faster.
This single-line command excludes unnecessary AI/ML and visualization libraries (like TensorFlow, Torch, etc.) to make the build smaller and faster.

for example,
```bash
pyinstaller gui_app/main.py --name "StockFetcher" --onefile --noconsole --icon=icon.ico --clean --exclude-module matplotlib --exclude-module scipy --exclude-module tensorflow --exclude-module torch --exclude-module keras
```

This will generate:
```
dist/StockFetcher.exe
```

### Step 3: Run the App
Double-click on `dist/StockFetcher.exe`  
or run it via terminal:
```bash
dist\StockFetcher.exe
```

---

## 🍏 Build Runnable App for macOS (.app)

> ⚠️ **Note:** You must build the macOS version on a **Mac machine** (PyInstaller does not support cross-compiling from Windows to macOS).

### Step 1: Install PyInstaller
```bash
pip install pyinstaller
```

### Step 2: Build macOS App Bundle
```bash
pyinstaller gui_app/main.py --name "StockFetcher" --onefile --windowed --icon=icon.icns
```

This will generate:
```
dist/StockFetcher.app
```

### Step 3: Run the App
Open it directly or run:
```bash
open dist/StockFetcher.app
```
