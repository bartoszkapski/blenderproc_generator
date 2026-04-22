# Generator danych syntetycznych w Blenderze z wykorzystaniem BlenderProc


## 0 Testowany venv na Python 3.12

## 1 instalacja BlenderProc

<pre>
git clone https://github.com/DLR-RM/BlenderProc.git
cd BlenderProc
pip install -e .
blenderproc quickstart
</pre>

## 1.1 BlenderKit module (nie jest wymagany)
download .zip from https://www.blenderkit.com/get-blenderkit/ \
drag & drop do uruchomionego blendera

## 2 Instalacja bibliotek
<pre>
cd ..
pip install -r requirements.txt
</pre>

## 3 Dataset
Linki do pobrania wszystkich wykorzystancyh (darmowych) zasobów:
<pre>
https://drive.google.com/drive/folders/1LTFbeolLsF4mZkhMGEZzKU-oazE2G4Rr
https://drive.google.com/drive/folders/12tkcvaEEtRqmUBiLtu5l1I-jc_K2JMAh?hl=PL
</pre>

umieścić je w lokalizacji: \
BlenderProc/resources/ \
W razie potrzeby dostosować ścieżki w pliku config.yaml

## 4 Pliki źródłowe
dataset_generator/optimalized_generator/ \
znajdują się tutaj:
- config.yaml - wybrane parametry generatora
- generate_data_v3.py - główny kod generatora (rgb + surowa głębia)
- part_scenes.py - pliki źródłowe od geenrowania scen
- part_render.py - pliki źródłowe od renderu
- part_physics.py - pliki źródłowe od symulacji fizyki
- generate_real_depth_thread.py - post-processing obrazów głębi

## 5 Przykładowe uruchomienie
Pełne działanie generatora wraz z post-processingiem zdjęć głębi
<pre>
blenderproc run generate_data_v3.py --seed 42 --num_samples 5 --num_repeats 3 --physics --post-process 4
</pre>

Tryb debug pozwalający na podgląd generowanych scen (nie uruchamia renderu) \
Otworzy się okno Blendera i trzeba w nim ręcznie uruchomić skrypt przez przycisk "Run BlenderProc" bezpośrednio nad podglądem kodu
<pre>
blenderproc debug generate_data_v3.py --seed 42 --num_samples 1 --num_repeats 1 --debug
</pre>