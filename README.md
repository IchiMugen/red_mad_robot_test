# MVP: Мониторинг репутации в СМИ  

## Описание  
Программа анализирует новости из датасета. "Topic Labeled News Dataset", определяет тональность(позитивная, негативная, нейтральная), 
ищет "упоминания компании" и выявляет "репутационные риски".
Работает быстро благодаря поддержке "GPU" и автоматически выбирает "GPU или CPU"в зависимости от доступности.  

---

## Требования  
- Python: 3.10 или 3.12 (протестировано)  
- Зависимости: См. `requirements.txt`
- CUDA (опционально): Для ускорения на GPU (версия 12.1–12.5)  

---

## Установка  

### Установите Python  
Загрузите и установите [Python](https://www.python.org/downloads/).  

### Запуск:

######

Windows
## Через скрипт
cd C:\путь\к\папке
.\setup.ps1
python mvp_analist_001.py

## Вручную

cd C:\путь\к\папке
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
python mvp_analist_002.py

######

Linux/macOS

## Через скрипт

cd /путь/к/папке
chmod +x setup.sh
./setup.sh
python3 mvp_analist_002.py

##  Вручную

cd /путь/к/папке
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
python3 mvp_analist_002.py

## Использование:
- Запустите программу.
- Введите название компании.
- Укажите количество строк для анализа (Enter = весь датасет).
- Дождитесь завершения анализа (~2–4 минуты на GPU).
- Графики откроются в одном окне; после закрытия появятся результаты.