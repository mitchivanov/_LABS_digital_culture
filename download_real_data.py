"""
Загрузка реального датасета Himalayan Expeditions с Kaggle
"""

import kagglehub
import pandas as pd
import os
import glob

def download_himalayan_data():
    """
    Загружает датасет Himalayan Expeditions с Kaggle
    """
    print("Загрузка датасета Himalayan Expeditions с Kaggle...")
    
    try:
        # Загружаем датасет
        path = kagglehub.dataset_download("siddharth0935/himalayan-expeditions")
        print(f"Данные загружены в: {path}")
        
        # Ищем CSV файлы в загруженной директории
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        if not csv_files:
            print("CSV файлы не найдены в загруженной директории")
            return None
            
        print(f"Найденные CSV файлы:")
        for i, file in enumerate(csv_files):
            print(f"  {i+1}. {os.path.basename(file)}")
            
        # Копируем основные файлы в рабочую директорию
        copied_files = []
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            dest_path = filename
            
            # Читаем и сохраняем файл в рабочую директорию
            df = pd.read_csv(csv_file)
            df.to_csv(dest_path, index=False)
            copied_files.append(dest_path)
            
            print(f"Файл скопирован: {dest_path}")
            print(f"  Размер: {df.shape}")
            print(f"  Колонки: {list(df.columns)}")
            print()
            
        return copied_files
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def explore_downloaded_data():
    """
    Исследует загруженные данные
    """
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("CSV файлы не найдены в текущей директории")
        return
        
    print("=== АНАЛИЗ ЗАГРУЖЕННЫХ ДАННЫХ ===")
    
    for csv_file in csv_files:
        print(f"\n--- Файл: {csv_file} ---")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"Размер: {df.shape}")
            print(f"Колонки: {list(df.columns)}")
            print(f"Типы данных:")
            print(df.dtypes)
            print(f"\nПервые 3 строки:")
            print(df.head(3))
            print(f"\nПропущенные значения:")
            print(df.isnull().sum()[df.isnull().sum() > 0])
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"Ошибка при чтении файла {csv_file}: {e}")

def main():
    """
    Основная функция
    """
    print("=== ЗАГРУЗКА РЕАЛЬНОГО ДАТАСЕТА HIMALAYAN EXPEDITIONS ===")
    
    # Загружаем данные
    files = download_himalayan_data()
    
    if files:
        print(f"\nУспешно загружено файлов: {len(files)}")
        
        # Исследуем загруженные данные
        explore_downloaded_data()
        
        print("\n=== ДАННЫЕ ГОТОВЫ К АНАЛИЗУ ===")
        print("Теперь можно запустить: python digital_culture.py")
    else:
        print("Не удалось загрузить данные")

if __name__ == "__main__":
    main() 