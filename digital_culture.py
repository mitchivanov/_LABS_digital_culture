"""
Himalayan Expeditions Dataset Analysis
Анализ данных экспедиций в Гималаи с применением многомерных статистических методов
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import multivariate_normal
import warnings
import glob
import os
warnings.filterwarnings('ignore')

# Настройка стиля графиков
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

class HimalayanExpeditionsAnalyzer:
    """Анализатор данных гималайских экспедиций"""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.clean_data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_columns = []
        self.categorical_columns = []
        
        # Словарь переменных
        self.column_names = {
            'year': 'Год экспедиции',
            'smttime': 'Время на вершине (мин)',
            'smtdays': 'Дни до вершины',
            'totdays': 'Общая продолжительность (дни)',
            'highpoint': 'Наивысшая точка (м)',
            'camps': 'Количество лагерей',
            'rope': 'Веревка (м)',
            'totmembers': 'Всего участников',
            'smtmembers': 'Достигли вершины',
            'mdeaths': 'Смерти участников',
            'tothired': 'Всего нанятых',
            'smthired': 'Нанятые на вершине',
            'hdeaths': 'Смерти нанятых',
            'chksum': 'Контрольная сумма',
            'season': 'Сезон',
            'host': 'Страна-хозяйка',
            'nation': 'Национальность',
            'peakid': 'ID пика',
            'termreason': 'Причина завершения'
        }
        
    def get_readable_name(self, column):
        return self.column_names.get(column, column)
        
    def load_data(self, data_path=None):
        if data_path:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(data_path, encoding=encoding)
                    print(f"Успешно загружено с кодировкой: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print("Ошибка: не удалось прочитать файл")
                return False
        else:
            csv_files = glob.glob("*.csv")
            
            if not csv_files:
                print("CSV файлы не найдены. Запустите: python download_real_data.py")
                return False
                
            main_files = [f for f in csv_files if any(keyword in f.lower() 
                         for keyword in ['exped', 'members', 'expedition', 'climb'])]
            
            main_files = [f for f in main_files if 'dictionary' not in f.lower()]
            
            if main_files:
                if any('exped' in f.lower() for f in main_files):
                    data_path = [f for f in main_files if 'exped' in f.lower()][0]
                elif any('members' in f.lower() for f in main_files):
                    data_path = [f for f in main_files if 'members' in f.lower()][0]
                else:
                    data_path = main_files[0]
            else:
                non_dict_files = [f for f in csv_files if 'dictionary' not in f.lower()]
                data_path = non_dict_files[0] if non_dict_files else csv_files[0]
                
            print(f"Загружаем данные из файла: {data_path}")
            
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(data_path, encoding=encoding)
                    print(f"Успешно загружено с кодировкой: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print("Ошибка: не удалось прочитать файл")
                return False
        
        print(f"Данные загружены: {self.data.shape}")
        print(f"Колонки: {list(self.data.columns)}")
        
        self._identify_column_types()
        
        return True
    
    def _identify_column_types(self):
        self.numerical_columns = []
        self.categorical_columns = []
        
        for col in self.data.columns:
            if self.data[col].dtype in ['int64', 'float64']:
                if not any(keyword in col.lower() for keyword in ['id', 'index']):
                    self.numerical_columns.append(col)
            else:
                self.categorical_columns.append(col)
                
        print(f"Числовые колонки ({len(self.numerical_columns)}): {self.numerical_columns}")
        print(f"Категориальные колонки ({len(self.categorical_columns)}): {self.categorical_columns}")
        
    def explore_data(self):
        if self.data is None:
            print("Сначала загрузите данные")
            return
            
        print("=== ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ ===")
        print(f"Размер датасета: {self.data.shape}")
        print(f"\nИнформация о данных:")
        print(self.data.info())
        print(f"\nПервые 5 строк:")
        print(self.data.head())
        
        if self.numerical_columns:
            print(f"\nСтатистическое описание числовых колонок:")
            print(self.data[self.numerical_columns].describe())
        
        print(f"\nПропущенные значения:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        if self.categorical_columns:
            print(f"\nКатегориальные переменные:")
            for col in self.categorical_columns[:5]:
                print(f"\n{col}: {self.data[col].nunique()} уникальных значений")
                print(self.data[col].value_counts().head())
        
    def split_data(self, test_size=0.2, random_state=42):
        if self.data is None:
            print("Сначала загрузите данные")
            return
            
        stratify_col = None
        potential_target_cols = ['success', 'died', 'summited', 'outcome', 'result']
        
        for col in potential_target_cols:
            if col in self.data.columns:
                stratify_col = self.data[col]
                break
                
        try:
            self.train_data, self.test_data = train_test_split(
                self.data, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify_col
            )
        except:
            self.train_data, self.test_data = train_test_split(
                self.data, 
                test_size=test_size, 
                random_state=random_state
            )
        
        print(f"Train выборка: {self.train_data.shape}")
        print(f"Test выборка: {self.test_data.shape}")
        
    def detect_outliers(self, columns=None, contamination=0.1):
        if self.train_data is None:
            print("Сначала разделите данные на выборки")
            return
            
        if columns is None:
            columns = self.numerical_columns
        
        if not columns:
            print("Нет числовых колонок для анализа выбросов")
            return
            
        data_for_outliers = self.train_data[columns].dropna()
        
        if len(data_for_outliers) == 0:
            print("Нет данных для анализа выбросов")
            return
            
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(data_for_outliers)
        
        outlier_indices = data_for_outliers.index[outliers == -1]
        
        print(f"Обнаружено выбросов: {len(outlier_indices)} из {len(data_for_outliers)} записей")
        
        self.clean_data = self.train_data.drop(index=outlier_indices)
        
        print(f"Размер данных после очистки: {self.clean_data.shape}")
        
    def visualize_data(self, x_col=None, y_col=None, hue_col=None):
        if self.clean_data is None:
            data_to_plot = self.train_data
        else:
            data_to_plot = self.clean_data
            
        if data_to_plot is None:
            print("Нет данных для визуализации")
            return
            
        if x_col is None and len(self.numerical_columns) > 0:
            x_col = self.numerical_columns[0]
        if y_col is None and len(self.numerical_columns) > 1:
            y_col = self.numerical_columns[1]
        if hue_col is None and len(self.categorical_columns) > 0:
            hue_col = self.categorical_columns[0]
        
        if not x_col or not y_col:
            print("Недостаточно числовых колонок для построения scatter plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        if hue_col and hue_col in data_to_plot.columns:
            plot_data = data_to_plot[[x_col, y_col, hue_col]].dropna()
            top_categories = plot_data[hue_col].value_counts().head(10).index
            plot_data = plot_data[plot_data[hue_col].isin(top_categories)]
            sns.scatterplot(data=plot_data, x=x_col, y=y_col, hue=hue_col, alpha=0.7)
        else:
            plot_data = data_to_plot[[x_col, y_col]].dropna()
            sns.scatterplot(data=plot_data, x=x_col, y=y_col, alpha=0.7)
            
        x_label = self.get_readable_name(x_col)
        y_label = self.get_readable_name(y_col)
        
        plt.title(f'Точечный график: {y_label} vs {x_label}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def plot_histograms(self, columns=None, by_category=None):
        if self.clean_data is None:
            data_to_plot = self.train_data
        else:
            data_to_plot = self.clean_data
            
        if data_to_plot is None:
            print("Нет данных для построения гистограмм")
            return
            
        if columns is None:
            columns = self.numerical_columns[:9]
            
        num_cols = len(columns)
        if num_cols == 0:
            print("Нет числовых колонок для построения гистограмм")
            return
            
        cols_per_row = 3
        num_rows = (num_cols + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 5*num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1) if num_cols > 1 else [axes]
        
        for i, col in enumerate(columns):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            
            if num_rows == 1:
                ax = axes[col_idx] if num_cols > 1 else axes
            else:
                ax = axes[row, col_idx]
                
            data_col = data_to_plot[col].dropna()
            
            if by_category and by_category in data_to_plot.columns:
                categories = data_to_plot[by_category].value_counts().head(5).index
                for cat in categories:
                    cat_data = data_to_plot[data_to_plot[by_category] == cat][col].dropna()
                    if len(cat_data) > 0:
                        ax.hist(cat_data, alpha=0.6, label=str(cat), bins=30)
                ax.legend()
            else:
                ax.hist(data_col, bins=30, alpha=0.7, edgecolor='black')
                
            readable_name = self.get_readable_name(col)
            ax.set_title(f'{readable_name}')
            ax.set_xlabel(readable_name)
            ax.set_ylabel('Частота')
            
        for i in range(num_cols, num_rows * cols_per_row):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            if num_rows == 1:
                if num_cols > 1:
                    fig.delaxes(axes[col_idx])
            else:
                fig.delaxes(axes[row, col_idx])
                
        plt.tight_layout()
        plt.show()

    def explain_variables(self):
        print("=== ПЕРЕМЕННЫЕ HIMALAYAN EXPEDITIONS ===")
        
        print("\nВРЕМЕННЫЕ МЕТРИКИ:")
        print("• year - Год экспедиции (1905-2024)")
        print("• smttime - Время пребывания на вершине в минутах")
        print("• smtdays - Количество дней от базового лагеря до вершины")
        print("• totdays - Общая продолжительность экспедиции в днях")
        
        print("\nВЫСОТА И МАРШРУТ:")
        print("• highpoint - Наивысшая достигнутая точка в метрах")
        print("• camps - Количество установленных лагерей")
        print("• rope - Общая длина использованной веревки в метрах")
        
        print("\nУЧАСТНИКИ:")
        print("• totmembers - Общее количество участников экспедиции")
        print("• smtmembers - Количество участников, достигших вершины")
        print("• tothired - Общее количество нанятых помощников")
        print("• smthired - Количество нанятых, достигших вершины")
        
        print("\nБЕЗОПАСНОСТЬ:")
        print("• mdeaths - Количество смертей среди участников")
        print("• hdeaths - Количество смертей среди нанятых")
        
        if self.data is not None:
            print("\nСТАТИСТИКА:")
            if 'totmembers' in self.data.columns:
                members_stats = self.data['totmembers']
                print(f"• Размер команды: {members_stats.mean():.1f} человек (среднее)")
            if 'smtmembers' in self.data.columns:
                success_rate = (self.data['smtmembers'] > 0).mean() * 100
                print(f"• Успешность экспедиций: {success_rate:.1f}%")

class PointGenerator:
    """Генератор точек на основе многомерного нормального распределения"""
    
    def __init__(self):
        self.fitted = False
        self.means = {}
        self.covariances = {}
        self.categorical_probs = {}
        
    def fit(self, data, height_col=None, weight_col=None, gender_col=None):
        if height_col is None:
            height_candidates = [col for col in data.columns 
                               if any(keyword in col.lower() for keyword in 
                                    ['height', 'altitude', 'elevation', 'highpoint', 'year', 'camps'])]
            height_col = height_candidates[0] if height_candidates else None
            
        if weight_col is None:
            weight_candidates = [col for col in data.columns 
                               if any(keyword in col.lower() for keyword in 
                                    ['weight', 'mass', 'members', 'team_size', 'totmembers', 'days', 'totdays'])]
            weight_col = weight_candidates[0] if weight_candidates else None
            
        if gender_col is None:
            gender_candidates = [col for col in data.columns 
                               if any(keyword in col.lower() for keyword in 
                                    ['sex', 'gender', 'season', 'category', 'host', 'nation'])]
            gender_col = gender_candidates[0] if gender_candidates else None
            
        if not height_col or not weight_col:
            print("Не найдены подходящие колонки для height и weight")
            return False
            
        print(f"Используемые колонки: height={height_col}, weight={weight_col}, category={gender_col}")
        
        work_data = data[[height_col, weight_col, gender_col]].dropna() if gender_col else data[[height_col, weight_col]].dropna()
        
        if gender_col:
            categories = work_data[gender_col].unique()
            
            for category in categories:
                cat_data = work_data[work_data[gender_col] == category]
                if len(cat_data) > 1:
                    hw_data = cat_data[[height_col, weight_col]].values
                    
                    self.means[category] = np.mean(hw_data, axis=0)
                    self.covariances[category] = np.cov(hw_data.T)
                    
            category_counts = work_data[gender_col].value_counts()
            self.categorical_probs = (category_counts / len(work_data)).to_dict()
            
        else:
            hw_data = work_data[[height_col, weight_col]].values
            self.means['default'] = np.mean(hw_data, axis=0)
            self.covariances['default'] = np.cov(hw_data.T)
            self.categorical_probs = {'default': 1.0}
            
        self.height_col = height_col
        self.weight_col = weight_col
        self.gender_col = gender_col
        self.fitted = True
        
        print("Параметры модели:")
        for category in self.means:
            print(f"  {category}:")
            print(f"    Среднее: {self.means[category]}")
            print(f"    Ковариация:\n{self.covariances[category]}")
            
        return True
        
    def generate_points(self, n_points=100):
        if not self.fitted:
            print("Сначала обучите модель методом fit()")
            return None
            
        result = []
        
        for i in range(n_points):
            category = np.random.choice(
                list(self.categorical_probs.keys()),
                p=list(self.categorical_probs.values())
            )
            
            point = np.random.multivariate_normal(
                self.means[category], 
                self.covariances[category]
            )
            
            result.append({
                self.height_col: point[0],
                self.weight_col: point[1],
                self.gender_col if self.gender_col else 'category': category
            })
            
        return pd.DataFrame(result)
    
    def log_likelihood(self, data):
        if not self.fitted:
            print("Сначала обучите модель")
            return None
            
        if self.gender_col:
            work_data = data[[self.height_col, self.weight_col, self.gender_col]].dropna()
        else:
            work_data = data[[self.height_col, self.weight_col]].dropna()
            
        log_likelihood = 0
        processed_count = 0
        
        for _, row in work_data.iterrows():
            if self.gender_col:
                category = row[self.gender_col]
            else:
                category = 'default'
                
            if category in self.means:
                point = np.array([row[self.height_col], row[self.weight_col]])
                
                p_category = self.categorical_probs.get(category, 0)
                
                try:
                    p_point_given_category = multivariate_normal.pdf(
                        point, 
                        self.means[category], 
                        self.covariances[category]
                    )
                    
                    p_total = p_category * p_point_given_category
                    
                    if p_total > 0:
                        log_likelihood += np.log(p_total)
                        processed_count += 1
                        
                except Exception as e:
                    continue
                    
        return log_likelihood, processed_count
    
    def mean_log_likelihood(self, data):
        result = self.log_likelihood(data)
        if result is not None:
            log_likelihood, processed_count = result
            if processed_count > 0:
                mean_ll = log_likelihood / processed_count
                return mean_ll
            else:
                return 0
        return None

class GaussianMixtureAnalyzer:
    """Анализ с использованием смешанного Гауссова распределения"""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = None
        self.fitted = False
        
    def fit(self, data, height_col, weight_col):
        work_data = data[[height_col, weight_col]].dropna()
        
        if len(work_data) < self.n_components:
            print("Недостаточно данных для обучения")
            return False
            
        self.model = GaussianMixture(n_components=self.n_components, random_state=42)
        self.model.fit(work_data.values)
        self.fitted = True
        
        print(f"Модель обучена с {self.n_components} компонентами")
        print(f"Веса компонент: {self.model.weights_}")
        print(f"Средние значения:\n{self.model.means_}")
        
        return True
        
    def generate_points(self, n_points=100):
        if not self.fitted:
            print("Сначала обучите модель")
            return None
            
        samples, _ = self.model.sample(n_points)
        return samples
        
    def predict_proba(self, data, height_col, weight_col):
        if not self.fitted:
            print("Сначала обучите модель")
            return None
            
        work_data = data[[height_col, weight_col]].dropna()
        return self.model.predict_proba(work_data.values)

def main():
    print("=== АНАЛИЗ HIMALAYAN EXPEDITIONS DATASET ===")
    
    analyzer = HimalayanExpeditionsAnalyzer()
    
    if not analyzer.load_data():
        print("Не удалось загрузить данные")
        print("Сначала запустите: python download_real_data.py")
        return
        
    analyzer.explore_data()
    analyzer.split_data()
    analyzer.detect_outliers()
    
    print("\n=== ВИЗУАЛИЗАЦИЯ ДАННЫХ ===")
    analyzer.visualize_data()
    
    print("\n=== ГИСТОГРАММЫ ===")
    analyzer.plot_histograms()
    
    print("\n=== ГЕНЕРАТОР ТОЧЕК (МНОГОМЕРНОЕ НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ) ===")
    generator = PointGenerator()
    
    data_for_generator = analyzer.clean_data if analyzer.clean_data is not None else analyzer.train_data
    
    if generator.fit(data_for_generator):
        generated_points = generator.generate_points(n_points=50)
        print("Сгенерированные точки:")
        print(generated_points.head())
        
        train_likelihood = generator.mean_log_likelihood(data_for_generator)
        print(f"\nСредний логарифм правдоподобия на train: {train_likelihood:.3f}")
        
        if analyzer.test_data is not None:
            test_likelihood = generator.mean_log_likelihood(analyzer.test_data)
            print(f"Средний логарифм правдоподобия на test: {test_likelihood:.3f}")
            
        if generated_points is not None and len(analyzer.numerical_columns) >= 2:
            plt.figure(figsize=(15, 5))
            
            x_col = generator.height_col
            y_col = generator.weight_col
            
            plt.subplot(1, 3, 1)
            orig_data = data_for_generator[[x_col, y_col]].dropna()
            plt.scatter(orig_data[x_col], orig_data[y_col], alpha=0.6, label='Original')
            plt.xlabel(analyzer.get_readable_name(x_col))
            plt.ylabel(analyzer.get_readable_name(y_col))
            plt.title('Исходные данные')
            
            plt.subplot(1, 3, 2)
            gen_x_col = generated_points.columns[0]
            gen_y_col = generated_points.columns[1]
            plt.scatter(generated_points[gen_x_col], generated_points[gen_y_col], alpha=0.6, color='red', label='Generated')
            plt.xlabel(analyzer.get_readable_name(gen_x_col))
            plt.ylabel(analyzer.get_readable_name(gen_y_col))
            plt.title('Сгенерированные данные')
            
            plt.subplot(1, 3, 3)
            plt.scatter(orig_data[x_col], orig_data[y_col], alpha=0.6, label='Исходные')
            plt.scatter(generated_points[gen_x_col], generated_points[gen_y_col], alpha=0.6, color='red', label='Сгенерированные')
            plt.xlabel(analyzer.get_readable_name(x_col))
            plt.ylabel(analyzer.get_readable_name(y_col))
            plt.title('Сравнение')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
    
    print("\n=== СМЕШАННОЕ ГАУССОВО РАСПРЕДЕЛЕНИЕ ===")
    gmm_analyzer = GaussianMixtureAnalyzer(n_components=3)
    
    if len(analyzer.numerical_columns) >= 2:
        x_col = generator.height_col if generator.fitted else analyzer.numerical_columns[0]
        y_col = generator.weight_col if generator.fitted else analyzer.numerical_columns[1]
        
        if gmm_analyzer.fit(data_for_generator, x_col, y_col):
            gmm_points = gmm_analyzer.generate_points(n_points=100)
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            orig_data = data_for_generator[[x_col, y_col]].dropna()
            plt.scatter(orig_data[x_col], orig_data[y_col], alpha=0.6)
            plt.xlabel(analyzer.get_readable_name(x_col))
            plt.ylabel(analyzer.get_readable_name(y_col))
            plt.title('Исходные данные')
            
            plt.subplot(1, 3, 2)
            plt.scatter(gmm_points[:, 0], gmm_points[:, 1], alpha=0.6, color='green')
            plt.xlabel(analyzer.get_readable_name(x_col))
            plt.ylabel(analyzer.get_readable_name(y_col))
            plt.title('GMM сгенерированные')
            
            plt.subplot(1, 3, 3)
            plt.scatter(orig_data[x_col], orig_data[y_col], alpha=0.5, label='Исходные')
            plt.scatter(gmm_points[:, 0], gmm_points[:, 1], alpha=0.6, color='green', label='GMM')
            if generated_points is not None:
                plt.scatter(generated_points[generator.height_col], generated_points[generator.weight_col], alpha=0.6, color='red', label='Многомерное нормальное')
            plt.xlabel(analyzer.get_readable_name(x_col))
            plt.ylabel(analyzer.get_readable_name(y_col))
            plt.title('Сравнение всех методов')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            probabilities = gmm_analyzer.predict_proba(data_for_generator, x_col, y_col)
            print(f"Средние вероятности принадлежности к компонентам: {np.mean(probabilities, axis=0)}")
    
    print("\n=== АНАЛИЗ ЗАВЕРШЕН ===")

if __name__ == "__main__":
    main()
