# Установка: pip install -r requirements.txt
import kagglehub
import os
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import torch
from datasets import Dataset

def print_progress(current, total, label):
    progress = current / total
    bar_length = 20
    filled = int(bar_length * progress)
    bar = '/' * filled + '-' * (bar_length - filled)
    sys.stdout.write(f"\r{label}: [{bar}] {int(progress * 100)}%")
    sys.stdout.flush()

def load_data():
    stages = 3
    print_progress(0, stages, "Загрузка приложения")
    path = kagglehub.dataset_download("kotartemiy/topic-labeled-news-dataset")
    print_progress(1, stages, "Загрузка приложения")
    csv_path = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")][0]
    print_progress(2, stages, "Загрузка приложения")
    data = pd.read_csv(csv_path, sep=';')
    print_progress(3, stages, "Загрузка приложения")
    print("\nДанные загружены")
    return data

def check_mentions(title, company_name):
    return company_name.lower() in title.lower()

def check_risk(title, alert_words):
    return any(word in title.lower() for word in alert_words)

def analyze_news_batch(data, company_name, classifier, alert_words):
    titles = data["title"].tolist()
    topics = data["topic"].tolist()
    dates = data["published_date"].tolist()
    total = len(titles)
    batch_size = 256
    results = []

    dataset = Dataset.from_dict({"text": titles})
    print("Анализ данных начат...")
    batch_results = classifier(dataset["text"], truncation=True, max_length=512, batch_size=batch_size)

    for i in range(0, total, batch_size):
        batch_subset = batch_results[i:i + batch_size]
        for j, result in enumerate(batch_subset):
            confidence = result["score"]
            label = result["label"]
            if confidence < 0.7:
                sentiment = "НЕЙТРАЛЬНЫЙ"
            else:
                sentiment = "ПОЗИТИВНЫЙ" if label == "POSITIVE" else "НЕГАТИВНЫЙ"
            title = titles[i + j]
            results.append({
                "title": title,
                "topic": topics[i + j],
                "sentiment": sentiment,
                "confidence": confidence,
                "mentions": check_mentions(title, company_name),
                "high_risk": check_risk(title, alert_words),
                "date": dates[i + j]
            })
        if (i + batch_size) % 1000 < batch_size or i + batch_size >= total:
            print_progress(min(i + batch_size, total), total, "Анализ данных")
    
    return results

def plot_results(data, results_df):
    fig = plt.figure(figsize=(20, 10))  # Стандартный размер окна, как раньше

    # Два графика сверху
    ax1 = plt.subplot(2, 2, 1)  # Первый ряд, первый столбец
    ax2 = plt.subplot(2, 2, 2)  # Первый ряд, второй столбец
    ax3 = plt.subplot(2, 1, 2)  # Второй ряд, полный размах

    # График 1: Топ-5 тем
    sns.countplot(y=data["topic"], order=data["topic"].value_counts().index[:5], palette="viridis", hue=data["topic"], legend=False, ax=ax1)
    ax1.set_title("Топ-5 тем новостей")
    ax1.set_xlabel("Количество")
    ax1.set_ylabel("Тема")

    # График 2: Тональность с рисками
    if not results_df.empty:
        sns.countplot(x="sentiment", hue="high_risk", data=results_df, palette="Set2", ax=ax2)
        ax2.set_title("Тональность (с учетом риска)")
        ax2.set_xlabel("Тональность")
        ax2.set_ylabel("Количество")
        ax2.legend(title="Высокий риск", labels=["Нет", "Да"])

    # График 3: Тренд сентимента по времени
    if not results_df.empty:
        results_df["date"] = pd.to_datetime(results_df["date"], errors="coerce")
        daily_sentiment = results_df.groupby(["date", "sentiment"]).size().unstack().fillna(0)
        daily_sentiment.rolling(window=14, min_periods=1).mean().plot(kind="line", ax=ax3, marker="o", markersize=4, linewidth=1.5)
        ax3.set_title("Тренд сентимента по времени")
        ax3.set_xlabel("Дата")
        ax3.set_ylabel("Количество статей")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend(title="Тональность")
        ax3.set_xlim(daily_sentiment.index.min(), daily_sentiment.index.max())

    plt.tight_layout()
    plt.show()  # Оставляем стандартное открытие окна

def main():
    print("GPU доступен:", torch.cuda.is_available())
    data = load_data()

    stages = 3
    print_progress(0, stages, "Инициализация модели")
    device = 0 if torch.cuda.is_available() else -1
    print_progress(1, stages, "Инициализация модели")
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
    print_progress(3, stages, "Инициализация модели")
    print("\nМодель загружена")
    alert_words = ["crisis", "scandal", "fraud", "bankrupt", "lawsuit", "failure", "delay"]

    company_name = input("Введите название компании: ")
    sample_size_input = input("Сколько строк анализировать? (по умолчанию весь датасет): ")
    sample_size = len(data) if not sample_size_input else int(sample_size_input)
    sample_data = data.head(sample_size)

    print(f"Анализ {sample_size} строк...")
    results = analyze_news_batch(sample_data, company_name, classifier, alert_words)
    print("\nАнализ завершён")

    results_df = pd.DataFrame(results)
    results_df.to_csv("mvp_results.csv", index=False)
    print("Результаты сохранены в 'mvp_results.csv'")

    plot_results(data, results_df)
    print("\nРаспределение тональности (%):")
    print(results_df["sentiment"].value_counts(normalize=True) * 100)
    print(f"Количество упоминаний компании: {results_df['mentions'].sum()}")
    if results_df["mentions"].sum() > 0:
        print("\nНовости с упоминаниями компании:")
        print(results_df[results_df["mentions"]][["title", "sentiment", "confidence"]])
    print(f"Новости с высоким риском: {results_df['high_risk'].sum()}")

if __name__ == "__main__":
    main()