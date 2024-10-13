from django.shortcuts import render
from django.views import View
from .forms import ReviewForm
import nltk
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

# Загрузка необходимых ресурсов nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Инициализация токенизатора и модели distilroberta-base
tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer")
model = RobertaForSequenceClassification.from_pretrained("./model")  # Убедитесь, что путь правильный
model.eval()  # Переводим модель в режим оценки

# Функция предсказания сентимента и рейтинга
def predict_sentiment_and_rating(text):
    # Токенизация входного текста
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Прогнозирование
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Применение софтмакс для получения вероятностей
    probabilities = torch.softmax(logits, dim=1)
    positive_prob = probabilities[0][1].item()  # Вероятность положительного класса
    
    # Определение сентимента
    sentiment = "Положительный" if positive_prob >= 0.5 else "Отрицательный"
    
    # Вычисление рейтинга от 1 до 10
    rating = int(positive_prob * 10) + 1  # Диапазон рейтинга от 1 до 10
    
    return sentiment, rating

# Представление для главной страницы
class IndexView(View):
    def get(self, request):
        form = ReviewForm()
        # Получение результатов из сессии
        sentiment = request.session.get('sentiment', None)
        rating = request.session.get('rating', None)
        context = {
            'form': form,
            'sentiment': sentiment,
            'rating': rating
        }
        return render(request, 'sentiment/index.html', context)

    def post(self, request):
        print("POST-запрос получен")  # Логирование
        form = ReviewForm(request.POST)
        if form.is_valid():
            print("Форма валидна")  # Логирование
            review_text = form.cleaned_data['review']
            print(f"Текст отзыва: {review_text}")  # Логирование
            
            try:
                # Предсказание сентимента и рейтинга
                sentiment, rating = predict_sentiment_and_rating(review_text)
                print(f"Сентимент: {sentiment}, Рейтинг: {rating}")  # Логирование

                # Сохранение результатов в сессии
                request.session['sentiment'] = sentiment
                request.session['rating'] = rating

                # Возвращаем результат на странице без перезагрузки
                context = {
                    'form': form,
                    'sentiment': sentiment,
                    'rating': str(rating)
                }
                print(f'CONTEXT:', end=' ')
                print(context)
                return render(request, 'sentiment/index.html', context)

            except Exception as e:
                print(f"Ошибка при обработке отзыва: {e}")  # Логирование ошибки
                # Выводим ошибку пользователю
                context = {
                    'form': form,
                    'error': "Произошла ошибка при обработке вашего отзыва."
                }
                return render(request, 'sentiment/index.html', context)

        # Если форма не валидна, выводим ошибки
        print("Форма не валидна")  # Логирование
        return render(request, 'sentiment/index.html', {'form': form})
