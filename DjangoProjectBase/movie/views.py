from django.shortcuts import render
from django.http import HttpResponse

from .models import Movie

import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64

from openai import OpenAI
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv


def home(request):
    searchTerm = request.GET.get('searchMovie')
    if searchTerm:
        movies = Movie.objects.filter(title__icontains=searchTerm)
    else:
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm':searchTerm, 'movies':movies})


def about(request):
    return render(request, 'about.html')

def signup(request):
    email = request.GET.get('email') 
    return render(request, 'signup.html', {'email':email})


def statistics_view(request):
    matplotlib.use('Agg')
    all_movies = Movie.objects.all()
    movie_counts_by_year = {}
    for movie in all_movies:
        print(movie.genre)
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    year_graphic = generate_bar_chart(movie_counts_by_year, 'Year', 'Number of movies')

    movie_counts_by_genre = {}
    for movie in all_movies:
        genres = movie.genre.split(',')[0].strip() if movie.genre else "None"
        if genres in movie_counts_by_genre:
            movie_counts_by_genre[genres] += 1
        else:
            movie_counts_by_genre[genres] = 1

    genre_graphic = generate_bar_chart(movie_counts_by_genre, 'Genre', 'Number of movies')

    return render(request, 'statistics.html', {'year_graphic': year_graphic, 'genre_graphic': genre_graphic})


def generate_bar_chart(data, xlabel, ylabel):
    keys = [str(key) for key in data.keys()]
    plt.bar(keys, data.values())
    plt.title('Movies Distribution')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def movie_recommendation(request):
    recommended_movie = None
    similarity_score = None
    prompt = None
    error_message = None

    if request.method == 'GET' and request.GET.get('prompt'):
        prompt = request.GET.get('prompt').strip()

        if prompt:
            try:
                BASE_DIR = Path(__file__).resolve().parent.parent
                env_path = BASE_DIR / 'openAI.env'
                load_dotenv(dotenv_path=env_path, override=True)
                api_key = os.environ.get('openai_apikey')
                client = OpenAI(api_key=api_key)

                response = client.embeddings.create(
                    input=[prompt],
                    model="text-embedding-3-small"
                )
                prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

                best_movie = None
                max_similarity = -1

                for movie in Movie.objects.all():
                    if movie.emb:
                        movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
                        if movie_emb.shape == prompt_emb.shape:
                            similarity = cosine_similarity(prompt_emb, movie_emb)
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_movie = movie

                recommended_movie = best_movie
                similarity_score = round(float(max_similarity), 4) if max_similarity > -1 else None

            except Exception as e:
                error_message = f"Error al generar la recomendación: {str(e)}"

    return render(request, 'recommendation.html', {
        'recommended_movie': recommended_movie,
        'similarity_score': similarity_score,
        'prompt': prompt,
        'error_message': error_message,
    })