# movie/management/commands/view_embeddings.py

from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = 'Muestra los embeddings de todas las películas'

    def handle(self, *args, **kwargs):
        movies_with_embeddings = Movie.objects.exclude(embedding__isnull=True)

        if not movies_with_embeddings.exists():
            self.stdout.write(self.style.WARNING("⚠️ No hay películas con embeddings guardados."))
            return

        for movie in movies_with_embeddings:
            self.stdout.write(self.style.SUCCESS(f"🎬 {movie.title}"))
            self.stdout.write(f"Embedding (primeros 5 valores): {movie.embedding[:5]}")
            self.stdout.write(f"Tamaño del embedding: {len(movie.embedding)}")
            self.stdout.write("-" * 50)
