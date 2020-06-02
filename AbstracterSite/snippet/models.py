from django.db import models

# Create your models here.
class Snippet(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    content = models.TextField()
    abstract = models.TextField()
    rouge = models.TextField()

    class Meta:
        ordering = ['created']

