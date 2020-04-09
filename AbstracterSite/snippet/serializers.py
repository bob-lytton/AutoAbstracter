from rest_framework import serializers
from snippet.models import Snippet

# import the summarize_model
import sys
import os
sys.path.append(os.path.realpath('../'))    # this is related to the dir where the start server command is run
from summarize_model import summarize

class SnippetSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    abstract = serializers.CharField(required=False, allow_blank=True)
    content = serializers.CharField()

    def create(self, validated_data):
        """
        Create and return a new `Snippet` instance, given the validated data.
        """
        # print("debug:", type(validated_data))   # class dict
        # print("debug:", validated_data)
        content = validated_data.get('content', '')
        # print("debug:", content)
        abstract = summarize.summarize(content)
        data = {'content':content, 'abstract':abstract}
        return Snippet.objects.create(**data)

    def update(self, instance, validated_data):
        """
        Update and return an existing `Snippet` instance, given the validated data.
        """
        instance.content = validated_data.get('content', instance.content)
        instance.abstract = summarize.summarize(instance.content)
        instance.save()
        return instance

