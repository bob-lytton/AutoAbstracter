from rest_framework import serializers
from snippet.models import Snippet

# import the summarize_model
import sys
import os
# print(os.path.realpath('.'))
sys.path.append(os.path.realpath('.'))      # this is for debugging on vscode workspace
sys.path.append(os.path.realpath('../'))    # this is related to the dir where the start server command is run
from summarize_model import summarize

class SnippetSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    abstract = serializers.CharField(required=False, allow_blank=True)
    content = serializers.CharField()
    rouge = serializers.JSONField(required=False)
    max_length = serializers.IntegerField(read_only=True)
    min_length = serializers.IntegerField(read_only=True)
    num_beams = serializers.IntegerField(read_only=True)
    default_min_length = '60'   # because request.data automatically transforms the integers into list of strings
    default_max_length = '100'
    default_num_beams = '4'

    def create(self, validated_data):
        """
        Create and return a new `Snippet` instance, given the validated data.
        """
        print("debug:", validated_data.keys())
        print(validated_data)
        # TODO: May optimize by defining to_internal_value() method.
        content = validated_data.get('content', '')
        gold_summary = validated_data.get('gold_summary', None)
        if type(content) == list:
            content = str(content[0])
        if gold_summary:
            if type(gold_summary) == list:
                gold_summary = str(gold_summary[0])
        min_length = int(validated_data.get('min_length', self.default_min_length))
        max_length = int(validated_data.get('max_length', self.default_max_length))
        num_beams = int(validated_data.get('num_beams', self.default_num_beams))
        print("debug: min_length={}, max_length={}, num_beams={}".format(min_length, max_length, num_beams))

        rouge = ''
        if gold_summary:
            abstract, rouge = summarize.summarize(
                    content, 
                    gold_summary=gold_summary, 
                    min_length=min_length, 
                    max_length=max_length, 
                    num_beams=num_beams)
        else:
            abstract = summarize.summarize(
                    content, 
                    min_length=min_length, 
                    max_length=max_length, 
                    num_beams=num_beams)
        # abstract = "sorry i don't know."    # debug
        print("debug:", "type(rouge):", type(rouge), '\nrouge:', rouge)
        data = {'content':content, 'abstract':abstract, 'rouge':rouge}
        return Snippet.objects.create(**data)

    def update(self, instance, validated_data):
        """
        Update and return an existing `Snippet` instance, given the validated data.
        """
        instance.content = validated_data.get('content', instance.content)
        instance.abstract = summarize.summarize(instance.content)
        instance.save()
        return instance

