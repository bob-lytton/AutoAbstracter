# Development Notes

1. Serialization & Deserialization:

    - To make an object be able to transmit on the network, the objected need to be serialized at first.
    - Design a `Serializer` Class in `snippet.serializers`.
    - Serialization

      ```python
      serializer = SnippetSerializer(snippet)
      content = JSONRenderer().render(serializer.data)  # b'{"id":2,"content":"Hello World."}'
      ```

    - Deserialization

      ```python
      import io

      stream = io.BytesIO(content)
      data = JSONParser().parse(stream) # {'id': 2, 'content': 'Hello World.'}, a dict
      ```

2. View:

    - A view function, or view for short, is a Python function that takes a Web request and returns a Web response.
    - Core of the back-end.

3. Url:

    - Each app needs an url.
    - Edit the url of the app in `app/urls.py`, and register the app url in the project's `proj/urls.py`

4. Call NLP Model:

    - Here is how the back-end works with the NLP model.
    - It receives a snippet of an article, and stores it in `content` field of an `Snippet` object.
    - Then calls the NLP model to generate the `abstract` field of the `Snippet` object.
    - Then the back-end returns the `Snippet` object.
    - Here is the process of HTTP Request and Response:
      - Request: a json includes `content` field.
      - Generate the summary of the `content`, and regard it as the `abstract`.
      - Response: a `Snippet` includes `content` and `abstract` fields.






