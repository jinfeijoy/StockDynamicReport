# This is a tutorial

Reference link: https://alexwlchan.net/2017/03/extensions-in-python-markdown/ 

Code blocks must be indented by 4 whitespaces.
Python-Markdown has a auto-guess function which works
pretty well:

    print("Hello, World")
    # some comment
    for letter in "this is a test":
        print(letter)

In cases where Python-Markdown has problems figuring out which
programming language we use, we can also add the language-tag
explicitly. One way to do this would be:


    :::python
    print("Hello, World")

or we can highlight certain lines to
draw the reader's attention:


    :::python hl_lines="1 5"
    print("highlight me!")
    # but not me!
    for letter in "this is a test":
    print(letter)
    # I want to be highlighted, too!

This is a GFM fenced code block:

```python
def hello_world(name):
    """Prints a friendly greeting."""
    print('Hello %s!' % name)
```

This GFM fenced code block would have line numbers:

```python linenums
def hello_world(name):
    """Prints a friendly greeting."""
    print('Hello %s!' % name)
```