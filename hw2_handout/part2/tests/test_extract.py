from mini_data_pipeline.pipeline import extract_text


def test_extract_utf8_and_remove_nonvisible_content():
    html = b"""
    <html><head><style>.x { color: red }</style></head>
    <body><h1>Visible title</h1><p>First   line &amp; text.</p>
    <script>hidden@example.com attack destroy threaten</script>
    <p>Second line.</p></body></html>
    """
    assert extract_text(html, encoding="utf-8") == (
        "Visible title\nFirst line & text.\nSecond line."
    )


def test_extract_cp1252():
    html = '<html><body><p>Un caf\xe9 \xe0 Paris.</p></body></html>'.encode("cp1252")
    assert extract_text(html, encoding="cp1252") == "Un café à Paris."


def test_extract_empty_page():
    html = b"<html><head><script>ignored()</script></head><body>  </body></html>"
    assert extract_text(html, encoding="utf-8") == ""


def test_extract_removes_empty_lines():
    html = (
        b"<html><body><pre>First line\r\n \t \r\n\r\nSecond line</pre>"
        b"<p>   </p><p>Third line</p></body></html>"
    )
    assert extract_text(html, encoding="utf-8") == (
        "First line\nSecond line\nThird line"
    )

