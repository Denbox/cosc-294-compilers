test: test-compiler test-interpreter

test-compiler:
    @echo "Running compiler tests"
    uv run python -m unittest compiler/main.py

test-interpreter:
    @echo "Running interpreter tests"
    cd interpreter && cargo test
