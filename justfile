test: test-compiler test-interpreter

gen-bytecode:
    @echo "Generating bytecode module"
    uv run python compiler/bytecode_generator.py

test-compiler:
    @echo "Running compiler tests"
    uv run python -m unittest compiler/main.py

test-interpreter: gen-bytecode
    @echo "Running interpreter tests"
    cd interpreter && cargo test
