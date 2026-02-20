# This file generates interpreter/src/bytecode.rs from main.py.
# It assumes that it is executed from the root of the project.
# This means we can safely write to `./interpreter/src/bytecode.rs`
from main import insns, Insn


# After unboxing, find the Insn label
def index_to_insn(index: int, arity: int) -> Insn:
    return list(filter(lambda x: x.arity == arity, insns))[index]


# This presumes that qword and insn are already matched
# This is for rust generation
# TODO: Fix pretty printing of integer values to be in hex representation
def unbox_inner(insn: Insn) -> str:
    match insn.bytecode:
        case "LOAD64":
            return f"Some(LOAD64(qword << {insn.shift}))"
        case _:
            return f"Some(Insn::{insn.bytecode})"


def generate_rust_module(insn_list):
    lines: list[str] = []
    # Enum bytecode definition
    lines.append("pub enum Insn {")
    for insn in insn_list:
        lines.append(f"\t{insn.bytecode},")
    lines.append("}")
    lines.append("")
    # Unbox function definition
    lines.append("pub fun unbox(qword: u64) -> Option<Insn> {")
    for insn in insn_list:
        lines.append(f"\t if qword & {insn.mask} == {insn.tag} {{")
        lines.append("\t\t" + unbox_inner(insn))
        lines.append("\t}")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    # with open("./interpreter/src/bytecode.rs", "w") as f:
    #     f.write(generate_rust_module(insns))
    print(generate_rust_module(insns))
