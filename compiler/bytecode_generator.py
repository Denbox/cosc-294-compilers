# This file generates interpreter/src/bytecode.rs from main.py.
# It assumes that it is executed from the root of the project.
# This means we can safely write to `./interpreter/src/bytecode.rs`
from main import insns, Insn


# After unboxing, find the Insn label
def index_to_insn(index: int, arity: int) -> Insn:
    return list(filter(lambda x: x.arity == arity, insns))[index]


# This presumes that qword and insn are already matched
# This is for rust generation
def unbox_inner(insn: Insn) -> str:
    match insn.bytecode:
        case "LOAD64":
            return f"Some(Insn::LOAD64(qword >> {hex(insn.shift)}))"
        case _:
            return f"Some(Insn::{insn.bytecode})"


def generate_rust_module(insn_list):
    text: list[str] = []
    # Enum bytecode definition
    text.append("#[derive(Debug, Clone)]\n")
    text.append("pub enum Insn {\n")
    for insn in insn_list:
        match insn.bytecode:
            case "LOAD64":
                text.append(f"\t{insn.bytecode}(u64),\n")
            case _:
                text.append(f"\t{insn.bytecode},\n")
    text.append("}\n\n")

    # Unbox function definition
    text.append("pub fn unbox(qword: u64) -> Option<Insn> {\n\t")
    for index, insn in enumerate(insn_list):
        text.append(
            f"if qword & 0x{hex(insn.mask)[2:].zfill(8)} == 0x{hex(insn.tag)[2:].zfill(8)} {{\n"
        )
        text.append(f"\t\t{unbox_inner(insn)}\n")
        text.append("\t} else ")
    text.append("{\n\t\tNone\n\t}\n}")
    return "".join(text)


if __name__ == "__main__":
    with open("./interpreter/src/bytecode.rs", "w") as f:
        f.write(generate_rust_module(insns))
