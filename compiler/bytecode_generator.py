# This file generates interpreter/src/bytecode.rs from main.py.
# It assumes that it is executed from the root of the project.
# This means we can safely write to `./interpreter/src/bytecode.rs`
from main import insns, insns_by_bytecode, Insn


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


# We just made this function ugly so the generated rust can be pretty printed
def box_inner(insn: Insn) -> str:
    # This doesn't need to be computed every time but speed doesn't matter here
    longest = max(map(len, ["LOAD64(value)"] + list(insns_by_bytecode))) + len("Insn::")
    match insn.bytecode:
        case "LOAD64":
            lhs_no_space = "Insn::LOAD64(value)"
            lhs = f"{lhs_no_space}{' ' * (longest - len(lhs_no_space))} => "
            rhs = f"(value << {hex(insn.shift)}) + {hex(insn.tag)}"
        case _:
            # return f"Insn::{insn.bytecode} =>
            lhs_no_space = f"Insn::{insn.bytecode}"
            lhs = f"{lhs_no_space}{' ' * (longest - len(lhs_no_space))} => "
            rhs = f"0x{hex(insn.tag)[2:].zfill(8)}"
    return lhs + rhs


def generate_rust_module(insn_list):
    text: list[str] = []
    # Enum bytecode definition
    text.append("#[derive(Debug, Clone, PartialEq)]\n")
    text.append("pub enum Insn {\n")
    for insn in insn_list:
        match insn.bytecode:
            case "LOAD64":
                text.append(f"\t{insn.bytecode}(u64),\n")
            case _:
                text.append(f"\t{insn.bytecode},\n")
    text.append("}\n\n")

    # Box function definition. Technically this is not needed for the interpreter, but it's nice to have
    text.append("pub fn box_insn(insn: Insn) -> u64 {\n\t")
    text.append("match insn {\n")
    for insn in insn_list:
        text.append(f"\t\t{box_inner(insn)},\n")
    text.append("\t}\n")
    text.append("}\n")

    # Unbox function definition
    text.append("pub fn unbox(qword: u64) -> Option<Insn> {\n\t")
    for insn in insn_list:
        text.append(
            f"if qword & 0x{hex(insn.mask)[2:].zfill(8)} == 0x{hex(insn.tag)[2:].zfill(8)} "
        )
        text.append(f"{{{unbox_inner(insn)}}}\n")
        text.append("\telse ")
    text.append("{None}\n")
    text.append("}")
    return "".join(text)


if __name__ == "__main__":
    with open("./interpreter/src/bytecode.rs", "w") as f:
        f.write(generate_rust_module(insns))
