import re
from collections import deque
from typing import Deque, Dict, List, Optional, Set
from dataclasses import dataclass, asdict, field
import networkx

CONDITIONAL_JUMPS = (
	"beq", "bne", "blt", "bltu", "bge", "bgeu", "beqz", "bnez", "bltz", "blez", "bgtz", "bgez", "bgt",
	"bgtu", "ble", "bleu"
)

# Regex to extract operation from a line in the RISC-V Object Dump
OPERATION_REGEX = re.compile(r'\s*[0-9a-f]+:\t[0-9a-f]+\s+(?P<operation>\S+)')

# Regex to extract function labels from RISC-V Object Dump
FUNCTION_REGEX = re.compile(r'^(?P<address>[a-f0-9].+?)\s(?P<func><.*?>)')

# Regex to extract all relevant fields from RISC-V object dump
INSTRUCTION_REGEX = re.compile(
	r'\s*(?P<address>[0-9a-f]+):'
	r'\s*(?P<code>[0-9a-f]+)\s+'
	r'(?P<operation>[a-zA-Z_]+)'
	r'(\s+(?P<operands>[^\n#<]*))?'
	r'(\s*(?P<target><[^>]*>))?'
	r'(?P<comment>\s*#.*)?'
)


@dataclass
class Instruction:
	operation: str
	address: str = field(default=None)
	code: str = field(default=None)
	operands: List[str] = field(default=None)
	target: str = field(default=None)
	comment: str = None

	@staticmethod
	def parse_op(asm: str) -> Optional['Instruction']:
		if op_match := OPERATION_REGEX.match(asm):
			return Instruction(operation=op_match.group("operation"))

	@staticmethod
	def parse(asm: str) -> Optional['Instruction']:
		if match := INSTRUCTION_REGEX.match(asm):
			groups = match.groupdict()
			groups['operands'] = [v.strip() for v in groups['operands'].split(',')]
			return Instruction(**groups)


@dataclass
class BBNode:
	start_line: int
	end_line: int = None
	explored: bool = False
	note: str = None

	@property
	def id(self):
		return self.start_line

	def finish(self, line: int) -> 'BBNode':
		self.end_line = line
		self.explored = True
		return self


@dataclass
class BBJump:
	source: BBNode
	dest: BBNode
	line: int
	jump: Instruction
	branch_true: bool = True

	@property
	def readable_jump(self) -> str:
		return ("!" if not self.branch_true else "") + f"0x{self.jump.address}"

	def __hash__(self):
		return self.source.id + self.dest.id

	def __eq__(self, other: 'BBJump'):
		return self.source.id == other.source.id and self.dest.id == other.dest.id


@dataclass
class CFGExtractor:
	def __init__(self, *, function_name: str, binary_dump: str, binary_trace: str):
		self.function_name = function_name
		self.binary_dump = binary_dump.splitlines()
		self.binary_trace = binary_trace

		self._blocks: Dict[int, BBNode] = {}
		self._edges: Set[BBJump] = set()

		self._to_explore: Deque[BBNode] = deque()
		self._function_registry: Dict[str, int] = {}
		self._instruction_registry: Dict[str, int] = {}

		self._populate_indexes()

	def _populate_indexes(self):
		reached_code = False
		for line_num, contents in enumerate(self.binary_dump, 1):
			if contents == '':
				continue

			if match := FUNCTION_REGEX.match(contents):
				self._function_registry[match.group('func')] = line_num + 1  # starts on next line
				reached_code = True
				continue

			if reached_code:
				# Instructions start with prefixed [HEX]:
				inst_addr: str = contents.split(':')[0].strip()
				self._instruction_registry[inst_addr] = line_num

	def process_basic_block(self, block: BBNode):
		current_index = block.start_line - 1  # line numbers start at 1

		while self.binary_dump[current_index] != '':
			line = self.binary_dump[current_index]
			op = Instruction.parse_op(line)

			# If function ends, so does the basic block
			if op.operation == "ret":
				block.finish(current_index + 1)
				break
			elif op.operation in CONDITIONAL_JUMPS:
				instruction = Instruction.parse(line)
				block.finish(current_index)
				line_no = current_index + 1
				jump_target = self._locate_jump(instruction)
				jump_block = self._get_or_create_block(jump_target)

				if self.binary_dump[current_index + 2] != '':
					non_branch_block: BBNode = self._get_or_create_block(current_index + 2)
					self._edges.add(BBJump(block, non_branch_block, line_no, instruction, False))
					self._add_blocks_if_not_explored(non_branch_block)

				self._edges.add(BBJump(block, jump_block, line_no, instruction))
				self._add_blocks_if_not_explored(jump_block)

				break
			current_index += 1

	def _add_blocks_if_not_explored(self, *blocks: BBNode):
		for block in blocks:
			if not block.explored:
				self._to_explore.append(block)

	def _get_or_create_block(self, line_no: int, note: str = None) -> BBNode:
		if not self._blocks.get(line_no):
			self._blocks[line_no] = BBNode(start_line=line_no, note=note)
		return self._blocks[line_no]

	def _locate_jump(self, instruction: Instruction, operand_idx: int = -1) -> int:
		target = instruction.operands[operand_idx]

		if target not in self._instruction_registry:
			raise ValueError(f"Cannot locate the instruction for address {target} from {instruction}")

		start_line = self._instruction_registry[target]
		return start_line

	def construct_graph(self) -> networkx.DiGraph:
		cfg = networkx.DiGraph()
		for line in self._blocks:
			node = self._blocks[line]
			cfg.add_node(node.id, **asdict(node))

		for edge in self._edges:
			cfg.add_edge(
				edge.source.id,
				edge.dest.id,
				readable_jump=edge.readable_jump, **asdict(edge)
			)

		return cfg

	def extract(self, function: str) -> networkx.DiGraph:
		if not re.match(function, r'<.*?>'):
			function = f"<{function}>"

		start_line: int = self._function_registry.get(function)

		if not start_line:
			raise Exception(f"Function {function} could not be located in the assembly. "
							f"Found {self._function_registry}")

		# We will start DFS exploration with the main function
		root_node: BBNode = self._get_or_create_block(start_line, f"<{function}>")

		self._to_explore.append(root_node)

		while len(self._to_explore) > 0:
			block = self._to_explore.pop()
			self.process_basic_block(block)

		cfg: networkx.DiGraph = self.construct_graph()
		return cfg

