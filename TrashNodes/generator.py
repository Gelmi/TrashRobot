#!/usr/bin/env python3
"""
generator.py — RoMan Generator Node
=====================================
Subscreve /id (std_msgs/Int32).
Ao receber um ID, seleciona aleatoriamente 8 itens da tabela de lixo
(4 recicláveis, 4 não recicláveis) e publica cada um em /exp_data
(roman_msgs/ExpData), com box_index de 0 a 7.

Classificação das caixas:
  0–3 → justification     (recicláveis)
  4–7 → no_justification  (não recicláveis)
"""

import random
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Int32
# from roman_msgs.msg import ExpData


# ─────────────────────────────────────────────────────────────────────────────
# Tabela de itens
# ─────────────────────────────────────────────────────────────────────────────

RECYCLABLE_ITEMS = [
    # Easy
    {"item_name": "Plastic Bottle",  "ground_truth": "recyclable", "difficulty": "easy"},
    {"item_name": "Metal Can",       "ground_truth": "recyclable", "difficulty": "easy"},
    {"item_name": "Cardboard Box",   "ground_truth": "recyclable", "difficulty": "easy"},
    {"item_name": "Paper Sheet",     "ground_truth": "recyclable", "difficulty": "easy"},
    # {"item_name": "Glass Bottle",    "ground_truth": "recyclable", "difficulty": "easy"},
    # Hard
    {"item_name": "Tetra Pak",       "ground_truth": "recyclable", "difficulty": "hard"},
    {"item_name": "Plastic Bag",     "ground_truth": "recyclable", "difficulty": "hard"},
    {"item_name": "Blister Pack",    "ground_truth": "recyclable", "difficulty": "hard"},
    # {"item_name": "Aluminium Foil",  "ground_truth": "recyclable", "difficulty": "hard"},
    {"item_name": "Bubble Wrap",     "ground_truth": "recyclable", "difficulty": "hard"},
]

NON_RECYCLABLE_ITEMS = [
    # Easy
    {"item_name": "Paper Towel",     "ground_truth": "non-recyclable", "difficulty": "easy"},
    {"item_name": "Used Tissue",     "ground_truth": "non-recyclable", "difficulty": "easy"},
    {"item_name": "Surgical Mask",   "ground_truth": "non-recyclable", "difficulty": "easy"},
    # {"item_name": "Food Waste",      "ground_truth": "non-recyclable", "difficulty": "easy"},
    {"item_name": "Broken Ceramic",  "ground_truth": "non-recyclable", "difficulty": "easy"},
    # Hard
    {"item_name": "Black Plastic",         "ground_truth": "non-recyclable", "difficulty": "hard"},
    {"item_name": "Plasticized Paper Cup", "ground_truth": "non-recyclable", "difficulty": "hard"},
    {"item_name": "Waxed Cardboard",       "ground_truth": "non-recyclable", "difficulty": "hard"},
    {"item_name": "Foam",                  "ground_truth": "non-recyclable", "difficulty": "hard"},
    # {"item_name": "Wooden Packaging",      "ground_truth": "non-recyclable", "difficulty": "hard"},
]


def sample_experiment() -> list[dict]:
    """
    Seleciona 8 itens balanceados:
      - 4 recicláveis    (2 easy + 2 hard)
      - 4 não recicláveis (2 easy + 2 hard)

    Também define:
      - Ordem dos itens com restrições experimentais
      - Quais itens serão incorretos (apenas hard)
    """

    def balanced_sample(pool: list[dict]) -> list[dict]:
        easy = [i for i in pool if i["difficulty"] == "easy"]
        hard = [i for i in pool if i["difficulty"] == "hard"]
        selected = random.sample(easy, 2) + random.sample(hard, 2)
        random.shuffle(selected)
        return selected

    # Step 1 — Sample items
    recyclable_group = balanced_sample(RECYCLABLE_ITEMS)
    non_recyclable_group = balanced_sample(NON_RECYCLABLE_ITEMS)

    items = recyclable_group + non_recyclable_group
    # random.shuffle(items)  # initial shuffle

    # Step 2 — Separate by difficulty
    easy_items = [i for i in items if i["difficulty"] == "easy"]
    hard_items = [i for i in items if i["difficulty"] == "hard"]

    # Step 3 — Choose incorrect items (ONLY hard)
    incorrect_hard = random.sample(hard_items, 2)
    correct_hard = [i for i in hard_items if i not in incorrect_hard]

    # Step 4 — Define positions
    positions = list(range(8))

    # First error between positions 2–4 (0-based → trials 3–5)
    first_error_pos = random.choice([2, 3, 4])

    # Second error between positions 5–7 (trials 6–8)
    second_error_pos = random.choice([5, 6, 7])

    # Step 5 — Build ordered list with constraints
    experiment = [None] * 8

    # Assign incorrect items
    experiment[first_error_pos] = {**incorrect_hard[0], "is_correct": False}
    experiment[second_error_pos] = {**incorrect_hard[1], "is_correct": False}

    # Remaining positions
    remaining_positions = [i for i in positions if experiment[i] is None]

    # Combine remaining items
    remaining_items = easy_items + correct_hard
    random.shuffle(remaining_items)

    for pos, item in zip(remaining_positions, remaining_items):
        experiment[pos] = {**item, "is_correct": True}

    # Step 6 — Enforce first 2 trials always correct
    for i in [0, 1]:
        if not experiment[i]["is_correct"]:
            # swap with a later correct item
            for j in range(2, 8):
                if experiment[j]["is_correct"]:
                    experiment[i], experiment[j] = experiment[j], experiment[i]
                    break
    print(experiment)
    # Step 7 — Assign box_index
    for idx, item in enumerate(experiment):
        if item["ground_truth"] == "recyclable":
            # recyclable: 0–3
            item["box_index"] = idx % 4
        else:
            # non-recyclable: 4–7
            item["box_index"] = (idx % 4) + 4

    return experiment

#
# # ─────────────────────────────────────────────────────────────────────────────
# # Nó ROS2
# # ─────────────────────────────────────────────────────────────────────────────
#
# class GeneratorNode(Node):
#     def __init__(self):
#         super().__init__('generator_node')
#         self.get_logger().info('Generator node started. Waiting for /id...')
#
#         self.pub_exp = self.create_publisher(ExpData, '/exp_data', 10)
#         self.sub_id  = self.create_subscription(Int32, '/id', self._on_id_received, 10)
#
#         self._current_id: int | None = None
#         self._pending_timer = None  # referência ao timer ativo
#
#     def _on_id_received(self, msg: Int32):
#         pid = msg.data
#
#         if pid == self._current_id:
#             self.get_logger().warn(f'ID {pid} already active — ignoring.')
#             return
#
#         self._current_id = pid
#         self.get_logger().info(f'Received ID: {pid} — generating experiment...')
#
#         experiment = sample_experiment()
#         self._publish_next(experiment, index=0)
#
#     def _publish_next(self, experiment: list[dict], index: int):
#         """Publica um item e agenda o próximo com timer one-shot."""
#         if index >= len(experiment):
#             self.get_logger().info('All 8 items published. Sequence complete.')
#             return
#
#         # Publica item atual
#         item = experiment[index]
#         msg = ExpData()
#         msg.item_name    = item['item_name']
#         msg.ground_truth = item['ground_truth']
#         msg.difficulty   = item['difficulty']
#         msg.box_index    = item['box_index']
#         msg.classification = 'justification'
#         msg.suggestion = True
#
#         self.pub_exp.publish(msg)
#         self.get_logger().info(
#             f'  [{msg.box_index}] {msg.item_name} '
#             f'({msg.difficulty}, {msg.ground_truth})'
#         )
#
#         # Cria timer one-shot: cancela a si mesmo após disparar
#         def fire():
#             self._pending_timer.cancel()
#             self._pending_timer = None
#             self._publish_next(experiment, index + 1)
#
#         self._pending_timer = self.create_timer(0.15, fire)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Entrypoint
# # ─────────────────────────────────────────────────────────────────────────────
#
# def main(args=None):
#     rclpy.init(args=args)
#     node = GeneratorNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
#
#
# if __name__ == '__main__':
#     main()
