from __future__ import annotations

import unittest

import route_choice_lab as lab
import route_choice_lab_guarded_runner as guarded


class GuardedDecoderTests(unittest.TestCase):
    def candidate(self, key: str, nodes: list[int], texts: list[str]) -> lab.Candidate:
        edges = [lab.norm(a, b) for a, b in zip(nodes, nodes[1:])]
        return lab.Candidate(key, nodes, edges, evidence_texts=texts)

    def test_conflicting_direct_evidence_stays_unresolved(self) -> None:
        candidate = self.candidate(
            "a",
            [1, 2, 3],
            ["犬は泳ぐ", "鳥が空を飛ぶ", "犬は歩く"],
        )
        guarded.conservative_decode_candidate("犬は", candidate, [candidate])
        self.assertEqual(candidate.decoded_text, "未解釈の経路")
        self.assertEqual(candidate.decode_method, "direct-conflict")

    def test_single_consistent_direct_evidence_decodes(self) -> None:
        candidate = self.candidate(
            "a",
            [1, 2, 3],
            ["犬は泳ぐ"],
        )
        guarded.conservative_decode_candidate("犬は", candidate, [candidate])
        self.assertEqual(candidate.decoded_text, "泳ぐ")
        self.assertEqual(candidate.decode_method, "direct-consensus")

    def test_same_label_is_limited_but_distinct_routes_remain_possible(self) -> None:
        selected: list[lab.Candidate] = []
        signatures = set()
        counts: dict[str, int] = {}
        candidates = [
            self.candidate(str(index), [index, index + 10, index + 20], ["犬は泳ぐ"])
            for index in range(1, 5)
        ]
        for candidate in candidates:
            candidate.decoded_text = "泳ぐ"
            guarded._append_unique(candidate, selected, signatures, counts)
        self.assertEqual(len(selected), 2)
        self.assertNotEqual(
            guarded.research.structural_route_signature(selected[0]),
            guarded.research.structural_route_signature(selected[1]),
        )

    def test_unknown_label_is_limited_to_one(self) -> None:
        selected: list[lab.Candidate] = []
        signatures = set()
        counts: dict[str, int] = {}
        for index in range(1, 4):
            candidate = self.candidate(str(index), [index, index + 10, index + 20], [])
            candidate.decoded_text = "未解釈の経路"
            guarded._append_unique(candidate, selected, signatures, counts)
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
