import unittest

from rewards.math_rule_reward import extract_answer, math_rule_reward


class RewardTests(unittest.TestCase):
    def test_boxed_answer(self):
        self.assertEqual(extract_answer(r"work \\boxed{42}"), "42")

    def test_numeric_normalization(self):
        self.assertEqual(math_rule_reward("The answer is 1,024.", "#### 1024"), 1.0)

    def test_wrong_answer(self):
        self.assertEqual(math_rule_reward("The answer is 3", "#### 4"), 0.0)


if __name__ == "__main__":
    unittest.main()
