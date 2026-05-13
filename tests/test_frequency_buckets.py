"""Tests for the occurrence-weighted frequency-bucket fix (report 019 item 3).

The function used to bucket tokens by *rank* (top 25% of unique tokens
go to q1, etc.). On Zipfian corpora this is degenerate: the top 25% of
tokens by rank also account for ~99% of occurrences. Every test target
lands in q1. The fix: bucket by cumulative *occurrence mass*, so each
bucket holds ~25% of total token occurrences.
"""

from __future__ import annotations

import unittest

from energy_memory.phase2.metrics import build_frequency_buckets


class FrequencyBucketTests(unittest.TestCase):
    def test_empty_counts(self):
        self.assertEqual(build_frequency_buckets({}), {})

    def test_uniform_distribution(self):
        """8 tokens each appearing 10 times -> each quartile has 2 tokens."""
        counts = {f"t{i}": 10 for i in range(8)}
        buckets = build_frequency_buckets(counts)
        # Total mass = 80; each quartile = 20 = 2 tokens
        from collections import Counter
        bucket_sizes = Counter(buckets.values())
        self.assertEqual(bucket_sizes["q1_most_frequent"], 2)
        self.assertEqual(bucket_sizes["q2"], 2)
        self.assertEqual(bucket_sizes["q3"], 2)
        self.assertEqual(bucket_sizes["q4_least_frequent"], 2)

    def test_zipfian_distribution_concentrates_mass_at_top(self):
        """One dominant token + many tiny tokens -> dominant token alone
        takes q1; the rare tail is spread across the remaining buckets.

        Note: a dominant token whose mass spans multiple bucket-widths
        is assigned to the bucket where its cumulative-start falls (q1
        here). Buckets it spans through are then "consumed" — the rare
        tail picks up at whichever bucket comes after the dominant
        token's cumulative range.
        """
        counts = {"common": 1000}
        for i in range(999):
            counts[f"rare_{i}"] = 1
        buckets = build_frequency_buckets(counts)
        # The common token (50% of total mass) takes q1.
        self.assertEqual(buckets["common"], "q1_most_frequent")
        rare_buckets = {buckets[f"rare_{i}"] for i in range(999)}
        # The rare tail is *not* in q1 (the dominant token took it).
        self.assertNotIn("q1_most_frequent", rare_buckets)
        # The dominant token's mass (1000) already covers q1 + q2 ranges
        # (each 25% = 499.75 cumulative), so rare tokens pick up at q3.
        # The rare tail spans q3 and q4_least_frequent.
        self.assertEqual(rare_buckets, {"q3", "q4_least_frequent"})

    def test_fix_actually_changes_zipfian_outcome(self):
        """The pre-fix (rank-based) and post-fix (mass-based) behaviors
        produce visibly different bucket assignments on a Zipfian corpus.

        On 100 tokens with frequency 1/(rank+1), the *rank* quartile
        boundaries are at ranks 25, 50, 75. The *mass* quartile
        boundaries are far earlier because most mass sits at low ranks.
        """
        counts = {f"t{i}": int(1_000_000 / (i + 1)) for i in range(100)}
        buckets = build_frequency_buckets(counts)
        # On this distribution the first few tokens carry most mass, so
        # q1 should be a small number of tokens (much less than 25).
        from collections import Counter
        sizes = Counter(buckets.values())
        self.assertLess(sizes["q1_most_frequent"], 5)  # was 25 under rank
        # And q4 (rare tail) must contain many more tokens than q1 since
        # they each carry tiny mass.
        self.assertGreater(sizes["q4_least_frequent"], sizes["q1_most_frequent"])

    def test_every_token_assigned(self):
        counts = {"a": 5, "b": 3, "c": 1}
        buckets = build_frequency_buckets(counts)
        self.assertEqual(set(buckets.keys()), {"a", "b", "c"})

    def test_buckets_respect_ordering(self):
        """A more-frequent token is never in a worse-named bucket than a
        less-frequent one."""
        bucket_order = {
            "q1_most_frequent": 0, "q2": 1, "q3": 2, "q4_least_frequent": 3,
        }
        counts = {f"t{i}": 100 - i for i in range(40)}
        buckets = build_frequency_buckets(counts)
        ranked = sorted(counts.items(), key=lambda x: -x[1])
        for (tok_high, _), (tok_low, _) in zip(ranked, ranked[1:]):
            self.assertLessEqual(
                bucket_order[buckets[tok_high]],
                bucket_order[buckets[tok_low]],
                f"more-frequent token {tok_high} ended up in a later bucket "
                f"than less-frequent {tok_low}",
            )


if __name__ == "__main__":
    unittest.main()
